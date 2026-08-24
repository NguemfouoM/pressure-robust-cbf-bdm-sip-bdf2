from __future__ import annotations

import csv
import json
from pathlib import Path

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element, mixed_element
from dolfinx import cpp, fem, io, mesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector
from dolfinx.io import gmsh as gmshio


def _gmsh_mesh(name: str, lc: float, comm: MPI.Comm):
    """Create a conforming triangular L-domain or perforated enclosure."""
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add(name)
        occ = gmsh.model.occ
        if name == "l_shape":
            outer = occ.addRectangle(0.0, 0.0, 0.0, 2.0, 2.0)
            cut = occ.addRectangle(1.0, 1.0, 0.0, 1.0, 1.0)
            surfaces, _ = occ.cut([(2, outer)], [(2, cut)])
        elif name == "obstacle_enclosure":
            outer = occ.addRectangle(0.0, 0.0, 0.0, 3.0, 1.5)
            disks = [
                (2, occ.addDisk(1.05, 0.75, 0.0, 0.22, 0.22)),
                (2, occ.addDisk(1.95, 0.75, 0.0, 0.22, 0.22)),
            ]
            surfaces, _ = occ.cut([(2, outer)], disks)
        else:
            raise ValueError(f"Unknown complex domain: {name}")
        occ.synchronize()
        surface_tags = [tag for dim, tag in surfaces if dim == 2]
        gmsh.model.addPhysicalGroup(2, surface_tags, 1)
        boundary = gmsh.model.getBoundary([(2, tag) for tag in surface_tags], oriented=False)
        gmsh.model.addPhysicalGroup(1, sorted({tag for dim, tag in boundary if dim == 1}), 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.65 * lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    if comm.rank == 0:
        gmsh.finalize()
    return msh, cell_tags, facet_tags


def _scalar(form) -> float:
    local = fem.assemble_scalar(fem.form(form))
    return float(MPI.COMM_WORLD.allreduce(local, op=MPI.SUM))


def _heat_source(name: str, x, amplitude: float):
    if name == "l_shape":
        center = (0.42, 0.38)
        width = 32.0
    else:
        center = (0.38, 0.75)
        width = 22.0
    return amplitude * ufl.exp(-width * ((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2))


def _save_snapshot(msh, u_h, T_h, output: Path) -> None:
    """Store serial P1 visualization data without using it in diagnostics."""
    cell = msh.basix_cell()
    V = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(2,)))
    W = fem.functionspace(msh, element("Lagrange", cell, 1))
    uv = fem.Function(V)
    Tv = fem.Function(W)
    uv.interpolate(fem.Expression(u_h, V.element.interpolation_points))
    Tv.interpolate(fem.Expression(T_h, W.element.interpolation_points))
    uv.x.scatter_forward()
    Tv.x.scatter_forward()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Snapshot export is intentionally serial; run this benchmark with one MPI rank.")
    xy = W.tabulate_dof_coordinates()[:, :2].copy()
    velocity = uv.x.array.reshape((-1, 2)).copy()
    temperature = Tv.x.array.copy()
    cells = msh.geometry.dofmap[:, :3].copy()
    np.savez_compressed(output, xy=xy, cells=cells, velocity=velocity, temperature=temperature)


def solve_case(name: str, beta_f: float, cfg: dict, results: Path) -> dict:
    comm = MPI.COMM_WORLD
    msh, _, _ = _gmsh_mesh(name, float(cfg["mesh_size"]), comm)
    cell = msh.basix_cell()
    k = int(cfg["degree"])
    X = fem.functionspace(
        msh,
        mixed_element(
            [element("BDM", cell, k), element("DG", cell, k - 1), element("Lagrange", cell, k)]
        ),
    )
    state, old, older = fem.Function(X), fem.Function(X), fem.Function(X)
    u, p, T = ufl.split(state)
    u0, _, T0 = ufl.split(old)
    um1, _, Tm1 = ufl.split(older)
    v, z, S = ufl.TestFunctions(X)
    x = ufl.SpatialCoordinate(msh)
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    dx, ds, dS = ufl.dx(domain=msh), ufl.ds(domain=msh), ufl.dS(domain=msh)
    dt = fem.Constant(msh, PETSc.ScalarType(cfg["dt"]))
    eps = lambda w: ufl.sym(ufl.grad(w))
    ustar, Tstar = 2 * u0 - um1, 2 * T0 - Tm1
    # Backward Euler supplies the first history value; the coefficients are
    # then switched to BDF2 without rebuilding the nonlinear problem.
    a0 = fem.Constant(msh, PETSc.ScalarType(1.0))
    a1 = fem.Constant(msh, PETSc.ScalarType(-1.0))
    a2 = fem.Constant(msh, PETSc.ScalarType(0.0))
    bdf_u = (a0 * u + a1 * u0 + a2 * um1) / dt
    bdf_T = (a0 * T + a1 * T0 + a2 * Tm1) / dt
    nu, kappa = float(cfg["nu"]), float(cfg["kappa"])
    sigma = float(cfg["sip_penalty"]) * k * k

    viscous = ufl.inner(2 * nu * eps(u), eps(v)) * dx
    viscous += -ufl.inner(ufl.avg(2 * nu * eps(u)), ufl.outer(ufl.jump(v), n("+"))) * dS
    viscous += -ufl.inner(ufl.avg(2 * nu * eps(v)), ufl.outer(ufl.jump(u), n("+"))) * dS
    viscous += sigma * ufl.avg(nu / h) * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
    viscous += -ufl.inner(2 * nu * eps(u), ufl.outer(v, n)) * ds
    viscous += -ufl.inner(2 * nu * eps(v), ufl.outer(u, n)) * ds
    viscous += sigma * nu / h * ufl.inner(u, v) * ds

    un = ufl.dot(ustar("+"), n("+"))
    upwind = ufl.conditional(ufl.ge(un, 0), u("+"), u("-"))
    momentum_convection = -ufl.inner(ufl.outer(u, ustar), ufl.grad(v)) * dx
    momentum_convection += ufl.inner(un * upwind, ufl.jump(v)) * dS
    thermal_convection = 0.5 * (ufl.dot(u, ufl.grad(T)) * S - ufl.dot(u, ufl.grad(S)) * T) * dx
    speed = ufl.sqrt(ufl.inner(u, u) + 1.0e-24)
    gravity = ufl.as_vector(tuple(cfg["gravity"]))
    source = _heat_source(name, x, float(cfg["heat_amplitude"]))

    F = ufl.inner(bdf_u, v) * dx + viscous + momentum_convection
    F += float(cfg["alpha"]) * ufl.inner(u, v) * dx
    F += beta_f * speed ** (float(cfg["forchheimer_r"]) - 2.0) * ufl.inner(u, v) * dx
    F += -p * ufl.div(v) * dx + z * ufl.div(u) * dx
    F += -float(cfg["gamma"]) * T * ufl.inner(gravity, v) * dx
    F += bdf_T * S * dx + kappa * ufl.inner(ufl.grad(T), ufl.grad(S)) * dx
    F += thermal_convection - source * S * dx

    tdim, fdim = msh.topology.dim, msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.full(xx.shape[1], True))
    V0, _ = X.sub(0).collapse()
    W0, _ = X.sub(2).collapse()
    bc_u = fem.dirichletbc(fem.Function(V0), fem.locate_dofs_topological((X.sub(0), V0), fdim, facets), X.sub(0))
    bc_T = fem.dirichletbc(fem.Function(W0), fem.locate_dofs_topological((X.sub(2), W0), fdim, facets), X.sub(2))
    Q0, _ = X.sub(1).collapse()
    msh.topology.create_connectivity(tdim, tdim)
    gauge_cells = np.asarray([0], dtype=np.int32) if comm.rank == 0 else np.asarray([], dtype=np.int32)
    gauge = fem.locate_dofs_topological((X.sub(1), Q0), tdim, gauge_cells)
    bc_p = fem.dirichletbc(fem.Function(Q0), gauge, X.sub(1))

    options = {
        "snes_type": "newtonls", "snes_linesearch_type": "bt",
        "snes_rtol": cfg["snes_rtol"], "snes_atol": cfg["snes_atol"],
        "snes_max_it": cfg["snes_max_it"], "snes_error_if_not_converged": True,
        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearProblem(F, state, bcs=[bc_u, bc_p, bc_T],
                               petsc_options_prefix=f"complex_{name}_{beta_f:g}_",
                               petsc_options=options)
    steps = int(round(float(cfg["final_time"]) / float(cfg["dt"])))
    total_iterations = 0
    for step in range(1, steps + 1):
        state.x.array[:] = old.x.array
        problem.solve()
        state.x.scatter_forward()
        total_iterations += int(problem.solver.getIterationNumber())
        older.x.array[:] = old.x.array
        old.x.array[:] = state.x.array
        older.x.scatter_forward(); old.x.scatter_forward()
        if step == 1:
            a0.value = PETSc.ScalarType(1.5)
            a1.value = PETSc.ScalarType(-2.0)
            a2.value = PETSc.ScalarType(0.5)

    u_h, p_h, T_h = state.split()
    divergence = np.sqrt(_scalar(ufl.div(u_h) ** 2 * dx))
    dg0 = fem.functionspace(msh, ("DG", 0))
    cell_balance = assemble_vector(fem.form(ufl.div(u_h) * ufl.TestFunction(dg0) * dx))
    cell_balance.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    local_flux = float(np.max(np.abs(cell_balance.array))) if cell_balance.array.size else 0.0
    flux = comm.allreduce(local_flux, op=MPI.MAX)
    speed_l2 = np.sqrt(_scalar(ufl.inner(u_h, u_h) * dx))
    thermal_energy = np.sqrt(_scalar(T_h * T_h * dx))
    heat_flux = _scalar(-kappa * ufl.dot(ufl.grad(T_h), n) * ds)
    cells_local = msh.topology.index_map(tdim).size_local
    cells_global = comm.allreduce(cells_local, op=MPI.SUM)

    tag = f"{name}_beta_{beta_f:g}".replace(".", "p")
    results.mkdir(parents=True, exist_ok=True)
    _save_snapshot(msh, u_h, T_h, results / f"{tag}.npz")
    with io.XDMFFile(comm, results / f"{tag}.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return {
        "domain": name, "beta_f": beta_f, "cells": cells_global,
        "velocity_l2": speed_l2, "temperature_l2": thermal_energy,
        "boundary_heat_flux": heat_flux, "divergence_l2": divergence,
        "flux_imbalance": flux, "snes_iterations": total_iterations,
    }


def run(config_path: Path, results: Path) -> None:
    cfg = json.loads(config_path.read_text())
    rows = []
    for name in cfg["domains"]:
        for beta in cfg["beta_values"]:
            rows.append(solve_case(name, float(beta), cfg, results / "fields"))
    if MPI.COMM_WORLD.rank == 0:
        results.mkdir(parents=True, exist_ok=True)
        with (results / "complex_domain_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
