from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element, mixed_element
from dolfinx import cpp, fem, io, mesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector


@dataclass
class Result:
    n: int
    h: float
    dt: float
    u_l2: float
    u_dg: float
    p_l2: float
    T_l2: float
    T_h1: float
    divergence_l2: float
    flux_imbalance: float
    snes_iterations: int


def _global_scalar(value) -> float:
    return float(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(value)), op=MPI.SUM))


def _exact_fields(msh, time, cfg):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    decay_rate = float(cfg.get("decay_rate", 1.0))
    decay = ufl.exp(-decay_rate * time)
    u = ufl.as_vector(
        (
            pi * decay * ufl.sin(pi * x[0]) ** 2 * ufl.sin(2 * pi * x[1]),
            -pi * decay * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]) ** 2,
        )
    )
    p = decay * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    T = decay * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    return u, p, T


def _coefficients(T, cfg):
    return cfg["nu_0"] + cfg["nu_slope"] * T, cfg["kappa_0"] + cfg["kappa_slope"] * T


def solve_level(ncells: int, cfg: dict, output_dir: Path) -> Result:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, ncells, ncells, cell_type=mesh.CellType.triangle)
    cell = msh.basix_cell()
    k = int(cfg["degree"])
    # Basix distinguishes simplex BDM elements from cubical BDM elements.
    # The manufactured mesh is triangular, hence the simplex family is BDM.
    cell_name = getattr(cell, "name", str(cell)).lower()
    bdm_family = "BDM" if cell_name in {"triangle", "tetrahedron"} else "BDMCF"
    velocity_el = element(bdm_family, cell, k)
    pressure_el = element("DG", cell, max(k - 1, 0))
    temperature_el = element("Lagrange", cell, k)
    X = fem.functionspace(msh, mixed_element([velocity_el, pressure_el, temperature_el]))

    state = fem.Function(X, name="state")
    old = fem.Function(X, name="old")
    older = fem.Function(X, name="older")
    u, p, T = ufl.split(state)
    u0, p0, T0 = ufl.split(old)
    um1, pm1, Tm1 = ufl.split(older)
    v, z, S = ufl.TestFunctions(X)

    t = fem.Constant(msh, PETSc.ScalarType(0.0))
    dt_value = cfg["dt_factor"] / ncells
    steps = int(np.ceil(cfg["final_time"] / dt_value))
    dt_value = cfg["final_time"] / steps
    dt = fem.Constant(msh, PETSc.ScalarType(dt_value))
    uex, pex, Tex = _exact_fields(msh, t, cfg)
    nu_exact, kappa_exact = _coefficients(Tex, cfg)
    eps = lambda w: ufl.sym(ufl.grad(w))
    gravity = ufl.as_vector(tuple(cfg["gravity"]))
    r = float(cfg["forchheimer_r"])
    decay_rate = float(cfg.get("decay_rate", 1.0))
    f_exact = (
        -decay_rate * uex
        - ufl.div(2 * nu_exact * eps(uex))
        + ufl.dot(uex, ufl.nabla_grad(uex))
        + cfg["alpha"] * uex
        + cfg["beta_f"] * ufl.sqrt(ufl.inner(uex, uex) + 1.0e-24) ** (r - 2) * uex
        + ufl.grad(pex)
        - cfg["gamma"] * (Tex - cfg["temperature_reference"]) * gravity
    )
    q_exact = -decay_rate * Tex - ufl.div(kappa_exact * ufl.grad(Tex)) + ufl.dot(uex, ufl.grad(Tex))

    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    dS = ufl.Measure("dS", domain=msh)
    ds = ufl.Measure("ds", domain=msh)
    dx = ufl.dx(domain=msh)
    Tstar = 2 * T0 - Tm1
    ustar = 2 * u0 - um1
    nu, kappa = _coefficients(Tstar, cfg)
    bdf_u = (3 * u - 4 * u0 + um1) / (2 * dt)
    bdf_T = (3 * T - 4 * T0 + Tm1) / (2 * dt)

    sigma = float(cfg["sip_penalty"]) * k * k
    viscous = ufl.inner(2 * nu * eps(u), eps(v)) * dx
    viscous += -ufl.inner(ufl.avg(2 * nu * eps(u)), ufl.outer(ufl.jump(v), n("+"))) * dS
    viscous += -ufl.inner(ufl.avg(2 * nu * eps(v)), ufl.outer(ufl.jump(u), n("+"))) * dS
    viscous += sigma * ufl.avg(nu / h) * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
    viscous += -ufl.inner(2 * nu * eps(u), ufl.outer(v, n)) * ds
    viscous += -ufl.inner(2 * nu * eps(v), ufl.outer(u, n)) * ds
    viscous += sigma * nu / h * ufl.inner(u, v) * ds

    un_plus = ufl.dot(ustar("+"), n("+"))
    upwind_u = ufl.conditional(ufl.ge(un_plus, 0), u("+"), u("-"))
    convection = -ufl.inner(ufl.outer(u, ustar), ufl.grad(v)) * dx
    convection += ufl.inner(un_plus * upwind_u, ufl.jump(v)) * dS
    thermal_convection = 0.5 * (ufl.dot(u, ufl.grad(T)) * S - ufl.dot(u, ufl.grad(S)) * T) * dx
    speed = ufl.sqrt(ufl.inner(u, u) + 1.0e-24)

    residual = ufl.inner(bdf_u, v) * dx + viscous + convection
    residual += cfg["alpha"] * ufl.inner(u, v) * dx
    residual += cfg["beta_f"] * speed ** (r - 2) * ufl.inner(u, v) * dx
    residual += -p * ufl.div(v) * dx + z * ufl.div(u) * dx
    residual += -ufl.inner(f_exact + cfg["gamma"] * (T - cfg["temperature_reference"]) * gravity, v) * dx
    residual += bdf_T * S * dx + kappa * ufl.inner(ufl.grad(T), ufl.grad(S)) * dx
    residual += thermal_convection - q_exact * S * dx

    # Enforce the normal velocity trace strongly; tangential zero data are imposed by SIP.
    V0, _ = X.sub(0).collapse()
    facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological((X.sub(0), V0), msh.topology.dim - 1, facets)
    zero_u = fem.Function(V0)
    bc_u = fem.dirichletbc(zero_u, dofs, X.sub(0))

    # The manufactured temperature vanishes on the whole boundary.  This
    # essential condition must be imposed explicitly; otherwise the thermal
    # equation has the wrong (natural Neumann) boundary condition.
    W0, _ = X.sub(2).collapse()
    temperature_dofs = fem.locate_dofs_topological(
        (X.sub(2), W0), msh.topology.dim - 1, facets
    )
    zero_T = fem.Function(W0)
    bc_T = fem.dirichletbc(zero_T, temperature_dofs, X.sub(2))

    # Remove the one-dimensional constant-pressure nullspace. Pinning one
    # pressure degree of freedom only selects a representative of p + R and
    # leaves the velocity and elementwise incompressibility equations intact.
    Q0, _ = X.sub(1).collapse()
    if comm.rank == 0 and msh.topology.index_map(msh.topology.dim).size_local > 0:
        gauge_cells = np.asarray([0], dtype=np.int32)
    else:
        gauge_cells = np.asarray([], dtype=np.int32)
    # DG pressure dofs live on cells. Some DOLFINx stable images do not
    # construct the cell-to-cell connectivity eagerly.
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
    pressure_dofs = fem.locate_dofs_topological(
        (X.sub(1), Q0), msh.topology.dim, gauge_cells
    )
    zero_p = fem.Function(Q0)
    bc_p = fem.dirichletbc(zero_p, pressure_dofs, X.sub(1))

    options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": cfg["snes_rtol"],
        "snes_atol": cfg["snes_atol"],
        "snes_max_it": cfg["snes_max_it"],
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    problem = NonlinearProblem(
        residual,
        state,
        bcs=[bc_u, bc_p, bc_T],
        petsc_options_prefix=f"cbf_n{ncells}_",
        petsc_options=options,
    )

    def interpolate_initial(target, time_value):
        t.value = PETSc.ScalarType(time_value)
        ue, pe, Te = _exact_fields(msh, t, cfg)
        expressions = [ue, pe, Te]
        for i, expr in enumerate(expressions):
            collapsed = target.sub(i).collapse()
            e = fem.Expression(expr, collapsed.function_space.element.interpolation_points)
            collapsed.interpolate(e)
            target.sub(i).interpolate(collapsed)
        target.x.scatter_forward()

    interpolate_initial(older, 0.0)
    interpolate_initial(old, dt_value)
    state.x.array[:] = old.x.array
    total_iterations = 0
    for step in range(2, steps + 1):
        t.value = PETSc.ScalarType(step * dt_value)
        state.x.array[:] = old.x.array
        problem.solve()
        state.x.scatter_forward()
        total_iterations += int(problem.solver.getIterationNumber())
        older.x.array[:] = old.x.array
        old.x.array[:] = state.x.array
        older.x.scatter_forward()
        old.x.scatter_forward()

    u_h, p_h, T_h = state.split()
    p_mean = _global_scalar(p_h * dx) / _global_scalar(1.0 * dx)
    p_error = p_h - p_mean - pex
    u_l2 = np.sqrt(_global_scalar(ufl.inner(u_h - uex, u_h - uex) * dx))
    u_dg = np.sqrt(_global_scalar(ufl.inner(eps(u_h - uex), eps(u_h - uex)) * dx + ufl.inner(ufl.jump(u_h), ufl.jump(u_h)) / ufl.avg(h) * dS))
    p_l2 = np.sqrt(_global_scalar(p_error * p_error * dx))
    T_l2 = np.sqrt(_global_scalar((T_h - Tex) ** 2 * dx))
    T_h1 = np.sqrt(_global_scalar(ufl.inner(ufl.grad(T_h - Tex), ufl.grad(T_h - Tex)) * dx))
    divergence = np.sqrt(_global_scalar(ufl.div(u_h) ** 2 * dx))

    dg0 = fem.functionspace(msh, ("DG", 0))
    q0 = ufl.TestFunction(dg0)
    cell_balance = assemble_vector(fem.form(ufl.div(u_h) * q0 * dx))
    cell_balance.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    flux_local = float(np.max(np.abs(cell_balance.array))) if cell_balance.array.size else 0.0
    flux_imbalance = comm.allreduce(flux_local, op=MPI.MAX)
    entities = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)
    if hasattr(mesh, "h"):
        h_values = mesh.h(msh, msh.topology.dim, entities)
    else:
        h_values = cpp.mesh.h(msh._cpp_object, msh.topology.dim, entities)
    hmax_local = float(np.max(h_values)) if len(h_values) else 0.0
    hmax = comm.allreduce(float(hmax_local), op=MPI.MAX)

    # XDMF supports nodal Lagrange fields, not BDM fields.  These P1 copies
    # are used only for visualisation; all norms above use the mixed solution.
    V_vis = fem.functionspace(
        msh, element("Lagrange", cell, 1, shape=(msh.geometry.dim,))
    )
    Q_vis = fem.functionspace(msh, element("Lagrange", cell, 1))
    u_vis = fem.Function(V_vis, name="velocity")
    p_vis = fem.Function(Q_vis, name="pressure")
    T_vis = fem.Function(Q_vis, name="temperature")
    u_vis.interpolate(fem.Expression(u_h, V_vis.element.interpolation_points))
    p_vis.interpolate(fem.Expression(p_h - p_mean, Q_vis.element.interpolation_points))
    T_vis.interpolate(fem.Expression(T_h, Q_vis.element.interpolation_points))

    output_dir.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(comm, output_dir / f"manufactured_n{ncells}.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(u_vis, float(t.value))
        xdmf.write_function(p_vis, float(t.value))
        xdmf.write_function(T_vis, float(t.value))
    return Result(ncells, hmax, dt_value, u_l2, u_dg, p_l2, T_l2, T_h1, divergence, flux_imbalance, total_iterations)


def run(config_path: Path, results_dir: Path) -> None:
    cfg = json.loads(config_path.read_text())
    rows = [solve_level(int(n), cfg, results_dir / "fields") for n in cfg["meshes"]]
    if MPI.COMM_WORLD.rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / "manufactured_convergence.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(Result.__dataclass_fields__))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)


if __name__ == "__main__":
    run(Path("config.json"), Path("results"))
