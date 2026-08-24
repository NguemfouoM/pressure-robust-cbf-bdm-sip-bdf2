"""
Common FEniCSx solver for the BDM1-DG0-SIPG/BDF2 thermally coupled
convective Brinkman-Forchheimer model used in the manuscript.

Target: recent FEniCSx releases using basix.ufl.element / mixed_element.
The code deliberately uses the same structure as the paper:
  - BDM_k velocity, DG_{k-1} pressure, CG_k temperature
  - strong zero-normal BDM boundary condition
  - SIP/Nitsche tangential Brinkman diffusion
  - divergence-conforming upwind momentum convection
  - skew thermal convection
  - BDF2 with extrapolated advecting velocity and coefficients
  - implicit Forchheimer drag
  - nu(T)=1+0.1*tanh(T), kappa(T)=1+0.1*tanh(T)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh


@dataclass
class Parameters:
    k: int = 1
    alpha: float = 1.0
    beta_F: float = 1.0
    gamma: float = 1.0
    r: float = 3.0
    sigma: float = 20.0
    T_ref: float = 0.0
    gx: float = 0.0
    gy: float = -1.0


def global_sum(value, comm=MPI.COMM_WORLD):
    return comm.allreduce(value, op=MPI.SUM)


def global_max(value, comm=MPI.COMM_WORLD):
    return comm.allreduce(value, op=MPI.MAX)


def assemble_scalar_global(form):
    return global_sum(fem.assemble_scalar(fem.form(form)))


def l2_norm(expr, dx):
    val = assemble_scalar_global(ufl.inner(expr, expr) * dx)
    return float(np.sqrt(max(val, 0.0)))


def remove_mean(expr, dx, domain_measure=1.0):
    mean = assemble_scalar_global(expr * dx) / domain_measure
    return expr - PETSc.ScalarType(mean)


class CBFProblem:
    def __init__(self, N=48, params=Parameters()):
        self.comm = MPI.COMM_WORLD
        self.params = params
        self.msh = mesh.create_unit_square(
            self.comm, N, N,
            cell_type=mesh.CellType.triangle
        )
        self.tdim = self.msh.topology.dim
        self.fdim = self.tdim - 1
        self.msh.topology.create_connectivity(self.fdim, self.tdim)

        cell = self.msh.basix_cell()
        Vel = element("BDM", cell, params.k)
        Pel = element("DG", cell, params.k - 1)
        Tel = element("Lagrange", cell, params.k)
        Mel = mixed_element([Vel, Pel, Tel])

        self.X = fem.functionspace(self.msh, Mel)
        self.V, self.V_to_X = self.X.sub(0).collapse()
        self.Q, self.Q_to_X = self.X.sub(1).collapse()
        self.W, self.W_to_X = self.X.sub(2).collapse()

        self.dx = ufl.Measure("dx", domain=self.msh)
        self.dS = ufl.Measure("dS", domain=self.msh)
        self.ds = ufl.Measure("ds", domain=self.msh)
        self.n = ufl.FacetNormal(self.msh)
        self.h = ufl.CellDiameter(self.msh)
        self.I = ufl.Identity(self.msh.geometry.dim)

        self.dt = fem.Constant(self.msh, PETSc.ScalarType(0.01))
        self.time = fem.Constant(self.msh, PETSc.ScalarType(0.0))
        self.lambda_grad = fem.Constant(self.msh, PETSc.ScalarType(0.0))

        self.state = fem.Function(self.X, name="state")
        self.state_n = fem.Function(self.X, name="state_n")
        self.state_nm1 = fem.Function(self.X, name="state_nm1")
        self.test = ufl.TestFunctions(self.X)

        # Boundary conditions: BDM boundary DOFs are normal-flux DOFs.
        facets = mesh.locate_entities_boundary(
            self.msh, self.fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        u_zero = fem.Function(self.V)
        T_zero = fem.Function(self.W)
        dofs_u = fem.locate_dofs_topological((self.X.sub(0), self.V), self.fdim, facets)
        dofs_T = fem.locate_dofs_topological((self.X.sub(2), self.W), self.fdim, facets)
        self.bcs = [
            fem.dirichletbc(u_zero, dofs_u, self.X.sub(0)),
            fem.dirichletbc(T_zero, dofs_T, self.X.sub(2)),
        ]

        # Pressure gauge: pin one pressure degree of freedom.
        # This selects one representative of the pressure equivalence class
        # and has no effect on the velocity pressure-robustness property.
        q_to_x = np.asarray(self.Q_to_X, dtype=np.int32).reshape(-1)
        pressure_dof = np.array([q_to_x[0]], dtype=np.int32)
        p_zero = PETSc.ScalarType(0.0)
        self.bcs.append(fem.dirichletbc(p_zero, pressure_dof, self.X.sub(1)))

        self._build_exact_fields()
        self._build_forms()

    # ---------- basic tensor operators ----------
    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def tangential(self, v, normal):
        return v - ufl.dot(v, normal) * normal

    def nu(self, T):
        return 1.0 + 0.1 * ufl.tanh(T)

    def kappa(self, T):
        return 1.0 + 0.1 * ufl.tanh(T)

    def h_avg(self):
        return 0.5 * (self.h("+") + self.h("-"))

    def nu_facet(self, T):
        # Robust max-weighted penalty as stated in the corrected manuscript.
        return ufl.max_value(self.nu(T("+")), self.nu(T("-")))

    def sip(self, theta, u, v):
        n = self.n
        Pp = self.I - ufl.outer(n("+"), n("+"))
        jump_ut = Pp * (u("+") - u("-"))
        jump_vt = Pp * (v("+") - v("-"))

        stress_p_u = 2.0 * self.nu(theta("+")) * self.eps(u("+"))
        stress_m_u = 2.0 * self.nu(theta("-")) * self.eps(u("-"))
        stress_p_v = 2.0 * self.nu(theta("+")) * self.eps(v("+"))
        stress_m_v = 2.0 * self.nu(theta("-")) * self.eps(v("-"))

        tr_u_avg = 0.5 * (stress_p_u + stress_m_u) * n("+")
        tr_v_avg = 0.5 * (stress_p_v + stress_m_v) * n("+")

        vol = ufl.inner(2.0 * self.nu(theta) * self.eps(u), self.eps(v)) * self.dx
        interior = (
            - ufl.inner(tr_u_avg, jump_vt) * self.dS
            - ufl.inner(tr_v_avg, jump_ut) * self.dS
            + self.params.sigma * self.params.k**2
              * self.nu_facet(theta) / self.h_avg()
              * ufl.inner(jump_ut, jump_vt) * self.dS
        )

        # Boundary: normal component is imposed strongly; tangential trace by Nitsche/SIP.
        P = self.I - ufl.outer(n, n)
        ut = P * u
        vt = P * v
        tr_u = (2.0 * self.nu(theta) * self.eps(u)) * n
        tr_v = (2.0 * self.nu(theta) * self.eps(v)) * n
        boundary = (
            - ufl.inner(tr_u, vt) * self.ds
            - ufl.inner(tr_v, ut) * self.ds
            + self.params.sigma * self.params.k**2
              * self.nu(theta) / self.h
              * ufl.inner(ut, vt) * self.ds
        )
        return vol + interior + boundary

    def momentum_convection(self, w, u, v):
        n = self.n
        jump_u = u("+") - u("-")
        jump_v = v("+") - v("-")
        wn = ufl.dot(w("+"), n("+"))

        vol = ufl.inner(ufl.dot(w, ufl.grad(u)), v) * self.dx
        flux = - ufl.inner(wn * jump_u, ufl.avg(v)) * self.dS
        upwind = 0.5 * abs(wn) * ufl.inner(jump_u, jump_v) * self.dS
        boundary = 0.5 * abs(ufl.dot(w, n)) * ufl.inner(u, v) * self.ds
        return vol + flux + upwind + boundary

    def thermal_convection(self, w, T, S):
        return 0.5 * (
            ufl.inner(ufl.dot(w, ufl.grad(T)), S)
            - ufl.inner(ufl.dot(w, ufl.grad(S)), T)
        ) * self.dx

    # ---------- exact manufactured fields ----------
    def _build_exact_fields(self):
        x = ufl.SpatialCoordinate(self.msh)
        t = self.time
        pi = np.pi
        decay = ufl.exp(-t)

        self.u_exact = pi * decay * ufl.as_vector((
            ufl.sin(pi*x[0])**2 * ufl.sin(2*pi*x[1]),
            -ufl.sin(2*pi*x[0]) * ufl.sin(pi*x[1])**2
        ))
        self.p_exact = decay * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
        self.T_exact = decay * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
        self.phi = ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])

        g = ufl.as_vector((self.params.gx, self.params.gy))

        du_dt = -self.u_exact
        dT_dt = -self.T_exact

        visc = -ufl.div(2.0 * self.nu(self.T_exact) * self.eps(self.u_exact))
        conv = ufl.dot(self.u_exact, ufl.grad(self.u_exact))
        drag = self.params.alpha * self.u_exact
        speed = ufl.sqrt(ufl.inner(self.u_exact, self.u_exact) + 1.0e-30)
        forch = self.params.beta_F * speed**(self.params.r - 2.0) * self.u_exact
        gradp = ufl.grad(self.p_exact)
        buoy = self.params.gamma * (self.T_exact - self.params.T_ref) * g

        self.f_exact = du_dt + visc + conv + drag + forch + gradp - buoy

        heat_diff = -ufl.div(self.kappa(self.T_exact) * ufl.grad(self.T_exact))
        heat_conv = ufl.dot(self.u_exact, ufl.grad(self.T_exact))
        self.q_exact = dT_dt + heat_diff + heat_conv

    def _build_forms(self):
        u, p, T = ufl.split(self.state)
        un, pn, Tn = ufl.split(self.state_n)
        unm1, pnm1, Tnm1 = ufl.split(self.state_nm1)
        v, q, S = self.test

        dt = self.dt
        u_star = 2.0 * un - unm1
        T_star = 2.0 * Tn - Tnm1
        D2u = (3.0*u - 4.0*un + unm1) / (2.0*dt)
        D2T = (3.0*T - 4.0*Tn + Tnm1) / (2.0*dt)

        g = ufl.as_vector((self.params.gx, self.params.gy))
        speed = ufl.sqrt(ufl.inner(u, u) + PETSc.ScalarType(1.0e-30))
        f_rhs = self.f_exact + self.lambda_grad * ufl.grad(self.phi)

        F = (
            ufl.inner(D2u, v) * self.dx
            + self.sip(T_star, u, v)
            + self.momentum_convection(u_star, u, v)
            + self.params.alpha * ufl.inner(u, v) * self.dx
            + self.params.beta_F * ufl.inner(speed**(self.params.r-2.0) * u, v) * self.dx
            - p * ufl.div(v) * self.dx
            - ufl.inner(f_rhs, v) * self.dx
            - self.params.gamma * ufl.inner((T - self.params.T_ref) * g, v) * self.dx
            + ufl.div(u) * q * self.dx
            + ufl.inner(D2T, S) * self.dx
            + self.kappa(T_star) * ufl.inner(ufl.grad(T), ufl.grad(S)) * self.dx
            + self.thermal_convection(u, T, S)
            - self.q_exact * S * self.dx
        )
        self.F = fem.form(F)
        dstate = ufl.TrialFunction(self.X)
        self.J = fem.form(ufl.derivative(F, self.state, dstate))

    def _interpolate_ufl_to_subfunction(self, expr, mixed_function, sub_index):
        sub = mixed_function.sub(sub_index)
        Vsub, _ = sub.collapse()
        pts = Vsub.element.interpolation_points
        ex = fem.Expression(expr, pts)
        tmp = fem.Function(Vsub)
        tmp.interpolate(ex)
        # Scatter values into the mixed subfunction via interpolation.
        sub.interpolate(tmp)
        mixed_function.x.scatter_forward()

    def set_exact_history(self, t_nm1, t_n):
        """Use exact manufactured values for the two BDF2 history states."""
        self.time.value = PETSc.ScalarType(t_nm1)
        self._interpolate_ufl_to_subfunction(self.u_exact, self.state_nm1, 0)
        self._interpolate_ufl_to_subfunction(self.p_exact, self.state_nm1, 1)
        self._interpolate_ufl_to_subfunction(self.T_exact, self.state_nm1, 2)

        self.time.value = PETSc.ScalarType(t_n)
        self._interpolate_ufl_to_subfunction(self.u_exact, self.state_n, 0)
        self._interpolate_ufl_to_subfunction(self.p_exact, self.state_n, 1)
        self._interpolate_ufl_to_subfunction(self.T_exact, self.state_n, 2)
        self.state.x.array[:] = self.state_n.x.array
        self.state.x.scatter_forward()

    def solve_step(self, t_np1, lambda_grad=0.0):
        self.time.value = PETSc.ScalarType(t_np1)
        self.lambda_grad.value = PETSc.ScalarType(lambda_grad)

        problem = fem.petsc.NonlinearProblem(self.F, self.state, bcs=self.bcs, J=self.J)
        solver = PETSc.SNES().create(self.comm)
        solver.setType("newtonls")
        solver.getLineSearch().setType("bt")
        solver.setTolerances(rtol=1e-9, atol=1e-11, max_it=40)

        # Direct solve. If MUMPS is unavailable, PETSc will report it explicitly.
        ksp = solver.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass

        solver.setFunction(problem.F, problem.vector)
        solver.setJacobian(problem.J, problem.matrix)

        solver.solve(None, self.state.x.petsc_vec)
        self.state.x.scatter_forward()
        its = solver.getIterationNumber()
        reason = solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"SNES failed with reason {reason}")

        return int(its)

    def advance_history(self):
        self.state_nm1.x.array[:] = self.state_n.x.array
        self.state_n.x.array[:] = self.state.x.array
        self.state_nm1.x.scatter_forward()
        self.state_n.x.scatter_forward()

    def extract(self):
        return (
            self.state.sub(0).collapse(),
            self.state.sub(1).collapse(),
            self.state.sub(2).collapse(),
        )

    def run(self, dt, tf=1.0, lambda_grad=0.0, exact_start=True):
        self.dt.value = PETSc.ScalarType(dt)
        nsteps = int(round(tf/dt))
        if abs(nsteps*dt - tf) > 1e-12:
            raise ValueError("tf must be an integer multiple of dt")

        # For the verification tests we use exact manufactured history values at
        # t=0 and t=dt. This isolates the BDF2 discretization and preserves the
        # intended second-order start.
        if exact_start:
            self.set_exact_history(0.0, dt)
            first_step = 1
        else:
            raise NotImplementedError(
                "Use your production backward-Euler starter here if desired."
            )

        total_its = 0
        max_its = 0
        for j in range(first_step, nsteps):
            t_np1 = (j + 1) * dt
            its = self.solve_step(t_np1, lambda_grad=lambda_grad)
            total_its += its
            max_its = max(max_its, its)
            self.advance_history()

        u, p, T = self.extract()
        return {
            "u": u, "p": p, "T": T,
            "total_snes": total_its,
            "mean_snes": total_its / max(nsteps-first_step, 1),
            "max_snes": max_its,
            "nsteps": nsteps,
        }

    def vector_l2_difference(self, a, b):
        return l2_norm(a - b, self.dx)

    def scalar_l2_difference(self, a, b):
        return l2_norm(a - b, self.dx)

    def exact_errors(self, sol, t_final):
        self.time.value = PETSc.ScalarType(t_final)
        return {
            "u_l2": l2_norm(self.u_exact - sol["u"], self.dx),
            "T_l2": l2_norm(self.T_exact - sol["T"], self.dx),
        }
