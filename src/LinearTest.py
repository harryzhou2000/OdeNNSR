import numpy as np

import OdeNNSR.ODE as ODE

from copy import deepcopy


class LinearODE_FRHS(ODE.ODE_F_RHS):
    def __init__(self, A: np.ndarray, b: np.ndarray = None):
        self.A = A
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        self.b = b
        if b is None:
            self.b = np.zeros((A.shape[0], 1))

    def __call__(self, u, cStage, iStage):
        return self.A @ u + self.b


class LinearODE_FSOVLE(ODE.ODE_F_SOLVE_SingleStage):
    def __init__(self):
        pass

    def __call__(self, u0, dt, alphaRHS, fRHS: LinearODE_FRHS, fRes, cStage, iStage):
        J = np.eye(fRHS.A.shape[0]) * (1 / dt) - fRHS.A * alphaRHS
        uSol = np.linalg.solve(J, fRes + alphaRHS * fRHS.b)
        return uSol, fRHS(u=uSol, cStage=cStage, iStage=iStage)


def get_linear_ode(A):
    return LinearODE_FRHS(A), LinearODE_FSOVLE()


def test_linear_ode(A, u0, dt, nStep, solver: ODE.ImplicitOdeIntegrator):
    frhs, fsolve = get_linear_ode(A)
    u = u0
    t = 0
    us = [u]
    ts = [t]

    for iStep in range(1, nStep + 1):
        u = solver.step(dt, u, frhs, fsolve)
        t += dt
        us.append(u)
        ts.append(t)

    return us, ts


def test_oscillator(omega=1, dt=0.01, nStep=1000, solver=ODE.ESDIRK("BackwardEuler")):
    A = np.array([[0, 1], [-omega, 0]], dtype=np.float64)
    u0 = np.array([1, 0], dtype=np.float64).reshape(-1, 1)
    return test_linear_ode(A, u0, dt, nStep, solver)


def test_oscillator_bg_prepare(omega=1, dt=0.01, nStep=1000):
    from LinearTest import test_oscillator

    us_bg, ts_bg = test_oscillator(omega, dt, nStep, solver=ODE.ESDIRK("ESDIRK4"))
    solverC = ODE.ESDIRK("BackwardEuler")
    A = np.array([[0, 1], [-omega, 0]], dtype=np.float64)
    frhs, fsolve = get_linear_ode(A)
    
    ret = {}
    ret["dt"] = dt
    ret["nStep"] = nStep
    ret["us_bg"] = us_bg
    ret["ts_bg"] = ts_bg
    ret["solverC"] = solverC
    ret["frhs"] = frhs
    ret["fsolve"] = fsolve

    return ret


if __name__ == "__main__":
    # us, ts = test_oscillator()
    
    pass