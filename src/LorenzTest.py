import numpy as np

import OdeNNSR.ODE as ODE

from copy import deepcopy

import warnings


class LorenzODE_FRHS(ODE.ODE_F_RHS):
    def __init__(self, rho=28.0, sigma=10.0, beta=8 / 3):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.b = np.zeros((3, 1))

    def __call__(self, u, cStage, iStage):
        x = u[0]
        y = u[1]
        z = u[2]
        return (
            np.array(
                [
                    self.sigma * (y - x),
                    x * (self.rho - z) - y,
                    x * y - self.beta * z,
                ],
                dtype=np.float64,
            )
            + self.b
        )

    def Jacobian(self, u: np.ndarray, cStage, iStage):
        x = u.flat[0]
        y = u.flat[1]
        z = u.flat[2]

        return np.array(
            [
                [
                    -self.sigma,
                    (self.rho - z),
                    y,
                ],
                [
                    self.sigma,
                    -1,
                    x,
                ],
                [
                    0,
                    -x,
                    -self.beta,
                ],
            ],
            dtype=np.float64,
        )


class LorenzODE_FSOVLE(ODE.ODE_F_SOLVE_SingleStage):
    def __init__(self, nIter=128, thres=1e-5):
        self.nIter = nIter
        self.thres = thres
        pass

    def __call__(self, u0, dt, alphaRHS, fRHS: LorenzODE_FRHS, fRes, cStage, iStage):

        u = np.array(u0)
        for iIter in range(self.nIter):
            res = fRes + fRHS(u, cStage, iStage) * alphaRHS - (1 / dt) * u
            resNorm = np.linalg.norm(res)
            if iIter == 0:
                resNorm0 = resNorm
            else:
                if resNorm <= resNorm0 * self.thres:
                    break

            J = (
                np.eye(3) * (1 / dt)
                + np.eye(3)
                - fRHS.Jacobian(u, cStage, iStage) * alphaRHS
            )
            uInc = np.linalg.solve(J, res)
            u += uInc
        else:
            warnings.warn(
                f"dit not converge! {resNorm0} -> {resNorm}, dt = {dt}",
                category=RuntimeWarning,
            )

        return u, fRHS(u=u, cStage=cStage, iStage=iStage)


def test_lorenz_ode(
    dt, nStep, solver: ODE.ImplicitOdeIntegrator, nIter=128, thres=1e-5
):
    frhs, fsolve = LorenzODE_FRHS(), LorenzODE_FSOVLE(nIter=nIter, thres=thres)
    u0 = np.array([1, 0, 0], dtype=np.float64).reshape(3, 1)
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


# def test_oscillator(omega=1, dt=0.01, nStep=1000, solver=ODE.ESDIRK("BackwardEuler")):
#     A = np.array([[0, 1], [-omega, 0]], dtype=np.float64)
#     u0 = np.array([1, 0], dtype=np.float64).reshape(-1, 1)
#     return test_linear_ode(A, u0, dt, nStep, solver)


def test_lorenz_bg_prepare(dt=0.01, nStep=1000, nIter=128, thres=1e-5):
    us_bg, ts_bg = test_lorenz_ode(
        dt=dt, nStep=nStep, solver=ODE.ESDIRK("ESDIRK4"), nIter=nIter, thres=thres
    )
    solverC = ODE.ESDIRK("BackwardEuler")
    frhs, fsolve = LorenzODE_FRHS(), LorenzODE_FSOVLE(nIter=nIter, thres=thres)

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
    us, ts = test_lorenz_ode(0.01, 100, ODE.ESDIRK("ESDIRK4"))
    pass
