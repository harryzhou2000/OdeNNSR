import numpy as np
from abc import ABC, abstractmethod
from . import ODECopy


class ODE_F_RHS(ABC):
    @abstractmethod
    def __call__(self, u, cStage, iStage):  # -> f(u), f == du/dt(u)
        pass

    @abstractmethod
    def Jacobian(self, u: np.ndarray, cStage, iStage):
        pass

    @abstractmethod
    def dt(self, u: np.ndarray, cStage, iStage) -> float:
        pass


class ODE_F_SOLVE_SingleStage(ABC):
    @abstractmethod
    def __call__(self, u0, dt, alphaRHS, fRHS, fRes, cStage, iStage):  # -> (u,f(u))
        """solves: -(u - u0) / dt + alphaRHS * fRHS(u) + fRes == 0

        Warning: should treat u0 as immutable

        Args:
            u0 (_type_): _description_
            alphaRHS (_type_): _description_
            fRHS (_type_): _description_
            fRes (_type_): _description_
        """
        pass


class ImplicitOdeIntegrator(ABC):
    @abstractmethod
    def step(self, dt: float, u, fRHS: ODE_F_RHS, fSolve: ODE_F_SOLVE_SingleStage):
        pass


class ESDIRK(ImplicitOdeIntegrator):

    def __init__(self, method: str):
        super().__init__()
        from . import ESDIRK_Data

        butherAMap = {
            "ESDIRK4": ESDIRK_Data._ESDIRK_ButherA_ESDIRK4(),
            "ESDIRK3": ESDIRK_Data._ESDIRK_ButherA_ESDIRK3(),
            "Trapezoid": ESDIRK_Data._ESDIRK_ButherA_Trapezoid(),
            "BackwardEuler": ESDIRK_Data._ESDIRK_ButherA_BackwardEuler(),
        }
        if method in butherAMap:
            butcherA = butherAMap[method]
        else:
            raise ValueError(f"Method {method} not found!")

        butcherA = np.array(butcherA, dtype=np.float64)

        butcherB = butcherA[-1, :]
        butcherC = butcherA.sum(axis=1)
        self.butcherA = butcherA
        self.butcherB = butcherB
        self.butcherC = butcherC

        assert butcherA.shape[0] == butcherA.shape[1]
        self.nStage = butcherA.shape[0]
        assert butcherC[0] == 0
        assert (butcherA[0, :] == 0).all()
        self.rhsSeq = [None for _ in range(self.nStage)]
        self.uSeq = [None for _ in range(self.nStage)]

    def step(self, dt: float, u, fRHS: ODE_F_RHS, fSolve: ODE_F_SOLVE_SingleStage):
        uLast = u
        for iStage in range(1, self.nStage + 1):
            if iStage == 1:
                self.rhsSeq[iStage - 1] = fRHS(
                    u=u, cStage=self.butcherC[iStage - 1], iStage=iStage
                )
                continue

            fRes = uLast * (1 / dt)
            for jStage in range(1, iStage):
                fRes += self.butcherA[iStage - 1, jStage - 1] * self.rhsSeq[jStage - 1]
            self.uSeq[iStage - 1], self.rhsSeq[iStage - 1] = fSolve(
                u0=u,
                dt=dt,
                alphaRHS=self.butcherA[iStage - 1, iStage - 1],
                fRHS=fRHS,
                fRes=fRes,
                cStage=self.butcherC[iStage - 1],
                iStage=iStage,
            )
            u = self.uSeq[iStage - 1]

        return u
