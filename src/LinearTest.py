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


def test_linear_ode(A, u0, dt, nstep, solver: ODE.ImplicitOdeIntegrator):
    frhs, fsolve = get_linear_ode(A)
    u = u0
    t = 0
    us = [u]
    ts = [t]

    for iStep in range(1, nstep + 1):
        u = solver.step(dt, u, frhs, fsolve)
        t += dt
        us.append(u)
        ts.append(t)

    return us, ts


def test_oscillator(omega=1, dt=0.01, nStep=1000, solver=ODE.ESDIRK("BackwardEuler")):
    import matplotlib.pyplot

    A = np.array([[0, 1], [-omega, 0]], dtype=np.float64)
    u0 = np.array([1, 0], dtype=np.float64).reshape(-1, 1)
    return test_linear_ode(A, u0, dt, nStep, solver)


def test_oscillator_euler_cor(omega=1, dt=0.01, nStep=1000):
    us_bg, ts_bg = test_oscillator(omega, dt, nStep, solver=ODE.ESDIRK("ESDIRK4"))
    solverC = ODE.ESDIRK("BackwardEuler")
    A = np.array([[0, 1], [-omega, 0]], dtype=np.float64)
    frhs, fsolve = get_linear_ode(A)

    dt_expand_scale = 8
    test_corr_scale = 0.99

    import OdeNNSR.ScalarCorrectorNet

    batch_size = 64
    n_epoch = 40
    learning_rate = 1e-3
    import torch

    # torch.manual_seed(123)

    from torch.utils.data import Dataset, DataLoader

    class TimeHistoryDataset(Dataset):
        def __init__(self, jumps, us, ts):
            dLen = len(ts)
            idxs = np.arange(dLen)

            u0_v = []
            u1_v = []
            dt_v = []
            us = np.array(us)
            ts = np.array(ts)

            for jump in jumps:
                idxs_next = idxs + jump
                idx_a = idxs[idxs_next < dLen]
                idx_b = idxs_next[idxs_next < dLen]
                u0_v.extend(us[idx_a])
                u1_v.extend(us[idx_b])
                dt_v.extend(ts[idx_b] - ts[idx_a])
            self.u0 = u0_v
            self.u1 = u1_v
            self.dt = dt_v

        def __len__(self):
            return len(self.dt)

        def __getitem__(self, idx):
            u0 = self.u0[idx]
            u1 = self.u1[idx]
            dt = self.dt[idx]
            u1_L = solverC.step(dt, u0, frhs, fsolve)
            f0 = frhs(u0, 0.0, 1)
            f1 = frhs(u1, 1.0, 2)
            f1_L = frhs(u1_L, 1.0, 2)
            f1_I = (u1 - u0) / dt
            f1_cor = (f1_I - f1) / dt

            sample = torch.tensor(
                np.concatenate(
                    # [(f1_L - f0) / dt, u0, u1_L, np.array(dt).reshape(1, 1)],
                    # [(f1_L - f0) / dt, u0, u1_L],
                    [u0, u1_L],
                    # [u1_L],
                    axis=0,
                ),
                dtype=torch.float64,
            )

            label = torch.tensor(f1_cor, dtype=torch.float64)

            sample = sample.reshape(-1)
            label = label.reshape(-1)
            return sample, label

    dataset = TimeHistoryDataset([2, 4, 8, 16], us_bg, ts_bg)
    model = OdeNNSR.ScalarCorrectorNet.ScalarCorrector(dataset[0][0].__len__(), 2)

    train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i_epoch in range(n_epoch):
        model.train()
        epoch_loss = 0.0
        for i, (samples, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(samples)
            loss = torch.nn.functional.mse_loss(output, labels)
            loss.backward()
            loss_f = float(loss.cpu())
            epoch_loss += loss_f
            optimizer.step()

            # if i % 100 == 0:
            #     print(f"{i_epoch+1}, {i+1}, loss: {loss_f}")
        print(f"{i_epoch+1}, loss: {epoch_loss}")

    model.eval()

    dtC = dt * dt_expand_scale
    u = us_bg[0]
    u_nc = us_bg[0]
    t = 0
    us = [u]
    ts = [t]
    u_ncs = [u_nc]

    for i in range(nStep // dt_expand_scale):
        u1 = solverC.step(dtC, u, frhs, fsolve)
        u_nc = solverC.step(dtC, u_nc, frhs, fsolve)
        f1 = frhs(u1, 1.0, 2)
        f = frhs(u, 0.0, 1)
        f_cor = dtC * model(
            torch.tensor(
                np.concatenate(
                    # [(f1 - f) / dtC, u, u1, f, f1, np.array(dtC).reshape(1, 1)],
                    # [(f1 - f) / dtC, u, u1],
                    [u, u1],
                    # [u1],
                    dtype=np.float64,
                )
            ).reshape(-1)
        ).detach().numpy().reshape(-1, 1)

        frhsC = deepcopy(frhs)
        frhsC.b += test_corr_scale * f_cor
        # print(frhs.b)
        # print(frhsC.b)

        u = solverC.step(dtC, u, frhsC, fsolve)
        t += dtC

        us.append(u)
        ts.append(t)
        u_ncs.append(u_nc)

    return us_bg, ts_bg, us, ts, u_ncs


if __name__ == "__main__":
    # us, ts = test_oscillator()
    test_oscillator_euler_cor(omega=1, dt=0.01, nStep=1000)
