import numpy as np

import OdeNNSR.ODE as ODE

from copy import deepcopy

from collections import defaultdict


def test_scalar_euler_cor(
    dt,
    nStep,
    frhs,
    fsolve,
    us_bg,
    ts_bg,
    solverC,
    training,
    training_scale_set=[2, 4, 8, 16],
):

    import OdeNNSR.ScalarCorrectorNet
    import torch
    from torch.utils.data import Dataset, DataLoader

    batch_size = training["batch_size"]
    n_epoch = training["n_epoch"]
    learning_rate = training["learning_rate"]

    vec_dim = us_bg[0].__len__()

    # torch.manual_seed(123)

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
            self.cached_data = {}

        def __len__(self):
            return len(self.dt)

        def __getitem__(self, idx):
            u0 = self.u0[idx]
            u1 = self.u1[idx]
            dt = self.dt[idx]
            if idx not in self.cached_data:
                u1_L = solverC.step(dt, u0, frhs, fsolve)
                f0 = frhs(u0, 0.0, 1)
                f1 = frhs(u1, 1.0, 2)
                f1_L = frhs(u1_L, 1.0, 2)

                self.cached_data[idx] = (u0, u1, u1_L, f0, f1, f1_L)
            else:
                (u0, u1, u1_L, f0, f1, f1_L) = self.cached_data[idx]

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

    dataset = TimeHistoryDataset(training_scale_set, us_bg, ts_bg)
    model = OdeNNSR.ScalarCorrectorNet.ScalarCorrector(dataset[0][0].__len__(), vec_dim)

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

    ret = {}
    ret["dt"] = dt
    ret["nStep"] = nStep
    ret["us_bg"] = us_bg
    ret["ts_bg"] = ts_bg
    ret["model"] = model
    ret["solverC"] = solverC
    ret["frhs"] = frhs
    ret["fsolve"] = fsolve
    return ret


def test_oscillator_euler_cor_eval_run(
    dt,
    nStep,
    dt_expand_scale,
    us_bg,
    ts_bg,
    model,
    solverC,
    frhs,
    fsolve,
    test_corr_scale,
):
    import torch

    model.eval()

    dtC = dt * dt_expand_scale
    u = us_bg[0]
    u_nc = us_bg[0]
    t = 0
    us = [u]
    ts = [t]
    u_ncs = [u_nc]

    for i in range(round(nStep / dt_expand_scale)):
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
