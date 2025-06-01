import numpy as np

def levy(dim, beta=1.5):
    sigma = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) /
             (np.math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1/beta))
    return step

def polr(A, R0, N, t, Tmax, r):
    th = (1 + t/Tmax) * A * np.pi * np.random.rand(N)
    R = (r - t/Tmax) * R0 * np.random.rand(N)
    xR = R * np.sin(th)
    yR = R * np.cos(th)
    if np.max(np.abs(xR)) != 0:
        xR = xR / np.max(np.abs(xR))
    if np.max(np.abs(yR)) != 0:
        yR = yR / np.max(np.abs(yR))
    return xR, yR

def RTH(N, Tmax, low, high, dim, fobj):
    Xbestcost = np.inf
    Xbestpos = np.random.rand(dim)
    Xpos = np.zeros((N, dim))
    Xcost = np.zeros(N)

    # Initialization
    for i in range(N):
        Xpos[i, :] = low + (high - low) * np.random.rand(dim)
        Xcost[i] = fobj(Xpos[i, :])
        if Xcost[i] < Xbestcost:
            Xbestpos = Xpos[i, :].copy()
            Xbestcost = Xcost[i]
    A = 15
    R0 = 0.5
    r = 1.5
    Convergence_curve = []

    for t in range(Tmax):
        # 1- High Soaring
        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + np.sin(2.5 - t / Tmax)
        for i in range(N):
            Xnewpos = Xbestpos + (Xmean - Xpos[i, :]) * levy(dim) * TF
            Xnewpos = np.clip(Xnewpos, low, high)
            Xnewcost = fobj(Xnewpos)
            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :].copy()
                    Xbestcost = Xcost[i]

        # 2- Low Soaring
        Xmean = np.mean(Xpos, axis=0)
        for i in range(N-1):
            aa = np.random.permutation(N)
            Xpos = Xpos[aa, :]
            Xcost = Xcost[aa]
            x, y = polr(A, R0, N, t, Tmax, r)
            StepSize = Xpos[i, :] - Xmean
            Xnewpos = Xbestpos + (y[i] + x[i]) * StepSize
            Xnewpos = np.clip(Xnewpos, low, high)
            Xnewcost = fobj(Xnewpos)
            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :].copy()
                    Xbestcost = Xcost[i]

        # 3- Stopping & Swooping
        Xmean = np.mean(Xpos, axis=0)
        TF = 1 + 0.5 * np.sin(2.5 - t / Tmax)
        for i in range(N):
            b = np.random.permutation(N)
            Xpos = Xpos[b, :]
            Xcost = Xcost[b]
            x, y = polr(A, R0, N, t, Tmax, r)
            alpha = (np.sin(2.5 - t / Tmax)) ** 2
            G = 2 * (1 - (t / Tmax))
            StepSize1 = 1. * Xpos[i, :] - TF * Xmean
            StepSize2 = G * Xpos[i, :] - TF * Xbestpos
            Xnewpos = alpha * Xbestpos + x[i] * StepSize1 + y[i] * StepSize2
            Xnewpos = np.clip(Xnewpos, low, high)
            Xnewcost = fobj(Xnewpos)
            if Xnewcost < Xcost[i]:
                Xpos[i, :] = Xnewpos
                Xcost[i] = Xnewcost
                if Xcost[i] < Xbestcost:
                    Xbestpos = Xpos[i, :].copy()
                    Xbestcost = Xcost[i]

        Convergence_curve.append(Xbestcost)

    Cost = Xbestcost
    Pos = Xbestpos
    return Cost, Pos, Convergence_curve