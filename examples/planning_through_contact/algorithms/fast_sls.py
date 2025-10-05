from typing import NamedTuple

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    # Fallback dummy decorator
    def njit(func=None, **kwargs):
        if func is None:
            # If used with parentheses like @njit(parallel=True)
            def wrapper(f):
                return f

            return wrapper
        else:
            # If used as @njit without parentheses
            return func

    # Fallback prange = normal range
    prange = range

from pydrake.all import (
    MathematicalProgram,
    Solve,
)


class NormBall(NamedTuple):
    center: np.ndarray
    radius: float
    order: float  # p

    @property
    def dual_order(self) -> float:  # q
        if self.order == 1:
            return np.inf
        elif self.order == np.inf:
            return 1
        else:
            return 1 / (1 - 1 / self.order)


@njit
def _MatrixRiccati(A, B, E0, P, p, PN, pN):
    N = A.shape[0]
    nx = A.shape[2]
    nu = B.shape[2]
    nw = E0.shape[1]
    assert A.shape == (N, nx, nx)
    assert B.shape == (N, nx, nu)
    assert E0.shape == (nx, nw)
    assert P.shape == (N, nx + nu, nx + nu)
    assert p.shape == (N, nx + nu, nw)
    assert PN.shape == (nx, nx)
    assert pN.shape == (nx, nw)

    K = np.empty((N, nu, nx))
    k = np.empty((N, nu, nw))
    S = PN.copy()
    s = pN.copy()
    s0 = 0
    for i in range(N - 1, -1, -1):
        Pxx = P[i, :nx, :nx]
        Pux = P[i, nx:, :nx]
        Puu = P[i, nx:, nx:]
        px = p[i, :nx]
        pu = p[i, nx:]
        M = Puu + B[i].T @ S @ B[i]
        F = Pux + B[i].T @ S @ A[i]
        H = Pxx + A[i].T @ S @ A[i]
        f = pu + B[i].T @ s
        h = px + A[i].T @ s
        K[i] = -np.linalg.solve(M, F)
        k[i] = -np.linalg.solve(M, f)
        S = H + F.T @ K[i]
        s = h + F.T @ k[i]
        s0 = s0 - np.trace(k[i].T @ M @ k[i])

    Phi_x = np.empty((N + 1, nx, nw))
    Phi_u = np.empty((N, nu, nw))
    Phi_x[0] = E0
    for i in range(0, N):
        Phi_u[i] = K[i] @ Phi_x[i] + k[i]
        Phi_x[i + 1] = A[i] @ Phi_x[i] + B[i] @ Phi_u[i]

    return Phi_x, Phi_u, S, s, s0


@njit(parallel=True)
def _AllHorizonMatrixRiccati(A, B, E, P, p, PN, pN):
    N = A.shape[0]
    nx = A.shape[2]
    nu = B.shape[2]
    nw = E.shape[2]
    assert A.shape == (N, nx, nx)
    assert B.shape == (N, nx, nu)
    assert E.shape == (N, nx, nw)
    assert P.shape == (N, N, nx + nu, nx + nu)
    assert p.shape == (N, N, nx + nu, nw)
    assert PN.shape == (N, nx, nx)
    assert pN.shape == (N, nx, nw)

    Phi_x = np.zeros((N + 1, N, nx, nw))
    Phi_u = np.zeros((N, N, nu, nw))
    S = np.zeros((N, nx, nx))
    s = np.zeros((N, nx, nw))
    s0 = np.zeros((N,))
    for j in prange(N):
        Phi_x[j + 1 :, j], Phi_u[j + 1 :, j], S[j], s[j], s0[j] = (
            _MatrixRiccati(
                A[j + 1 :],
                B[j + 1 :],
                E[j],
                P[j + 1 :, j],
                p[j + 1 :, j],
                PN[j],
                pN[j],
            )
        )

    return Phi_x, Phi_u, S, s, s0


def _OptimizeUncertaintyTube(
    A, B, E, Q_bar, R_bar, Qf_bar, G, Gf, mu, muN, eta, etaN, w
):
    N = A.shape[0]
    nx = A.shape[2]
    nu = B.shape[2]
    nw = E.shape[2]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert A.shape == (N, nx, nx)
    assert B.shape == (N, nx, nu)
    assert E.shape == (N, nx, nw)
    assert Q_bar.shape == (nx, nx)
    assert R_bar.shape == (nu, nu)
    assert Qf_bar.shape == (nx, nx)
    assert G.shape == (N, nc, nx + nu)
    assert Gf.shape == (nf, nx)
    assert mu.shape == (N, nc)
    assert muN.shape == (nf,)
    assert eta.shape == (N, N, nc)
    assert etaN.shape == (N, nf)
    assert isinstance(w, NormBall)

    if w.order != 2:
        raise RuntimeError("Only supports 2-norm ball.")

    P_bar = np.vstack(
        (
            np.hstack((Q_bar, np.zeros((nx, nu)))),
            np.hstack((np.zeros((nu, nx)), R_bar)),
        )
    )
    P = np.zeros((N, N, nx + nu, nx + nu))
    p = np.zeros((N, N, nx + nu, nw))
    for k in range(N):
        for j in range(k):
            P[k, j] = P_bar + G[k].T @ np.diag(eta[k, j]) @ G[k]
            p[k, j] = 0.5 * G[k].T @ np.outer(mu[k], w.center)

    PN = np.zeros((N, nx, nx))
    pN = np.zeros((N, nx, nw))
    for j in range(N):
        PN[j] = Qf_bar + Gf.T @ np.diag(etaN[j]) @ Gf
        pN[j] = 0.5 * Gf.T @ np.outer(muN, w.center)

    Phi_x, Phi_u, S, s, s0 = _AllHorizonMatrixRiccati(A, B, E, P, p, PN, pN)

    return Phi_x, Phi_u, S, s, s0


@njit
def _beta_func(GPhi, q):
    if q == 2:
        return np.sum(GPhi**2, axis=1)
    elif q == 1:
        return np.sum(np.abs(GPhi), axis=1)
    else:
        raise ValueError(f"Unknown {q}-norm")


@njit
def _Compute_beta(Phi_x, Phi_u, G, Gf, q):
    N = Phi_x.shape[1]
    nx = Phi_x.shape[2]
    nu = Phi_u.shape[2]
    nw = Phi_u.shape[3]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert Phi_x.shape == (N + 1, N, nx, nw)
    assert Phi_u.shape == (N, N, nu, nw)
    assert G.shape == (N, nc, nx + nu)
    assert Gf.shape == (nf, nx)

    if q != 2:
        raise RuntimeError("Only supports q == 2.")

    beta = np.zeros((N, N, nc))
    for k in range(N):
        for j in range(k):
            beta[k, j] = _beta_func(
                G[k] @ np.vstack((Phi_x[k, j], Phi_u[k, j])), q
            )

    betaN = np.zeros((N, nf))
    for j in range(N):
        betaN[j] = _beta_func(Gf @ Phi_x[N, j], q)

    return beta, betaN


def _Compute_hct(Phi_x, Phi_u, G, Gf, w, eps=1e-10):
    N = Phi_x.shape[1]
    nx = Phi_x.shape[2]
    nu = Phi_u.shape[2]
    nw = Phi_u.shape[3]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert Phi_x.shape == (N + 1, N, nx, nw)
    assert Phi_u.shape == (N, N, nu, nw)
    assert G.shape == (N, nc, nx + nu)
    assert Gf.shape == (nf, nx)
    assert isinstance(w, NormBall)
    assert isinstance(eps, float) and eps >= 0

    beta, betaN = _Compute_beta(Phi_x, Phi_u, G, Gf, w.dual_order)

    hct = np.zeros((N, nc))
    for k in range(N):
        for j in range(k):
            hct[k] += w.radius * (beta[k, j] + eps) ** (1 / w.dual_order)

    hctN = np.zeros(nf)
    for j in range(N):
        hctN += w.radius * (betaN[j] + eps) ** (1 / w.dual_order)

    return hct, hctN


def _Compute_hcs(Phi_x, Phi_u, G, Gf, w):
    N = Phi_x.shape[1]
    nx = Phi_x.shape[2]
    nu = Phi_u.shape[2]
    nw = Phi_u.shape[3]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert Phi_x.shape == (N + 1, N, nx, nw)
    assert Phi_u.shape == (N, N, nu, nw)
    assert G.shape == (N, nc, nx + nu)
    assert Gf.shape == (nf, nx)
    assert isinstance(w, NormBall)

    hcs = np.zeros((N, nc))
    for k in range(N):
        for j in range(k):
            hcs[k] += G[k] @ np.vstack((Phi_x[k, j], Phi_u[k, j])) @ w.center

    hcsN = np.zeros(nf)
    for j in range(N):
        hcsN += Gf @ Phi_x[N, j] @ w.center

    return hcs, hcsN


def _Compute_eta(Phi_x, Phi_u, G, Gf, mu, muN, w, eps=1e-10):
    N = Phi_x.shape[1]
    nx = Phi_x.shape[2]
    nu = Phi_u.shape[2]
    nw = Phi_u.shape[3]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert Phi_x.shape == (N + 1, N, nx, nw)
    assert Phi_u.shape == (N, N, nu, nw)
    assert G.shape == (N, nc, nx + nu)
    assert Gf.shape == (nf, nx)
    assert mu.shape == (N, nc)
    assert muN.shape == (nf,)
    assert isinstance(w, NormBall)
    assert isinstance(eps, float) and eps >= 0

    q = w.dual_order
    beta, betaN = _Compute_beta(Phi_x, Phi_u, G, Gf, q)

    eta = np.zeros((N, N, nc))
    for k in range(N):
        for j in range(k):
            eta[k, j] = w.radius / q * (beta[k, j] + eps) ** (1 / q - 1) * mu[k]

    etaN = np.zeros((N, nf))
    for j in range(N):
        etaN[j] = w.radius / q * (betaN[j] + eps) ** (1 / q - 1) * muN

    return eta, etaN


def _OptimizeNominalTrajectory(
    A,
    B,
    x0,
    cost_func,
    G,
    g,
    Gf,
    gf,
    Phi_x=None,
    Phi_u=None,
    w=None,
    eps=1e-10,
    additional_constraints_func=None,
    slack_penalty=None,
):
    N = A.shape[0]
    nx = A.shape[2]
    nu = B.shape[2]
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert A.shape == (N, nx, nx)
    assert B.shape == (N, nx, nu)
    assert x0.shape == (nx,)
    assert G.shape == (N, nc, nx + nu)
    assert g.shape == (N, nc)
    assert Gf.shape == (nf, nx)
    assert gf.shape == (nf,)

    assert (Phi_x is None) == (Phi_u is None) == (w is None)
    constraint_tightening = w is not None
    if constraint_tightening:
        nw = Phi_x.shape[3]
        assert Phi_x.shape == (N + 1, N, nx, nw)
        assert Phi_u.shape == (N, N, nu, nw)
        assert isinstance(w, NormBall)
        assert isinstance(eps, float) and eps >= 0

    b = -g.astype(float)
    bN = -gf.astype(float)
    if constraint_tightening:
        hct, hctN = _Compute_hct(Phi_x, Phi_u, G, Gf, w, eps)
        hcs, hcsN = _Compute_hcs(Phi_x, Phi_u, G, Gf, w)
        b -= hct + hcs
        bN -= hctN + hcsN

    prog = MathematicalProgram()
    z = prog.NewContinuousVariables(N + 1, nx, "z")
    v = prog.NewContinuousVariables(N, nu, "v")

    prog.AddCost(cost_func(z, v))

    prog.AddLinearEqualityConstraint(z[0], x0)
    for k in range(N):
        prog.AddLinearEqualityConstraint(
            z[k + 1] - (A[k] @ z[k] + B[k] @ v[k]), np.zeros(nx)
        )

    slack_vars = []

    if nc != 0:
        constr = []
        for k in range(N):
            if slack_penalty is None:
                constrk = prog.AddLinearConstraint(
                    G[k],
                    lb=np.full(nc, -np.inf),
                    ub=b[k],
                    vars=np.concatenate((z[k], v[k])),
                )
            else:
                s, Sel = _NewSlackVariables(prog, G[k])
                constrk = prog.AddLinearConstraint(
                    np.hstack((G[k], -Sel)),
                    lb=np.full(nc, -np.inf),
                    ub=b[k],
                    vars=np.concatenate((z[k], v[k], s)),
                )
                slack_vars.append(s)
            constr.append(constrk)

    if nf != 0:
        if slack_penalty is None:
            constrN = prog.AddLinearConstraint(
                Gf,
                lb=np.full(nf, -np.inf),
                ub=bN,
                vars=z[N],
            )
        else:
            s, Sel = _NewSlackVariables(prog, Gf)
            constrN = prog.AddLinearConstraint(
                np.hstack((Gf, -Sel)),
                lb=np.full(nf, -np.inf),
                ub=bN,
                vars=np.concatenate((z[N], s)),
            )
            slack_vars.append(s)

    if len(slack_vars):
        slack_vars = np.concatenate(slack_vars)
        prog.AddLinearCost(
            np.ones(len(slack_vars)) * slack_penalty,
            slack_vars,
        )

    if additional_constraints_func is not None:
        additional_constraints_func(prog, z, v)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError(
            "Failed to solve nominal trajectory optimization due to "
            + str(result.get_solution_result()).split(".")[-1]
        )

    z_val = result.GetSolution(z).reshape(z.shape)
    v_val = result.GetSolution(v).reshape(v.shape)

    mu_val = np.zeros((N, nc))
    if nc != 0:
        for k in range(N):
            mu_val[k] = -result.GetDualSolution(constr[k])
    muN_val = -result.GetDualSolution(constrN) if nf != 0 else np.zeros(nf)

    return z_val, v_val, mu_val, muN_val


def _NewSlackVariables(prog, G):
    unique_G, indices = np.unique(G != 0, axis=0, return_inverse=True)
    s = prog.NewContinuousVariables(len(unique_G), "s")
    prog.AddLinearConstraint(
        s, lb=np.zeros(s.shape), ub=np.full(s.shape, np.inf)
    )
    Sel = np.zeros((len(G), len(s)))
    Sel[np.arange(len(G)), indices] = 1.0
    return s, Sel


def FastSLS(
    A,
    B,
    E,
    x0,
    cost_func,
    Q_bar,
    R_bar,
    Qf_bar,
    G,
    g,
    Gf,
    gf,
    w,
    eps=1e-10,
    additional_constraints_func=None,
    max_iters=10,
    tolerance=1e-6,
    slack_penalty=None,
):
    N = A.shape[0]
    nx = A.shape[2]
    nu = B.shape[2]
    nw = len(w.center)
    nc = G.shape[1]
    nf = Gf.shape[0]
    assert A.shape == (N, nx, nx)
    assert B.shape == (N, nx, nu)
    assert callable(E) or E.shape == (N, nx, nw)
    assert x0.shape == (nx,)
    assert Q_bar.shape == (nx, nx)
    assert R_bar.shape == (nu, nu)
    assert Qf_bar.shape == (nx, nx)
    assert G.shape == (N, nc, nx + nu)
    assert g.shape == (N, nc)
    assert Gf.shape == (nf, nx)
    assert gf.shape == (nf,)
    assert isinstance(w, NormBall)
    assert isinstance(eps, float) and eps >= 0.0
    assert slack_penalty is None or slack_penalty > 0.0

    z, v, mu, muN = _OptimizeNominalTrajectory(
        A=A,
        B=B,
        x0=x0,
        cost_func=cost_func,
        G=G,
        g=g,
        Gf=Gf,
        gf=gf,
        additional_constraints_func=additional_constraints_func,
        slack_penalty=slack_penalty,
    )
    eta = np.zeros((N, N, nc))
    etaN = np.zeros((N, nf))

    Phi_x, Phi_u, S, s, s0 = _OptimizeUncertaintyTube(
        A=A,
        B=B,
        E=E(z, v) if callable(E) else E,
        Q_bar=Q_bar,
        R_bar=R_bar,
        Qf_bar=Qf_bar,
        G=G,
        Gf=Gf,
        mu=mu,
        muN=muN,
        eta=eta,
        etaN=etaN,
        w=w,
    )

    def total_cost_func(z, v):
        cost = cost_func(z, v)
        if callable(E):
            E_new = E(z, v)
            assert E_new.shape == (N, nx, nw)
            for k in range(N):
                cost += np.trace(E_new[k].T @ S[k] @ E_new[k])
                cost += np.trace(2 * s[k].T @ E_new[k])
                cost += s0[k]
        return cost

    diff = np.inf
    for iter in range(max_iters):
        z_new, v_new, mu_new, muN_new = _OptimizeNominalTrajectory(
            A=A,
            B=B,
            x0=x0,
            cost_func=total_cost_func,
            G=G,
            g=g,
            Gf=Gf,
            gf=gf,
            Phi_x=Phi_x,
            Phi_u=Phi_u,
            w=w,
            eps=eps,
            additional_constraints_func=additional_constraints_func,
            slack_penalty=slack_penalty,
        )
        diff = max(
            np.abs(z - z_new).max(),
            np.abs(v - v_new).max(),
            np.abs(mu - mu_new).max() if nc != 0 else -np.inf,
            np.abs(muN - muN_new).max() if nf != 0 else -np.inf,
        )
        z, v, mu, muN = z_new, v_new, mu_new, muN_new

        eta, etaN = _Compute_eta(Phi_x, Phi_u, G, Gf, mu, muN, w, eps)
        Phi_x_new, Phi_u_new, S, s, s0 = _OptimizeUncertaintyTube(
            A=A,
            B=B,
            E=E(z, v) if callable(E) else E,
            Q_bar=Q_bar,
            R_bar=R_bar,
            Qf_bar=Qf_bar,
            G=G,
            Gf=Gf,
            mu=mu,
            muN=muN,
            eta=eta,
            etaN=etaN,
            w=w,
        )
        diff = max(
            diff,
            np.abs(Phi_x - Phi_x_new).max(),
            np.abs(Phi_u - Phi_u_new).max(),
        )
        Phi_x, Phi_u = Phi_x_new, Phi_u_new

        if diff < tolerance:
            break

    K = _Compute_K(Phi_x, Phi_u)

    return z, v, Phi_x, Phi_u, K


def _Compute_K(Phi_x, Phi_u):
    N = Phi_x.shape[1]
    nx = Phi_x.shape[2]
    nu = Phi_u.shape[2]
    nw = Phi_u.shape[3]
    assert Phi_x.shape == (N + 1, N, nx, nw)
    assert Phi_u.shape == (N, N, nu, nw)

    Phi_x = Phi_x[1:, :, :, :]

    Phi_x_inv = np.zeros((N, N, nw, nx))
    for k in range(N):
        for j in range(k, -1, -1):
            if j == k:
                Phi_x_inv[k, k] = np.linalg.pinv(Phi_x[k, k])
            else:
                Phi_x_inv[k, j] = (
                    -np.sum(
                        [
                            Phi_x_inv[k, i] @ Phi_x[i, j]
                            for i in range(j + 1, k + 1)
                        ],
                        axis=0,
                    )
                    @ Phi_x_inv[j, j]
                )

    K = np.zeros((N, N + 1, nu, nx))
    for k in range(N):
        for j in range(k + 1):
            K[k, j + 1] = np.sum(
                [Phi_u[k, i] @ Phi_x_inv[i, j] for i in range(j, k + 1)], axis=0
            )

    return K
