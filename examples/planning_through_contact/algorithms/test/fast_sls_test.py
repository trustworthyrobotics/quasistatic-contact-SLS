import unittest

import numpy as np

from examples.planning_through_contact.algorithms.fast_sls import (
    FastSLS,
    NormBall,
    _AllHorizonMatrixRiccati,
    _Compute_beta,
    _Compute_eta,
    _Compute_hcs,
    _Compute_hct,
    _Compute_K,
    _MatrixRiccati,
    _OptimizeNominalTrajectory,
    _OptimizeUncertaintyTube,
)
from pydrake.all import Expression, MathematicalProgram, Solve


class TestNormBall(unittest.TestCase):
    def test_norm_ball(self):
        center = np.ones(3) * 1.5
        radius = 0.5
        ball = NormBall(center=center, radius=radius, order=2)
        self.assertEqual(ball.order, 2)
        self.assertEqual(ball.dual_order, 2)

        ball = NormBall(center=center, radius=radius, order=np.inf)
        self.assertEqual(ball.order, np.inf)
        self.assertEqual(ball.dual_order, 1)

        ball = NormBall(center=center, radius=radius, order=1)
        self.assertEqual(ball.order, 1)
        self.assertEqual(ball.dual_order, np.inf)


class TestFastSLS(unittest.TestCase):
    def _test_matrix_riccati(self, nw=1, N=10):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        nx, nu = 2, 1

        P = np.ones((nx + nu, nx + nu)) + 10 * np.identity(nx + nu)
        p = np.ones((nx + nu, nw))

        A = np.repeat(A[np.newaxis, :, :], N, axis=0)
        B = np.repeat(B[np.newaxis, :, :], N, axis=0)
        E0 = 0.01 * np.ones((nx, nw))
        P = np.repeat(P[np.newaxis, :, :], N, axis=0)
        p = np.repeat(p[np.newaxis, :, :], N, axis=0)
        PN = np.ones((nx, nx)) + 10 * np.identity(nx)
        pN = np.ones((nx, nw))

        # Solve via Riccati recursion
        Phi_x, Phi_u, S, s, s0 = _MatrixRiccati(A, B, E0, P, p, PN, pN)
        self.assertEqual(Phi_x.shape, (N + 1, nx, nw))
        self.assertEqual(Phi_u.shape, (N, nu, nw))

        # Solve via mathematical program
        prog = MathematicalProgram()
        Phi_x_var = [
            prog.NewContinuousVariables(nx, nw, rf"\Phi^x_{k}")
            for k in range(N + 1)
        ]
        Phi_u_var = [
            prog.NewContinuousVariables(nu, nw, rf"\Phi^u_{k}")
            for k in range(N)
        ]
        for k in range(N):
            Phi_var = np.vstack((Phi_x_var[k], Phi_u_var[k]))
            prog.AddCost(
                np.trace(Phi_var.T @ P[k] @ Phi_var + 2 * p[k].T @ Phi_var)
            )
        prog.AddCost(
            np.trace(
                Phi_x_var[N].T @ PN @ Phi_x_var[N] + 2 * pN.T @ Phi_x_var[N]
            )
        )
        for k in range(N):
            prog.AddLinearEqualityConstraint(
                (
                    Phi_x_var[k + 1]
                    - (A[k] @ Phi_x_var[k] + B[k] @ Phi_u_var[k])
                ).flatten(),
                np.zeros(nx * nw),
            )
        prog.AddLinearEqualityConstraint(Phi_x_var[0].flatten(), E0.flatten())
        result = Solve(prog)
        self.assertTrue(result.is_success())

        # Compare the results
        for k in range(N + 1):
            np.testing.assert_allclose(
                Phi_x[k],
                result.GetSolution(Phi_x_var[k]).reshape(nx, nw),
            )
        for k in range(0, N):
            np.testing.assert_allclose(
                Phi_u[k],
                result.GetSolution(Phi_u_var[k]).reshape(nu, nw),
            )

        np.testing.assert_allclose(
            np.trace(E0.T @ S @ E0 + 2 * s.T @ E0) + s0,
            result.get_optimal_cost(),
        )

    def test_matrix_riccati(self):
        self._test_matrix_riccati(nw=1)
        self._test_matrix_riccati(nw=2)

    @staticmethod
    def newVariablePhi(prog, N, nx, nu, nw):
        Phi_x = np.empty((N + 1, N), dtype=object)
        Phi_u = np.empty((N, N), dtype=object)
        for k in range(N + 1):
            for j in range(k):
                Phi_x[k, j] = prog.NewContinuousVariables(
                    nx, nw, rf"\Phi^x_{{{k},{j}}}"
                )
        for k in range(N):
            for j in range(k):
                Phi_u[k, j] = prog.NewContinuousVariables(
                    nu, nw, rf"\Phi^u_{{{k},{j}}}"
                )
        return Phi_x, Phi_u

    @staticmethod
    def getSolutionPhi(prog_result, Phi_var):
        Phi = np.zeros(Phi_var.shape + Phi_var[1, 0].shape)
        for k in range(Phi.shape[0]):
            for j in range(k):
                Phi[k, j] = prog_result.GetSolution(Phi_var[k, j]).reshape(
                    Phi_var[k, j].shape
                )
        return Phi

    def _test_all_horizon_matrix_riccati(self, nw=1, N=10):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        nx, nu = 2, 1
        E = 0.01 * np.ones((nx, nw))

        P = np.ones((nx + nu, nx + nu)) + 10 * np.identity(nx + nu)
        p = np.ones((nx + nu, nw))
        PN = np.ones((nx, nx)) + 10 * np.identity(nx)
        pN = np.ones((nx, nw))

        A = np.repeat(A[np.newaxis, :, :], N, axis=0)
        B = np.repeat(B[np.newaxis, :, :], N, axis=0)
        E = np.repeat(E[np.newaxis, :, :], N, axis=0)
        P = np.tile(P, (N, N, 1, 1))
        p = np.tile(p, (N, N, 1, 1))
        PN = np.tile(PN, (N, 1, 1))
        pN = np.tile(pN, (N, 1, 1))

        # Solve via Riccati recursion
        Phi_x, Phi_u, S, s, s0 = _AllHorizonMatrixRiccati(A, B, E, P, p, PN, pN)
        self.assertEqual(Phi_x.shape, (N + 1, N, nx, nw))
        self.assertEqual(Phi_u.shape, (N, N, nu, nw))

        cost = 0
        for k in range(N):
            cost += np.trace(E[k].T @ S[k] @ E[k] + 2 * s[k].T @ E[k]) + s0[k]

        # Solve via mathematical program
        prog = MathematicalProgram()
        Phi_x_var, Phi_u_var = self.newVariablePhi(prog, N, nx, nu, nw)

        for j in range(N):
            for k in range(j + 1, N):
                Phi_var = np.vstack((Phi_x_var[k, j], Phi_u_var[k, j]))
                prog.AddCost(
                    np.trace(
                        Phi_var.T @ P[k, j] @ Phi_var + 2 * p[k, j].T @ Phi_var
                    )
                )
            prog.AddCost(
                np.trace(
                    Phi_x_var[N, j].T @ PN[j] @ Phi_x_var[N, j]
                    + 2 * pN[j].T @ Phi_x_var[N, j]
                )
            )
        for j in range(N):
            prog.AddLinearEqualityConstraint(
                Phi_x_var[j + 1, j].flatten(), E[j].flatten()
            )
            for k in range(j + 1, N):
                prog.AddLinearEqualityConstraint(
                    (
                        Phi_x_var[k + 1, j]
                        - (A[k] @ Phi_x_var[k, j] + B[k] @ Phi_u_var[k, j])
                    ).flatten(),
                    np.zeros(nx * nw),
                )
        result = Solve(prog)
        self.assertTrue(result.is_success())
        Phi_x_expected = self.getSolutionPhi(result, Phi_x_var)
        Phi_u_expected = self.getSolutionPhi(result, Phi_u_var)

        # Compare the results
        np.testing.assert_allclose(Phi_x, Phi_x_expected)
        np.testing.assert_allclose(Phi_u, Phi_u_expected)
        np.testing.assert_allclose(cost, result.get_optimal_cost())

    def test_all_horizon_matrix_riccati(self):
        self._test_all_horizon_matrix_riccati(nw=1)
        self._test_all_horizon_matrix_riccati(nw=2)

    def _test_optimize_uncertainty_tube(self, nw=1, N=10):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        nx, nu = 2, 1
        E = 0.01 * np.ones((nx, nw))

        Q_bar = np.ones((nx, nx)) + 10 * np.identity(nx)
        R_bar = np.ones((nu, nu)) + 10 * np.identity(nu)
        Qf_bar = np.ones((nx, nx)) + 10 * np.identity(nx)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        Gf = np.array([[-1, 0], [1, 0]])
        nc, nf = G.shape[0], Gf.shape[0]

        A = np.repeat(A[np.newaxis, :, :], N, axis=0)
        B = np.repeat(B[np.newaxis, :, :], N, axis=0)
        E = np.repeat(E[np.newaxis, :, :], N, axis=0)
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)

        np.random.seed(0)
        mu = 0.01 * np.random.rand(N, nc)
        muN = 0.01 * np.random.rand(nf)
        eta = 0.01 * np.random.randn(N, N, nc)
        etaN = 0.01 * np.random.randn(N, nf)

        w = NormBall(center=0.001 * np.ones(nw), radius=0.001, order=2)

        # Solve via Riccati recursion
        Phi_x, Phi_u, S, s, s0 = _OptimizeUncertaintyTube(
            A, B, E, Q_bar, R_bar, Qf_bar, G, Gf, mu, muN, eta, etaN, w
        )
        self.assertEqual(Phi_x.shape, (N + 1, N, nx, nw))
        self.assertEqual(Phi_u.shape, (N, N, nu, nw))

        cost = 0
        for k in range(N):
            cost += np.trace(E[k].T @ S[k] @ E[k] + 2 * s[k].T @ E[k]) + s0[k]

        # Solve via mathematical program
        prog = MathematicalProgram()
        Phi_x_var, Phi_u_var = self.newVariablePhi(prog, N, nx, nu, nw)

        for j in range(N):
            for k in range(j + 1, N):
                prog.AddCost(
                    np.trace(Phi_x_var[k, j].T @ Q_bar @ Phi_x_var[k, j])
                )
                prog.AddCost(
                    np.trace(Phi_u_var[k, j].T @ R_bar @ Phi_u_var[k, j])
                )
            prog.AddCost(np.trace(Phi_x_var[N, j].T @ Qf_bar @ Phi_x_var[N, j]))

        for k in range(N):
            for j in range(k):
                prog.AddCost(
                    mu[k].dot(
                        G[k]
                        @ np.vstack((Phi_x_var[k, j], Phi_u_var[k, j]))
                        @ w.center
                    )
                )
        for j in range(N):
            prog.AddCost(muN.dot(Gf @ Phi_x_var[N, j] @ w.center))

        def l2_norm_squared(a):
            return np.sum(a**2, axis=1)

        for k in range(N):
            for j in range(k):
                prog.AddCost(
                    eta[k, j].dot(
                        l2_norm_squared(
                            G[k] @ np.vstack((Phi_x_var[k, j], Phi_u_var[k, j]))
                        )
                    )
                )
        for j in range(N):
            prog.AddCost(etaN[j].dot(l2_norm_squared(Gf @ Phi_x_var[N, j])))

        for j in range(N):
            prog.AddLinearEqualityConstraint(
                Phi_x_var[j + 1, j].flatten(), E[j].flatten()
            )
            for k in range(j + 1, N):
                prog.AddLinearEqualityConstraint(
                    (
                        Phi_x_var[k + 1, j]
                        - (A[k] @ Phi_x_var[k, j] + B[k] @ Phi_u_var[k, j])
                    ).flatten(),
                    np.zeros(nx * nw),
                )
        result = Solve(prog)
        Phi_x_expected = self.getSolutionPhi(result, Phi_x_var)
        Phi_u_expected = self.getSolutionPhi(result, Phi_u_var)

        # Compare the results
        np.testing.assert_allclose(Phi_x, Phi_x_expected, atol=1e-10)
        np.testing.assert_allclose(Phi_u, Phi_u_expected, atol=1e-10)
        np.testing.assert_allclose(cost, result.get_optimal_cost())

    def test_optimize_uncertainty_tube(self):
        self._test_optimize_uncertainty_tube(nw=1)
        self._test_optimize_uncertainty_tube(nw=2)

    @staticmethod
    def randomPhi(N, nx, nu, nw):
        np.random.seed(0)
        Phi_x = np.random.randn(N + 1, N, nx, nw)
        for k in range(Phi_x.shape[0]):
            for j in range(k, Phi_x.shape[1]):
                Phi_x[k, j] = 0.0
        Phi_u = np.random.randn(N, N, nu, nw)
        for k in range(Phi_u.shape[0]):
            for j in range(k, Phi_u.shape[1]):
                Phi_u[k, j] = 0.0
        return Phi_x, Phi_u

    def _test_compute_beta(self, nw, N=10):
        nx, nu = 2, 1
        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        Gf = np.array([[-1, 0], [1, 0]])
        nc, nf = G.shape[0], Gf.shape[0]
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)

        # Function return value
        beta, betaN = _Compute_beta(Phi_x, Phi_u, G, Gf, q=2)
        self.assertEqual(beta.shape, (N, N, nc))
        self.assertEqual(betaN.shape, (N, nf))

        # Expected value
        def l2_norm_squared(a):
            return np.sum(a**2, axis=1)

        beta_expected = np.zeros((N, N, nc))
        for k in range(N):
            for j in range(k):
                beta_expected[k, j] = l2_norm_squared(
                    G[k] @ np.vstack((Phi_x[k, j], Phi_u[k, j]))
                )

        betaN_expected = np.zeros((N, nf))
        for j in range(N):
            betaN_expected[j] = l2_norm_squared(Gf @ Phi_x[N, j])

        np.testing.assert_allclose(beta, beta_expected)
        np.testing.assert_allclose(betaN, betaN_expected)

    def test_compute_beta(self):
        self._test_compute_beta(nw=1)
        self._test_compute_beta(nw=2)

    def _test_compute_hct(self, nw, N=10):
        nx, nu = 2, 1
        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        Gf = np.array([[-1, 0], [1, 0]])
        nc, nf = G.shape[0], Gf.shape[0]
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)

        w = NormBall(center=np.zeros(nw), radius=0.1, order=2)
        eps = 1e-10

        # Function return value
        hct, hctN = _Compute_hct(Phi_x, Phi_u, G, Gf, w, eps)
        self.assertEqual(hct.shape, (N, nc))
        self.assertEqual(hctN.shape, (nf,))

        # Expected value
        def l2_norm_squared(a):
            return np.sum(a**2, axis=1)

        hct_expected = np.zeros((N, nc))
        for k in range(N):
            for j in range(k):
                beta = l2_norm_squared(
                    G[k] @ np.vstack((Phi_x[k, j], Phi_u[k, j]))
                )
                hct_expected[k] += w.radius * (beta + eps) ** 0.5

        hctN_expected = np.zeros(nf)
        for j in range(N):
            beta = l2_norm_squared(Gf @ Phi_x[N, j])
            hctN_expected += w.radius * (beta + eps) ** 0.5

        np.testing.assert_allclose(hct, hct_expected)
        np.testing.assert_allclose(hctN, hctN_expected)

    def test_compute_hct(self):
        self._test_compute_hct(nw=1)
        self._test_compute_hct(nw=2)

    def _test_compute_hcs(self, nw, N=10):
        nx, nu = 2, 1
        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        Gf = np.array([[-1, 0], [1, 0]])
        nc, nf = G.shape[0], Gf.shape[0]
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)

        w = NormBall(center=0.01 * np.ones(nw), radius=0.01, order=2)

        # Function return value
        hcs, hcsN = _Compute_hcs(Phi_x, Phi_u, G, Gf, w)
        self.assertEqual(hcs.shape, (N, nc))
        self.assertEqual(hcsN.shape, (nf,))

        # Expected value
        hcs_expected = np.zeros((N, nc))
        for k in range(N):
            for j in range(k):
                hcs_expected[k] += (
                    G[k] @ np.vstack((Phi_x[k, j], Phi_u[k, j])) @ w.center
                )

        hcsN_expected = np.zeros(nf)
        for j in range(N):
            hcsN_expected += Gf @ Phi_x[N, j] @ w.center

        np.testing.assert_allclose(hcs, hcs_expected)
        np.testing.assert_allclose(hcsN, hcsN_expected)

    def test_compute_hcs(self):
        self._test_compute_hcs(nw=1)
        self._test_compute_hcs(nw=2)

    def _test_compute_eta(self, nw, N=10):
        nx, nu = 2, 1
        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        Gf = np.array([[-1, 0], [1, 0]])
        nc, nf = G.shape[0], Gf.shape[0]
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)

        mu = np.random.rand(N, nc)
        muN = np.random.rand(nf)

        w = NormBall(center=np.zeros(nw), radius=0.1, order=2)
        eps = 1e-10

        # Function return value
        eta, etaN = _Compute_eta(Phi_x, Phi_u, G, Gf, mu, muN, w, eps)
        self.assertEqual(eta.shape, (N, N, nc))
        self.assertEqual(etaN.shape, (N, nf))

        # Expected value
        eta_expected = np.zeros((N, N, nc))
        etaN_expected = np.zeros((N, nf))

        q = w.dual_order
        beta, betaN = _Compute_beta(Phi_x, Phi_u, G, Gf, q=q)
        for k in range(N):
            for j in range(k):
                eta_expected[k, j] = (
                    w.radius / q * (beta[k, j] + eps) ** (1 / q - 1) * mu[k]
                )
        for j in range(N):
            etaN_expected[j] = (
                w.radius / q * (betaN[j] + eps) ** (1 / q - 1) * muN
            )

        # Compare values
        np.testing.assert_allclose(eta, eta_expected)
        np.testing.assert_allclose(etaN, etaN_expected)

    def test_compute_eta(self):
        self._test_compute_eta(nw=1)
        self._test_compute_eta(nw=2)

    def _test_optimize_nominal_trajectory(
        self, nw, N=20, constraint_tighten=True
    ):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        nx, nu = 2, 1
        x0 = np.array([0.0, 0.0])

        l_xx = np.ones((nx, nx)) + 10 * np.identity(nx)
        l_x = np.ones(nx)
        l_uu = np.ones((nu, nu)) + 10 * np.identity(nu)
        l_u = np.ones(nu)
        l_xu = np.ones((nx, nu)) * 0.1
        lf_xx = np.ones((nx, nx)) + 10 * np.identity(nx)
        lf_x = np.ones(nx)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        g = np.array([-5, -5, -100, -100])
        Gf = np.array([[-1, 0], [1, 0]])
        gf = np.array([0.99, -1.01])
        nc, nf = G.shape[0], Gf.shape[0]

        A = np.repeat(A[np.newaxis, :, :], N, axis=0)
        B = np.repeat(B[np.newaxis, :, :], N, axis=0)
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)
        g = np.repeat(g[np.newaxis, :], N, axis=0)

        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)
        w = NormBall(center=np.ones(nw) * 1e-4, radius=1e-4, order=2)
        eps = 1e-10

        def cost_func(x, u):
            assert x.shape == (N + 1, nx)
            assert u.shape == (N, nu)
            cost = 0
            for k in range(N):
                cost += 0.5 * x[k].dot(l_xx @ x[k]) + l_x.dot(x[k])
                cost += 0.5 * u[k].dot(l_uu @ u[k]) + l_u.dot(u[k])
                cost += x[k].dot(l_xu @ u[k])
            cost += 0.5 * x[N].dot(lf_xx @ x[N]) + lf_x.dot(x[N])
            return cost

        # Result from the tested function
        z, v, mu, muN = _OptimizeNominalTrajectory(
            A,
            B,
            x0,
            cost_func,
            G,
            g,
            Gf,
            gf,
            Phi_x=Phi_x if constraint_tighten else None,
            Phi_u=Phi_u if constraint_tighten else None,
            w=w if constraint_tighten else None,
            eps=eps,
        )
        self.assertEqual(z.shape, (N + 1, nx))
        self.assertEqual(v.shape, (N, nu))
        self.assertEqual(mu.shape, (N, nc))
        self.assertEqual(muN.shape, (nf,))

        # Expected value
        prog = MathematicalProgram()
        z_var = prog.NewContinuousVariables(N + 1, nx, "z")
        v_var = prog.NewContinuousVariables(N, nu, "v")
        for k in range(N):
            prog.AddQuadraticCost(
                0.5 * z_var[k].dot(l_xx @ z_var[k])
                + l_x.dot(z_var[k])
                + 0.5 * v_var[k].dot(l_uu @ v_var[k])
                + l_u.dot(v_var[k])
                + z_var[k].dot(l_xu @ v_var[k])
            )
        prog.AddQuadraticCost(
            0.5 * z_var[N].dot(lf_xx @ z_var[N]) + lf_x.dot(z_var[N])
        )

        prog.AddLinearEqualityConstraint(z_var[0], x0)
        for k in range(N):
            prog.AddLinearEqualityConstraint(
                z_var[k + 1] - (A[k] @ z_var[k] + B[k] @ v_var[k]), np.zeros(nx)
            )

        if constraint_tighten:
            hcs, hcsN = _Compute_hcs(Phi_x, Phi_u, G, Gf, w)
            hct, hctN = _Compute_hct(Phi_x, Phi_u, G, Gf, w, eps)
            constr = []
            for k in range(N):
                constrk = prog.AddLinearConstraint(
                    G[k],
                    lb=np.full(nc, -np.inf),
                    ub=-g[k] - hcs[k] - hct[k],
                    vars=np.concatenate((z_var[k], v_var[k])),
                )
                constr.append(constrk)
            constrN = prog.AddLinearConstraint(
                Gf,
                lb=np.full(nf, -np.inf),
                ub=-gf - hcsN - hctN,
                vars=z_var[N],
            )
        else:
            constr = []
            for k in range(N):
                constrk = prog.AddLinearConstraint(
                    G[k],
                    lb=np.full(nc, -np.inf),
                    ub=-g[k],
                    vars=np.concatenate((z_var[k], v_var[k])),
                )
                constr.append(constrk)
            constrN = prog.AddLinearConstraint(
                Gf,
                lb=np.full(nf, -np.inf),
                ub=-gf,
                vars=z_var[N],
            )
        result = Solve(prog)
        assert result.is_success()
        z_expected = result.GetSolution(z_var).reshape(z_var.shape)
        v_expected = result.GetSolution(v_var).reshape(v_var.shape)

        mu_expected = np.zeros((N, nc))
        for k in range(N):
            mu_expected[k] = -result.GetDualSolution(constr[k])
            assert np.all(mu_expected[k] >= 0)
        muN_expected = np.zeros(nf)
        muN_expected[:] = -result.GetDualSolution(constrN)
        assert np.all(muN_expected >= 0)

        # Compare results
        np.testing.assert_allclose(z, z_expected, atol=1e-10)
        np.testing.assert_allclose(v, v_expected, atol=1e-10)
        np.testing.assert_allclose(mu, mu_expected, atol=1e-10)
        np.testing.assert_allclose(muN, muN_expected, atol=1e-10)

    def test_optimize_nominal_trajectory_with_tightening(self):
        self._test_optimize_nominal_trajectory(nw=1, constraint_tighten=True)
        self._test_optimize_nominal_trajectory(nw=2, constraint_tighten=True)

    def test_optimize_nominal_trajectory_without_tightening(self):
        self._test_optimize_nominal_trajectory(nw=1, constraint_tighten=False)
        self._test_optimize_nominal_trajectory(nw=2, constraint_tighten=False)

    def _test_fast_sls(self, nw, N=20, slack_penalty=None):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        nx, nu = 2, 1
        E = np.ones((nx, nw))
        x0 = np.array([0.0, 0.0])

        l_xx = np.ones((nx, nx)) + 10 * np.identity(nx)
        l_x = np.ones(nx)
        l_uu = np.ones((nu, nu)) + 10 * np.identity(nu)
        l_u = np.ones(nu)
        l_xu = np.ones((nx, nu)) * 0.1
        lf_xx = np.ones((nx, nx)) + 10 * np.identity(nx)
        lf_x = np.ones(nx)

        Q_bar = np.ones((nx, nx)) + 10 * np.identity(nx)
        R_bar = np.ones((nu, nu)) + 10 * np.identity(nu)
        Qf_bar = np.ones((nx, nx)) + 10 * np.identity(nx)

        G = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]])
        g = np.array([-5, -5, -100, -100])
        Gf = np.array([[-1, 0], [1, 0]])
        gf = np.array([0.99, -1.01])
        nc, nf = G.shape[0], Gf.shape[0]

        A = np.repeat(A[np.newaxis, :, :], N, axis=0)
        B = np.repeat(B[np.newaxis, :, :], N, axis=0)
        E = np.repeat(E[np.newaxis, :, :], N, axis=0)
        G = np.repeat(G[np.newaxis, :, :], N, axis=0)
        g = np.repeat(g[np.newaxis, :], N, axis=0)

        w = NormBall(center=np.ones(nw) * 1e-4, radius=1e-4, order=2)
        eps = 1e-10

        def cost_func(x, u):
            assert x.shape == (N + 1, nx)
            assert u.shape == (N, nu)
            cost = 0
            for k in range(N):
                cost += 0.5 * x[k].dot(l_xx @ x[k]) + l_x.dot(x[k])
                cost += 0.5 * u[k].dot(l_uu @ u[k]) + l_u.dot(u[k])
                cost += x[k].dot(l_xu @ u[k])
            cost += 0.5 * x[N].dot(lf_xx @ x[N]) + lf_x.dot(x[N])
            return cost

        # Solve via fast-SLS algorithm
        z, v, Phi_x, Phi_u, K = FastSLS(
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
            w=w,
            eps=eps,
            slack_penalty=slack_penalty,
        )

        # If slack variables are not involved, we do not check the correctness
        # of the result. Solvable is good enough for us for now.
        if slack_penalty is not None:
            return

        # Solve via mathematical program
        prog = MathematicalProgram()
        z_var = prog.NewContinuousVariables(N + 1, nx, "z")
        v_var = prog.NewContinuousVariables(N, nu, "v")
        Phi_x_var, Phi_u_var = self.newVariablePhi(prog, N, nx, nu, nw)

        for k in range(N):
            prog.AddQuadraticCost(
                0.5 * z_var[k].dot(l_xx @ z_var[k])
                + l_x.dot(z_var[k])
                + 0.5 * v_var[k].dot(l_uu @ v_var[k])
                + l_u.dot(v_var[k])
                + z_var[k].dot(l_xu @ v_var[k])
            )
        prog.AddQuadraticCost(
            0.5 * z_var[N].dot(lf_xx @ z_var[N]) + lf_x.dot(z_var[N])
        )

        for j in range(N):
            for k in range(j + 1, N):
                prog.AddCost(
                    np.trace(Phi_x_var[k, j].T @ Q_bar @ Phi_x_var[k, j])
                )
                prog.AddCost(
                    np.trace(Phi_u_var[k, j].T @ R_bar @ Phi_u_var[k, j])
                )
            prog.AddCost(np.trace(Phi_x_var[N, j].T @ Qf_bar @ Phi_x_var[N, j]))

        prog.AddLinearEqualityConstraint(z_var[0], x0)
        for k in range(N):
            prog.AddLinearEqualityConstraint(
                z_var[k + 1] - (A[k] @ z_var[k] + B[k] @ v_var[k]), np.zeros(nx)
            )

        for j in range(N):
            prog.AddLinearEqualityConstraint(
                Phi_x_var[j + 1, j].flatten(), E[j].flatten()
            )
            for k in range(j + 1, N):
                prog.AddLinearEqualityConstraint(
                    (
                        Phi_x_var[k + 1, j]
                        - (A[k] @ Phi_x_var[k, j] + B[k] @ Phi_u_var[k, j])
                    ).flatten(),
                    np.zeros(nx * nw),
                )

        def soft_l2_norm(a):
            return np.sqrt(np.sum(a**2, axis=1) + eps)

        for k in range(N):
            ct = np.array([Expression(0) for _ in range(nc)])
            for j in range(k):
                Phi_var = np.vstack((Phi_x_var[k, j], Phi_u_var[k, j]))
                ct += G[k] @ Phi_var @ w.center
                ct += w.radius * soft_l2_norm(G[k] @ Phi_var)
            lhs = ct + G[k] @ np.concatenate((z_var[k], v_var[k])) + g[k]
            for i in range(len(lhs)):
                prog.AddConstraint(lhs[i] <= 0)

        ctN = np.array([Expression(0) for _ in range(nf)])
        for j in range(N):
            ctN += Gf @ Phi_x_var[N, j] @ w.center
            ctN += w.radius * soft_l2_norm(Gf @ Phi_x_var[N, j])
        lhs = ctN + Gf @ z_var[N] + gf
        for i in range(len(lhs)):
            prog.AddConstraint(lhs[i] <= 0)

        result = Solve(prog)
        assert result.is_success()
        z_expected = result.GetSolution(z_var).reshape(z_var.shape)
        v_expected = result.GetSolution(v_var).reshape(v_var.shape)
        Phi_x_expected = self.getSolutionPhi(result, Phi_x_var)
        Phi_u_expected = self.getSolutionPhi(result, Phi_u_var)

        # Compare results
        np.testing.assert_allclose(z, z_expected, atol=1e-10)
        np.testing.assert_allclose(v, v_expected, atol=1e-10)
        np.testing.assert_allclose(Phi_x, Phi_x_expected, atol=1e-8)
        np.testing.assert_allclose(Phi_u, Phi_u_expected, atol=1e-8)

    def test_fast_sls(self):
        self._test_fast_sls(nw=1)
        self._test_fast_sls(nw=2)

        # When time horizon is 10, the hard constraints are satisfiable.
        with self.assertRaises(Exception) as context:
            self._test_fast_sls(nw=2, N=10, slack_penalty=None)
            self.assertEqual(
                str(context.exception),
                "Failed to solve nominal trajectory optimization due to"
                " kInfeasibleConstraints",
            )
        # Adding slack variables and penalize the slack in the objective
        # should allow the optimization problem to be solvable.
        self._test_fast_sls(nw=2, N=10, slack_penalty=1.0)

    def _test_compute_K(self, nw, N=10):
        nx, nu = 2, 1
        Phi_x, Phi_u = self.randomPhi(N, nx, nu, nw)

        K = _Compute_K(Phi_x, Phi_u)
        self.assertEqual(K.shape, (N, N + 1, nu, nx))
        # Check causality
        for k in range(N):
            for j in range(k + 1, N + 1):
                np.testing.assert_allclose(K[k, j], 0.0)

        def flatten(Phi):
            s0, s1, s2, s3 = Phi.shape
            return Phi.transpose(0, 2, 1, 3).reshape(s0 * s2, s1 * s3)

        # K = Φᵤ Φₓ⁻¹ ⇒ K Φₓ = Φᵤ
        np.testing.assert_allclose(flatten(K) @ flatten(Phi_x), flatten(Phi_u))

    def test_compute_K(self):
        self._test_compute_K(nw=1)
        self._test_compute_K(nw=2)
