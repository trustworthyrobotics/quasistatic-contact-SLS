import unittest

import numpy as np

from examples.planning_through_contact.algorithms import TrajOptProblem
from pydrake.all import LinearSystem


class TestCost(unittest.TestCase):
    def setUp(self):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        self.system = LinearSystem(A=A, B=B, time_period=dt)

    def test_initial_state(self):
        prob = TrajOptProblem(system=self.system, num_steps=100)

        x0 = np.array([1.0, 2.0])
        prob.SetInitialState(x0=x0)

        np.testing.assert_allclose(prob.GetInitialState(), x0)

    def test_quadratic_stage_cost(self):
        num_steps = 100
        prob = TrajOptProblem(system=self.system, num_steps=num_steps)
        x = prob.state()
        u = prob.input()

        Q = np.diag([1.0, 2.0])
        q = np.array([3.0, 4.0])
        R = np.diag([5.0])
        r = np.array([6.0])
        N = np.array([[7.0], [8.0]])
        cost = (
            0.5 * x.T @ Q @ x
            + q.dot(x)
            + 0.5 * u.T @ R @ u
            + r.dot(u)
            + x.T @ N @ u
        )
        prob.AddStageCost(cost=cost)
        self.assertTrue(prob.has_only_stage_and_terminal_cost())

        x = np.array([0.1, 0.2])
        u = np.array([0.3])
        L = prob.CalcStageCost(x=x, u=u)
        L_expected = (
            0.5 * x.T @ Q @ x
            + q.dot(x)
            + 0.5 * u.T @ R @ u
            + r.dot(u)
            + x.T @ N @ u
        )
        np.testing.assert_allclose(L, L_expected)

        L_x, L_u, L_xx, L_xu, L_uu = prob.CalcStageCostDerivatives(x=x, u=u)
        np.testing.assert_allclose(L_x, Q @ x + q + N @ u)
        np.testing.assert_allclose(L_u, R @ u + r + N.T @ x)
        np.testing.assert_allclose(L_xx, Q, atol=1e-16)
        np.testing.assert_allclose(L_xu, N, atol=1e-16)
        np.testing.assert_allclose(L_uu, R, atol=1e-16)

        xs = np.repeat(x[np.newaxis, :], repeats=num_steps + 1, axis=0)
        us = np.repeat(u[np.newaxis, :], repeats=num_steps, axis=0)
        np.testing.assert_allclose(
            prob.CalcTotalCost(xs=xs, us=us), L_expected * num_steps
        )

    def test_quadratic_terminal_cost(self):
        num_steps = 100
        prob = TrajOptProblem(system=self.system, num_steps=num_steps)
        x = prob.state()

        Q = np.diag([1.0, 2.0])
        q = np.array([3.0, 4.0])
        cost = 0.5 * x.T @ Q @ x + q.dot(x)
        prob.AddTerminalCost(cost=cost)
        self.assertTrue(prob.has_only_stage_and_terminal_cost())

        x = np.array([0.1, 0.2])
        Lf = prob.CalcTerminalCost(x=x)
        Lf_expected = 0.5 * x.T @ Q @ x + q.dot(x)
        np.testing.assert_allclose(Lf, Lf_expected)

        Lf_x, Lf_xx = prob.CalcTerminalCostDerivatives(x=x)
        np.testing.assert_allclose(Lf_x, Q @ x + q)
        np.testing.assert_allclose(Lf_xx, Q, atol=1e-16)

        xs = np.repeat(x[np.newaxis, :], repeats=num_steps + 1, axis=0)
        u = np.array([0.0])
        us = np.repeat(u[np.newaxis, :], repeats=num_steps, axis=0)
        np.testing.assert_allclose(
            prob.CalcTotalCost(xs=xs, us=us), Lf_expected
        )

    def test_generic_cost(self):
        num_steps = 100
        prob = TrajOptProblem(system=self.system, num_steps=num_steps)

        R = np.diag([1.0])
        for k in range(num_steps - 1):
            diff = prob.input(k + 1) - prob.input(k)
            prob.AddCost(diff.dot(R @ diff))
        self.assertFalse(prob.has_only_stage_and_terminal_cost())

        xs = np.zeros((num_steps + 1, 2))
        us = np.arange(num_steps).reshape(-1, 1)
        np.testing.assert_allclose(
            prob.CalcTotalCost(xs, us),
            num_steps - 1,
        )

    def test_uncertainty_tube_cost(self):
        prob = TrajOptProblem(system=self.system, num_steps=100)

        Q_bar = np.identity(2)
        R_bar = np.identity(1)
        Qf_bar = np.identity(2) * 10
        prob.SetUncertaintyTubeCost(Q_bar=Q_bar, R_bar=R_bar, Qf_bar=Qf_bar)

        np.testing.assert_equal(prob.GetUncertaintyTubeCost()[0], Q_bar)
        np.testing.assert_equal(prob.GetUncertaintyTubeCost()[1], R_bar)
        np.testing.assert_equal(prob.GetUncertaintyTubeCost()[2], Qf_bar)

    def test_stage_linear_constraint(self):
        N = 100
        prob = TrajOptProblem(system=self.system, num_steps=N)
        constraint = prob.GetStageLinearConstraint()
        self.assertTupleEqual(constraint[0].shape, (N, 0, 3))
        self.assertTupleEqual(constraint[1].shape, (N, 0))

        G = np.array([[1, 0, 0], [0, 1, 0]])
        g = np.array([1, 1])
        prob.AddStageLinearConstraint(
            G=np.expand_dims(G[0], axis=0),
            g=np.expand_dims(g[0], axis=0),
        )
        prob.AddStageLinearConstraint(
            G=np.expand_dims(G[1], axis=0),
            g=np.expand_dims(g[1], axis=0),
        )

        np.testing.assert_equal(
            prob.GetStageLinearConstraint()[0],
            np.repeat(G[np.newaxis, :, :], repeats=N, axis=0),
        )
        np.testing.assert_equal(
            prob.GetStageLinearConstraint()[1],
            np.repeat(g[np.newaxis, :], repeats=N, axis=0),
        )

    def test_terminal_linear_constraint(self):
        prob = TrajOptProblem(system=self.system, num_steps=100)
        self.assertTupleEqual(
            prob.GetTerminalLinearConstraint()[0].shape, (0, 2)
        )
        self.assertTupleEqual(prob.GetTerminalLinearConstraint()[1].shape, (0,))

        Gf = np.array([[1, 0], [0, 1]])
        gf = np.array([1, 1])
        prob.AddTerminalLinearConstraint(
            Gf=np.expand_dims(Gf[0], axis=0),
            gf=np.expand_dims(gf[0], axis=0),
        )
        prob.AddTerminalLinearConstraint(
            Gf=np.expand_dims(Gf[1], axis=0),
            gf=np.expand_dims(gf[1], axis=0),
        )

        np.testing.assert_equal(prob.GetTerminalLinearConstraint()[0], Gf)
        np.testing.assert_equal(prob.GetTerminalLinearConstraint()[1], gf)

    def test_generic_constraint(self):
        num_steps = 100
        prob = TrajOptProblem(system=self.system, num_steps=num_steps)

        u0 = prob.input(0)
        equality_constraint = prob.AddConstraint(expr=u0, lb=[0], ub=[0])

        xs = np.zeros((num_steps + 1, 2))
        us = np.zeros((num_steps, 1))
        self.assertTrue(equality_constraint(xs, us)[0])
        us = np.ones((num_steps, 1))
        self.assertFalse(equality_constraint(xs, us)[0])

        u1 = prob.input(1)
        inequality_constraint = prob.AddConstraint(
            abs(u0[0] - u1[0]), lb=0, ub=1
        )
        self.assertTrue(inequality_constraint(xs, us)[0])
        us[1] = np.array([3])
        self.assertFalse(inequality_constraint(xs, us)[0])
