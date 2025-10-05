import unittest

import numpy as np

from examples.planning_through_contact.algorithms import (
    IterativeTO,
    TrajOptProblem,
)
from pydrake.all import (
    FiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    LinearSystem,
    PiecewisePolynomial,
)


class TestIterativeTO(unittest.TestCase):
    def setUp(self):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        self.system = LinearSystem(A=A, B=B, time_period=dt)

    def test_iterative_to(self):
        # Iterative trajectory optimization
        system = self.system
        N = 100
        prob = TrajOptProblem(system=system, num_steps=N)
        x = prob.state()
        u = prob.input()

        x_goal = np.array([1.0, 0.0])
        Q = np.diag([1, 1])
        R = np.diag([1])
        prob.AddStageCost((x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u)
        prob.AddTerminalCost((x - x_goal).T @ Q @ (x - x_goal))

        x0 = [0.0, 0.0]
        prob.SetInitialState(x0)

        us_init = np.ones((N, 1)) * 0.0
        traj = IterativeTO(
            prob,
            us_init=us_init,
            trust_region_covar=None,
            use_contact_trust_region=False,
        )
        xs, us = traj.xs, traj.us

        nx, nu = len(x), len(u)
        self.assertEqual(xs.shape, (N + 1, nx))
        self.assertEqual(us.shape, (N, nu))

        # Finite horizon LQR
        dt = system.time_period()
        context = system.CreateDefaultContext()
        u0 = np.array([0.0])
        system.get_input_port().FixValue(context, u0)
        context.SetDiscreteState(x_goal)
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.Qf = Q
        options.xd = PiecewisePolynomial(x_goal)
        lqr = FiniteHorizonLinearQuadraticRegulator(
            system, context, t0=0, tf=dt * N, Q=Q, R=R, options=options
        )

        xs_expected = np.zeros((N + 1, nx))
        us_expected = np.zeros((N, nu))
        xs_expected[0] = x0
        for k in range(N):
            t = k * dt
            us_expected[k] = (
                u0
                - lqr.K.value(t) @ (xs_expected[k] - x_goal)
                - lqr.k0.value(t)
            )
            xs_expected[k + 1] = (
                system.A() @ xs_expected[k] + system.B() @ us_expected[k]
            )

        np.testing.assert_allclose(xs, xs_expected)
        np.testing.assert_allclose(us, us_expected)
