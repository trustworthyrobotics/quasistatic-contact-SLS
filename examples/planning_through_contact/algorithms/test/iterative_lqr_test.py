import copy
import unittest

import numpy as np

from examples.planning_through_contact.algorithms import (
    IterativeLQRController,
    TrajOptProblem,
)
from pydrake.all import (
    FiniteHorizonLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    LinearSystem,
)


class TestIterativeLQR(unittest.TestCase):
    def setUp(self):
        # Double integrator
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[dt**2 / 2], [dt]])
        self.system = LinearSystem(A=A, B=B, time_period=dt)

    def test_iterative_lqr(self):
        # Iterative LQR
        system = self.system
        num_steps = 100
        prob = TrajOptProblem(system=system, num_steps=num_steps)
        x = prob.state()
        u = prob.input()

        x_goal = np.array([1.0, 0.0])
        Q = np.diag([1, 1])
        R = np.diag([1])
        prob.AddStageCost((x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u)
        prob.AddTerminalCost((x - x_goal).T @ Q @ (x - x_goal))

        x0 = [0.0, 0.0]
        prob.SetInitialState(x0)

        us_init = np.ones((num_steps, 1)) * 0.0
        ilqr = IterativeLQRController(
            prob, us_init=us_init, regu_init=1e-16, min_regu=1e-16
        )

        self.assertEqual(ilqr.xs.shape, (num_steps + 1, 2))
        self.assertEqual(ilqr.us.shape, (num_steps, 1))
        self.assertEqual(ilqr.ks.shape, (num_steps, 1))
        self.assertEqual(ilqr.Ks.shape, (num_steps, 1, 2))

        # Finite horizon LQR
        dt = system.time_period()
        context = system.CreateDefaultContext()
        system.get_input_port().FixValue(context, [0.0])
        context.SetDiscreteState(x_goal)
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.Qf = Q
        lqr = FiniteHorizonLinearQuadraticRegulator(
            system, context, t0=0, tf=dt * num_steps, Q=Q, R=R, options=options
        )

        # Compare iterative LQR with finite horizon LQR
        for i in range(num_steps):
            np.testing.assert_allclose(
                ilqr.ks[i], np.squeeze(-lqr.k0.value(i * dt)), atol=1e-14
            )
            np.testing.assert_allclose(
                ilqr.Ks[i], -lqr.K.value(i * dt), atol=1e-14
            )

    def test_iterative_lqr_controller(self):
        # Iterative LQR
        system = self.system
        num_steps = 100
        prob = TrajOptProblem(system=system, num_steps=num_steps)
        x = prob.state()
        u = prob.input()

        x_goal = np.array([1.0, 0.0])
        Q = np.diag([1, 1])
        R = np.diag([1])
        prob.AddStageCost((x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u)
        prob.AddTerminalCost((x - x_goal).T @ Q @ (x - x_goal))

        x0 = [0.0, 0.0]
        prob.SetInitialState(x0)

        us_init = np.ones((num_steps, 1)) * 0.0
        controller = IterativeLQRController(
            prob, us_init=us_init, regu_init=1e-16, min_regu=1e-16
        )

        xs = controller.xs.copy()
        us = controller.us.copy()
        ks = controller.ks.copy()
        Ks = controller.Ks.copy()

        # Check controller outputs
        dt = system.time_period()
        context = controller.CreateDefaultContext()
        for i in range(num_steps):
            x = np.random.rand(2)
            controller.get_input_port().FixValue(context, x)
            context.SetTime(i * dt)
            u = controller.get_output_port().Eval(context)
            u_expected = us[i] + ks[i] + Ks[i] @ (x - xs[i])
            np.testing.assert_allclose(u, u_expected)

        # Check copying
        controller.Clone()
        copy.copy(controller)
        copy.deepcopy(controller)
