import copy
import unittest

import numpy as np

from examples.planning_through_contact.algorithms import (
    IterativeSLSController,
    TrajOptProblem,
)
from pydrake.all import QuasidynamicLinearPusher


class TestIterativeSLS(unittest.TestCase):
    def test_iterative_sls_controller(self):
        system = QuasidynamicLinearPusher(k_a=1.0, m_o=0.01, w=0.1, dt=0.02)
        context = system.CreateDefaultContext()
        system.set_dynamics_smoothing(context, 1e-3)

        num_steps = 50
        prob = TrajOptProblem(
            system=system,
            context=context,
            num_steps=num_steps,
        )
        x = prob.state()
        u = prob.input()

        prob.AddStageCost((x[1] - 0.5) ** 2 + 0.1 * u[0] ** 2)
        prob.AddTerminalCost((x[1] - 0.5) ** 2)

        x0 = [0.0, 0.2]
        prob.SetInitialState(x0)

        for k in range(num_steps - 1):
            prob.AddCost((prob.input(k) - prob.input(k + 1))[0] ** 2)

        prob.AddConstraint(prob.input(0), lb=[0.0], ub=[0.0])

        Q_bar = np.identity(2) * 1e2
        R_bar = np.identity(1) * 1e2
        Qf_bar = np.identity(2) * 1e3
        prob.SetUncertaintyTubeCost(Q_bar=Q_bar, R_bar=R_bar, Qf_bar=Qf_bar)

        us_init = np.ones((num_steps, 1)) * 0.0
        controller = IterativeSLSController(
            prob,
            us_init=us_init,
            delta_cap=0.05,
            max_outer_iters=18,
        )

        # Check copying
        controller.Clone()
        copy.copy(controller)
        copy.deepcopy(controller)
