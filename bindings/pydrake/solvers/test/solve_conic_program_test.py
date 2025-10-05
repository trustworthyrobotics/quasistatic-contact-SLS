import unittest

import numpy as np

from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiffTuple
from pydrake.common.test_utilities import numpy_compare
from pydrake.solvers import (
    NonnegativeCone,
    SecondOrderCone,
    SolveConicProgram,
)


class TestSolveConicProgram(unittest.TestCase):
    @numpy_compare.check_nonsymbolic_types
    def test_solve_conic_program(self, T):
        P = np.array([[3, 1, -1], [1, 4, 2], [-1, 2, 5]])
        q = np.array([1, 2, -3])
        A = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        b = np.array([2, 2, 0, 0, 0])
        if T == AutoDiffXd:
            P, q, A, b = InitializeAutoDiffTuple(P, q, A, b)
            q = q.reshape(-1)
            b = b.reshape(-1)
        cones = [NonnegativeCone(2), SecondOrderCone(3)]

        kwargs = {
            "P": P,
            "q": q,
            "A": A,
            "b": b,
            "cones": cones,
            "mu_target": 0,
        }
        x, z = SolveConicProgram(**kwargs, diff_wrt_mu=False)
        x, z, dx_dmu, dz_dmu = SolveConicProgram(**kwargs, diff_wrt_mu=True)

        self.assertTupleEqual(x.shape, dx_dmu.shape)
        self.assertTupleEqual(z.shape, dz_dmu.shape)

        # Check stationarity
        stationarity = P.dot(x) + q + A.T.dot(z)
        numpy_compare.assert_float_allclose(
            stationarity, np.zeros(len(x)), atol=1e-9
        )

        # Check primal feasibility
        s = b - A.dot(x)
        s_pos = s[:2]
        s_soc = s[2:]
        self.assertTrue(np.all(s_pos > 0))
        self.assertTrue(s_soc[0] > np.linalg.norm(s_soc[1:]))

        # Check dual feasibility
        z_pos = z[:2]
        z_soc = z[2:]
        self.assertTrue(np.all(z_pos > 0))
        self.assertTrue(z_soc[0] > np.linalg.norm(z_soc[1:]))

        # Check complementarity
        numpy_compare.assert_float_allclose(s.dot(z) / 4, 0.0, atol=1e-9)
