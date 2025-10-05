import numpy as np
import sympy as sp


class TrajOptProblem:
    def __init__(
        self, system, num_steps, context=None, num_states=None, num_inputs=None
    ):
        assert num_steps > 0
        self.system = system
        self.context = (
            context if context is not None else system.CreateDefaultContext()
        )
        self.system_ad = self.system.ToAutoDiffXd()
        self.context_ad = self.system_ad.CreateDefaultContext()
        self.context_ad.SetTimeStateAndParametersFrom(self.context)

        self.N = num_steps
        self.nx = (
            num_states
            if num_states is not None
            else self.context.get_discrete_state_vector().size()
        )
        self.nu = (
            num_inputs
            if num_inputs is not None
            else system.get_input_port().size()
        )

        x = sp.IndexedBase("x")
        self._xs = np.array(
            [[x[i, j] for j in range(self.nx)] for i in range(self.N + 1)]
        )
        u = sp.IndexedBase("u")
        self._us = np.array(
            [[u[i, j] for j in range(self.nu)] for i in range(self.N)]
        )

        generic_x = sp.IndexedBase("generic_x")
        self._generic_x = np.array([generic_x[j] for j in range(self.nx)])
        generic_u = sp.IndexedBase("generic_u")
        self._generic_u = np.array([generic_u[j] for j in range(self.nu)])

    def state(self, time_index=None):
        if time_index is None:
            return self._generic_x
        else:
            assert time_index < self.N
            return self._xs[time_index]

    def input(self, time_index=None):
        if time_index is None:
            return self._generic_u
        else:
            assert time_index < self.N
            return self._us[time_index]

    def SetInitialState(self, x0):
        assert len(x0) == self.nx
        self._initial_state = np.array(x0)

    def GetInitialState(self):
        return self._initial_state

    def AddStageCost(self, cost):
        if hasattr(self, "_stage_cost"):
            cost = self._stage_cost + cost
        L = sp.Matrix([cost])
        L_x = L.jacobian(self._generic_x)
        L_u = L.jacobian(self._generic_u)
        L_xx = L_x.jacobian(self._generic_x)
        L_xu = L_x.jacobian(self._generic_u)
        L_uu = L_u.jacobian(self._generic_u)

        exprs = [
            cost,
            np.array(L_x).reshape(-1),
            np.array(L_u).reshape(-1),
            np.array(L_xx),
            np.array(L_xu),
            np.array(L_uu),
        ]
        funcs = [None] * len(exprs)
        for i in range(len(funcs)):
            funcs[i] = sp.lambdify([self._generic_x, self._generic_u], exprs[i])

        self._stage_cost = cost
        self._stage_cost_func = funcs[0]
        self._stage_cost_deriv_funcs = funcs[1:]

    def CalcStageCost(self, x, u):
        if not hasattr(self, "_stage_cost"):
            return 0.0
        return self._stage_cost_func(x, u)

    def CalcStageCostDerivatives(self, x, u):
        if not hasattr(self, "_stage_cost"):
            return (0.0 for i in range(5))
        return (func(x, u) for func in self._stage_cost_deriv_funcs)

    def AddTerminalCost(self, cost):
        if hasattr(self, "_terminal_cost"):
            cost = self._terminal_cost + cost
        Lf = sp.Matrix([cost])
        Lf_x = Lf.jacobian(self._generic_x)
        Lf_xx = Lf_x.jacobian(self._generic_x)

        exprs = [
            cost,
            np.array(Lf_x).reshape(-1),
            np.array(Lf_xx),
        ]
        funcs = [None] * len(exprs)
        for i in range(len(funcs)):
            funcs[i] = sp.lambdify([self._generic_x], exprs[i])

        self._terminal_cost = cost
        self._terminal_cost_func = funcs[0]
        self._terminal_cost_deriv_funcs = funcs[1:]

    def CalcTerminalCost(self, x):
        if not hasattr(self, "_terminal_cost"):
            return 0.0
        return self._terminal_cost_func(x)

    def CalcTerminalCostDerivatives(self, x):
        if not hasattr(self, "_terminal_cost"):
            return (0.0 for i in range(2))
        return (func(x) for func in self._terminal_cost_deriv_funcs)

    def AddCost(self, cost):
        if not hasattr(self, "_generic_cost_funcs"):
            self._generic_cost_funcs = []
        cost_func = sp.lambdify([self._xs, self._us], cost)
        self._generic_cost_funcs.append(cost_func)

    def PopCost(self):
        del self._generic_cost_funcs[-1]

    def has_only_stage_and_terminal_cost(self):
        return (
            not hasattr(self, "_generic_cost_funcs")
            or not self._generic_cost_funcs
        )

    def CalcTotalCost(self, xs, us):
        N = self.N
        assert xs.shape == (N + 1, self.nx)
        assert us.shape == (N, self.nu)
        cost = 0.0
        # Stage cost.
        for i in range(N):
            cost += self.CalcStageCost(xs[i], us[i])
        # Terminal cost.
        cost += self.CalcTerminalCost(xs[N])
        # Generic cost.
        if hasattr(self, "_generic_cost_funcs"):
            for cost_func in self._generic_cost_funcs:
                cost += cost_func(xs, us)
        return cost

    def SetUncertaintyTubeCost(self, Q_bar, R_bar, Qf_bar=None):
        assert Q_bar.shape == (self.nx, self.nx)
        assert R_bar.shape == (self.nu, self.nu)
        Qf_bar = Qf_bar if Qf_bar is not None else Q_bar
        assert Qf_bar.shape == (self.nx, self.nx)
        self._uncertainty_tube_cost = (Q_bar, R_bar, Qf_bar)

    def GetUncertaintyTubeCost(self):
        return self._uncertainty_tube_cost

    def SetLinearCollisionConstraint(self, enable=True, distance_threshold=0.0):
        if enable:
            assert isinstance(distance_threshold, (float, int))
            self._collision_distance_threshold = float(distance_threshold)
        else:
            if hasattr(self, "_collision_distance_threshold"):
                del self._collision_distance_threshold

    def _CalcLinearCollisionConstraint(self, x):
        system, context = self.system, self.context
        system.SetState(context, x)
        J = system.EvalCollisionJacobian(context)
        phi = system.EvalCollisionSignedDistance(context)
        # J δx + ϕ ≥ 0  →  −J (x + δx) + J x − ϕ ≤ 0
        G = -J
        g = J @ x - phi
        return G, g

    def AddStageLinearConstraint(self, G, g):
        """
        Adds stage linear constraints: Gₖ [xₖᵀ uₖᵀ]ᵀ + gₖ ≤ 0 for all k=0,⋯,N-1
        """
        if np.ndim(G) == 2:
            G = np.repeat(G[np.newaxis, :, :], repeats=self.N, axis=0)
            g = np.repeat(g[np.newaxis, :], repeats=self.N, axis=0)
        nc = G.shape[1]
        assert G.shape == (self.N, nc, self.nx + self.nu)
        assert g.shape == (self.N, nc)
        if hasattr(self, "_stage_linear_constraint"):
            constraint = self._stage_linear_constraint
            G = np.concatenate((constraint[0], G), axis=1)
            g = np.concatenate((constraint[1], g), axis=1)
        self._stage_linear_constraint = (G, g)

    def GetStageLinearConstraint(self, xs=None):
        N, nx, nu = self.N, self.nx, self.nu

        if hasattr(self, "_stage_linear_constraint"):
            constraint = self._stage_linear_constraint
        else:
            constraint = np.zeros((N, 0, nx + nu)), np.zeros((N, 0))

        if not hasattr(self, "_collision_distance_threshold"):
            return constraint
        else:
            if xs is None:
                raise RuntimeError(
                    "Since collision constraints are enabled, xs must be"
                    " passed to GetStageLinearConstraint(xs)."
                )
            assert xs.shape == (N + 1, nx)
            assert np.issubdtype(xs.dtype, np.floating)
            for k in range(N):
                G_k, g_k = self._CalcLinearCollisionConstraint(xs[k])
                g_k += self._collision_distance_threshold
                nc = G_k.shape[0]
                G_k = np.hstack((G_k, np.zeros((nc, nu))))
                if k == 0:
                    G = np.zeros((N, nc, nx + nu))
                    g = np.zeros((N, nc))
                G[k], g[k] = G_k, g_k
            return (
                np.concatenate((constraint[0], G), axis=1),
                np.concatenate((constraint[1], g), axis=1),
            )

    def AddTerminalLinearConstraint(self, Gf, gf):
        """
        Adds terminal linear constraint: Gf x[N] + gf ≤ 0
        """
        nf = Gf.shape[0]
        assert Gf.shape == (nf, self.nx)
        assert gf.shape == (nf,)
        if hasattr(self, "_terminal_linear_constraint"):
            constraint = self._terminal_linear_constraint
            Gf = np.concatenate((constraint[0], Gf), axis=0)
            gf = np.concatenate((constraint[1], gf), axis=0)
        self._terminal_linear_constraint = (Gf, gf)

    def GetTerminalLinearConstraint(self, xs=None):
        N, nx = self.N, self.nx

        if hasattr(self, "_terminal_linear_constraint"):
            constraint = self._terminal_linear_constraint
        else:
            constraint = np.zeros((0, nx)), np.zeros((0,))

        if not hasattr(self, "_collision_distance_threshold"):
            return constraint
        else:
            if xs is None:
                raise RuntimeError(
                    "Since collision constraints are enabled, xs must be"
                    " passed to GetTerminalLinearConstraint(xs)."
                )
            assert xs.shape == (N + 1, nx)
            assert np.issubdtype(xs.dtype, np.floating)
            G_N, g_N = self._CalcLinearCollisionConstraint(xs[-1])
            g_N += self._collision_distance_threshold
            return (
                np.concatenate((constraint[0], G_N), axis=0),
                np.concatenate((constraint[1], g_N), axis=0),
            )

    def AddConstraint(self, expr, lb, ub):
        def as_1d_array(a):
            return np.array([a]) if np.ndim(a) == 0 else np.asarray(a)

        expr = as_1d_array(expr)
        lb = as_1d_array(lb)
        ub = as_1d_array(ub)
        assert expr.shape == lb.shape == ub.shape

        func = sp.lambdify([self._xs, self._us], expr)
        if not hasattr(self, "_generic_constraints"):
            self._generic_constraints = []
        self._generic_constraints.append((func, lb, ub))

        def eval_func(xs, us, func=func, lb=lb, ub=ub):
            expr = func(xs, us)
            return np.bitwise_and(lb <= expr, expr <= ub)

        return eval_func

    def PopConstraint(self):
        del self._generic_constraints[-1]

    def ApplyConstraintToMathematicalProgram(
        self,
        prog,
        xs,
        us,
        delta_xs,
        delta_us,
        include_stage_and_terminal_linear_constraint=True,
    ):
        assert xs.shape == (self.N + 1, self.nx)
        assert us.shape == (self.N, self.nu)
        assert delta_xs.shape == (self.N + 1, self.nx)
        assert delta_us.shape == (self.N, self.nu)
        assert np.issubdtype(xs.dtype, np.floating)
        assert np.issubdtype(us.dtype, np.floating)
        assert np.issubdtype(delta_xs.dtype, np.object_)
        assert np.issubdtype(delta_us.dtype, np.object_)

        if include_stage_and_terminal_linear_constraint:
            G, g = self.GetStageLinearConstraint(xs)
            if G.size != 0:
                for k in range(self.N):
                    prog.AddLinearConstraint(
                        G[k]
                        @ np.concatenate(
                            (xs[k] + delta_xs[k], us[k] + delta_us[k])
                        ),
                        lb=np.full_like(g[k], -np.inf),
                        ub=-g[k],
                    )

            Gf, gf = self.GetTerminalLinearConstraint(xs)
            if Gf.size != 0:
                prog.AddLinearConstraint(
                    Gf @ (xs[-1] + delta_xs[-1]),
                    lb=np.full_like(gf, -np.inf),
                    ub=-gf,
                )

        if hasattr(self, "_generic_constraints"):
            for func, lb, ub in self._generic_constraints:
                expr = func(xs + delta_xs, us + delta_us)
                for i in range(len(lb)):
                    prog.AddConstraint(expr[i], lb[i], ub[i])
