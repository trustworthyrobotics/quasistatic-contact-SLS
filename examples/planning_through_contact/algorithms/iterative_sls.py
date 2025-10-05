import numpy as np

from pydrake.all import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiffTuple,
    LeafSystem_,
    TemplateSystem,
)

try:
    from examples.planning_through_contact.algorithms.fast_sls import (
        FastSLS,
        NormBall,
    )
except ModuleNotFoundError:
    from algorithms.fast_sls import FastSLS, NormBall


def IterativeSLSController(prob, us_init, **kwargs):
    alg = IterativeSLSAlgorithm(prob)
    xs, us, Phi_x, Phi_u, Ks, xs_bounds, us_bounds = alg.Solve(
        us_init=us_init, **kwargs
    )
    dt = prob.system.GetUniquePeriodicDiscreteUpdateAttribute().period_sec()
    controller = Controller(xs=xs, us=us, Ks=Ks, dt=dt)

    controller.Phi_x = Phi_x
    controller.Phi_u = Phi_u
    controller.xs_lb = xs_bounds[0]
    controller.xs_ub = xs_bounds[1]
    controller.us_lb = us_bounds[0]
    controller.us_ub = us_bounds[1]

    return controller


@TemplateSystem.define("Controller_", T_list=[float, AutoDiffXd])
def Controller_(T):
    class Impl(LeafSystem_[T]):
        def _construct(
            self, xs, us, Ks, dt, time_dialation=1.0, converter=None
        ):
            super().__init__(converter)
            N = us.shape[0]
            nx, nu = xs.shape[1], us.shape[1]
            assert xs.shape == (N + 1, nx)
            assert us.shape == (N, nu)
            assert Ks.shape == (N, N + 1, nu, nx)
            self.xs = xs.copy()
            self.us = us.copy()
            self.Ks = Ks.copy()
            self.dt = dt
            self.time_comparison_tolerance = np.finfo(np.float64).eps

            self.DeclareVectorInputPort("x", nx)
            self.DeclareVectorOutputPort("u", nu, self.CalcOutput)

            self.x_hist = np.zeros((N + 1, nx))

            self.time_dialation = time_dialation

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self,
                xs=other.xs,
                us=other.us,
                Ks=other.Ks,
                dt=other.dt,
                time_dialation=other.time_dialation,
                converter=converter,
            )

        def SetTimeDialation(self, time_dialation):
            assert time_dialation >= 1.0
            self.time_dialation = time_dialation

        def CalcOutput(self, context, output):
            eps = self.time_comparison_tolerance
            i = int(
                np.floor(
                    (context.get_time() + eps) / (self.dt * self.time_dialation)
                )
            )
            i = min(i, self.us.shape[0] - 1)
            x = self.get_input_port().Eval(context)
            self.x_hist[i] = x
            u = self.us[i].copy()
            for j in range(i + 1):
                u += self.Ks[i, j] @ (self.x_hist[j] - self.xs[j])
            output.set_value(u)

    return Impl


Controller = Controller_[None]


class IterativeSLSAlgorithm:
    def __init__(self, prob, w=None):
        self.prob = prob
        self.N = prob.N
        self.nx = prob.nx
        self.nu = prob.nu
        self.nw = prob.system.GetDisturbanceMatrix(prob.context).shape[1]

        assert w is None or isinstance(w, NormBall)
        if w is None:
            w = NormBall(
                center=1.5 * np.ones(self.nw),
                radius=0.5 * np.sqrt(self.nw),
                order=2,
            )
        self.w = w

    def _Step(self, x, u, disable_smoothing=False):
        system, context = self.prob.system_ad, self.prob.context_ad
        nx, nu = self.nx, self.nu
        assert len(x) == nx
        assert len(u) == nu

        if not disable_smoothing:
            x_ad, u_ad = InitializeAutoDiffTuple(x, u)
            context.SetDiscreteState(x_ad)
            system.get_input_port().FixValue(context, u_ad)
            context.SetDiscreteState(
                system.EvalUniquePeriodicDiscreteUpdate(context)
            )

            xnext_ad = context.get_discrete_state_vector().CopyToVector()
            xnext = np.squeeze(ExtractValue(xnext_ad))
            dxnext_dxu = ExtractGradient(xnext_ad)

            dxnext_dx = dxnext_dxu[:, 0:nx]
            dxnext_du = dxnext_dxu[:, -nu:]

            E_ad = system.GetDisturbanceMatrix(context)
            E = ExtractValue(E_ad)
            dE_dxu = ExtractGradient(E_ad).reshape((nx, -1, nx + nu), order="A")
            dE_dx = dE_dxu[:, :, 0:nx]
            dE_du = dE_dxu[:, :, -nu:]

            return xnext, dxnext_dx, dxnext_du, E, dE_dx, dE_du
        else:
            kappa = system.get_dynamics_smoothing(context)
            system.set_dynamics_smoothing(context, 0.0)

            context.SetDiscreteState(x)
            system.get_input_port().FixValue(context, u)
            context.SetDiscreteState(
                system.EvalUniquePeriodicDiscreteUpdate(context)
            )
            xnext = context.get_discrete_state_vector().CopyToVector()
            xnext = np.squeeze(ExtractValue(xnext))

            system.set_dynamics_smoothing(context, kappa)

            return xnext

    def _Rollout(self, x0, us):
        N = us.shape[0]
        nx, nu, nw = self.nx, self.nu, self.nw
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        xs = np.empty((N + 1, nx))
        xs[0] = x0
        As = np.empty((N, nx, nx))
        Bs = np.empty((N, nx, nu))
        Es = np.empty((N, nx, nw))
        dEs_dx = np.empty((N, nx, nw, nx))
        dEs_du = np.empty((N, nx, nw, nu))

        for i in range(N):
            xs[i + 1], As[i], Bs[i], Es[i], dEs_dx[i], dEs_du[i] = self._Step(
                xs[i], us[i]
            )

        cost = self.prob.CalcTotalCost(xs, us)

        return xs, As, Bs, Es, dEs_dx, dEs_du, cost

    def _ForwardPass(self, xs, us, delta_xs, delta_us, Ks, feedback=False):
        N = us.shape[0]
        nx, nu, nw = self.nx, self.nu, self.nw
        assert xs.shape == (N + 1, nx)
        assert us.shape == (N, nu)
        assert delta_xs.shape == (N + 1, nx)
        assert delta_us.shape == (N, nu)
        assert Ks.shape == (N, N + 1, nu, nx)

        xs_true = np.empty_like(xs)
        xs_true[0] = xs[0]

        xs_new = np.empty_like(xs)
        xs_new[0] = xs[0]
        us_new = us + delta_us
        As = np.empty((N, nx, nx))
        Bs = np.empty((N, nx, nu))
        Es = np.empty((N, nx, nw))
        dEs_dx = np.empty((N, nx, nw, nx))
        dEs_du = np.empty((N, nx, nw, nu))

        for i in range(N):
            if feedback:
                for j in range(i + 1):
                    us_new[i] += Ks[i, j] @ (xs_true[j] - xs_new[j])
            xs_new[i + 1], As[i], Bs[i], Es[i], dEs_dx[i], dEs_du[i] = (
                self._Step(xs_new[i], us_new[i])
            )
            xs_true[i + 1] = self._Step(
                xs_true[i], us_new[i], disable_smoothing=True
            )

        cost_new = self.prob.CalcTotalCost(xs_new, us_new)

        return xs_new, us_new, As, Bs, Es, dEs_dx, dEs_du, cost_new

    def _BackwardPass(
        self,
        xs,
        us,
        As,
        Bs,
        Es,
        dEs_dx,
        dEs_du,
        w,
        delta_cap,
        max_iters,
        tolerance,
        slack_penalty,
    ):
        N, nx, nu, nw = self.N, self.nx, self.nu, self.nw
        assert xs.shape == (N + 1, nx)
        assert us.shape == (N, nu)
        assert As.shape == (N, nx, nx)
        assert Bs.shape == (N, nx, nu)
        assert Es.shape == (N, nx, nw)
        assert dEs_dx.shape == (N, nx, nw, nx)
        assert dEs_du.shape == (N, nx, nw, nu)
        assert isinstance(w, NormBall)
        assert delta_cap > 0.0
        assert max_iters > 0
        assert tolerance > 0.0

        Q_bar, R_bar, Qf_bar = self.prob.GetUncertaintyTubeCost()

        G, g = self.prob.GetStageLinearConstraint(xs=xs)
        for k in range(N):
            g[k] += G[k] @ np.concatenate((xs[k], us[k]))

        Gf, gf = self.prob.GetTerminalLinearConstraint(xs=xs)
        gf += Gf @ xs[N]

        def cost_func(delta_xs, delta_us):
            return self.prob.CalcTotalCost(xs + delta_xs, us + delta_us)

        def Es_func(delta_xs, delta_us):
            return (
                Es
                + (dEs_dx * delta_xs[:N, None, None, :]).sum(axis=-1)
                + (dEs_du * delta_us[:, None, None, :]).sum(axis=-1)
            )

        def additional_constraints_func(prog, delta_xs, delta_us):
            self.prob.ApplyConstraintToMathematicalProgram(
                prog=prog,
                xs=xs,
                us=us,
                delta_xs=delta_xs,
                delta_us=delta_us,
                include_stage_and_terminal_linear_constraint=False,
            )
            prog.AddBoundingBoxConstraint(
                -delta_cap,
                delta_cap,
                np.concatenate((delta_xs.reshape(-1), delta_us.reshape(-1))),
            )

        delta_xs, delta_us, Phi_x, Phi_u, K = FastSLS(
            A=As,
            B=Bs,
            E=Es_func,
            x0=np.zeros(nx),
            cost_func=cost_func,
            Q_bar=Q_bar,
            R_bar=R_bar,
            Qf_bar=Qf_bar,
            G=G,
            g=g,
            Gf=Gf,
            gf=gf,
            w=w,
            additional_constraints_func=additional_constraints_func,
            max_iters=max_iters,
            tolerance=tolerance,
            slack_penalty=slack_penalty,
        )

        return delta_xs, delta_us, Phi_x, Phi_u, K

    def Solve(
        self,
        us_init,
        delta_cap=1.0,
        slack_penalty=None,
        max_inner_iters=10,
        max_outer_iters=50,
        tolerance=1e-6,
    ):
        assert delta_cap > 0.0
        assert slack_penalty is None or slack_penalty > 0.0
        assert max_inner_iters > 0
        assert max_outer_iters > 0
        assert tolerance > 0.0

        x0 = np.array(self.prob.GetInitialState())
        us = np.array(us_init)

        N, nx, nu = self.N, self.nx, self.nu
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        w = self.w

        xs, As, Bs, Es, dEs_dx, dEs_du, cost_old = self._Rollout(x0, us)

        for iter in range(1, max_outer_iters + 1):
            delta_xs, delta_us, Phi_x, Phi_u, Ks = self._BackwardPass(
                xs=xs,
                us=us,
                As=As,
                Bs=Bs,
                Es=Es,
                dEs_dx=dEs_dx,
                dEs_du=dEs_du,
                w=w,
                delta_cap=delta_cap,
                max_iters=max_inner_iters,
                tolerance=tolerance,
                slack_penalty=slack_penalty,
            )

            xs, us, As, Bs, Es, dEs_dx, dEs_du, cost_new = self._ForwardPass(
                xs=xs,
                us=us,
                delta_xs=delta_xs,
                delta_us=delta_us,
                Ks=Ks,
                feedback=(iter == max_outer_iters),
            )

            if np.abs(cost_old - cost_new) < tolerance:
                break
            cost_old = cost_new

        xs_lb, xs_ub = xs.copy(), xs.copy()
        for k in range(xs.shape[0]):
            for j in range(k):
                Phi = Phi_x[k, j]
                Phi_norm = np.linalg.norm(Phi, ord=w.dual_order, axis=1)
                xs_lb[k] += Phi @ w.center + w.radius * Phi_norm
                xs_ub[k] += Phi @ w.center - w.radius * Phi_norm
        us_lb, us_ub = us.copy(), us.copy()
        for k in range(us.shape[0]):
            for j in range(k):
                Phi = Phi_u[k, j]
                Phi_norm = np.linalg.norm(Phi, ord=w.dual_order, axis=1)
                us_lb[k] += Phi @ w.center + w.radius * Phi_norm
                us_ub[k] += Phi @ w.center - w.radius * Phi_norm

        return xs, us, Phi_x, Phi_u, Ks, (xs_lb, xs_ub), (us_lb, us_ub)
