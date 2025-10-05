import numpy as np

from pydrake.all import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiffTuple,
    LeafSystem_,
    TemplateSystem,
)


def IterativeLQRController(prob, us_init, **kwargs):
    alg = IterativeLQRAlgorithm(prob)
    xs, us, ks, Ks = alg.Solve(us_init=us_init, **kwargs)
    dt = prob.system.GetUniquePeriodicDiscreteUpdateAttribute().period_sec()
    return Controller(xs=xs, us=us, ks=ks, Ks=Ks, dt=dt)


@TemplateSystem.define("Controller_", T_list=[float, AutoDiffXd])
def Controller_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, xs, us, ks, Ks, dt, converter=None):
            super().__init__(converter)
            N = us.shape[0]
            nx, nu = xs.shape[1], us.shape[1]
            assert xs.shape == (N + 1, nx)
            assert us.shape == (N, nu)
            assert ks.shape == (N, nu)
            assert Ks.shape == (N, nu, nx)
            self.xs = xs
            self.us = us
            self.ks = ks
            self.Ks = Ks
            self.dt = dt
            self.time_comparison_tolerance = np.finfo(np.float64).eps

            self.DeclareVectorInputPort("x", nx)
            self.DeclareVectorOutputPort("u", nu, self.CalcOutput)

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self,
                xs=other.xs,
                us=other.us,
                ks=other.ks,
                Ks=other.Ks,
                dt=other.dt,
                converter=converter,
            )

        def CalcOutput(self, context, output):
            eps = self.time_comparison_tolerance
            i = int(np.floor((context.get_time() + eps) / self.dt))
            i = min(i, self.us.shape[0] - 1)
            x = self.get_input_port().Eval(context)
            u = self.us[i] + self.ks[i] + self.Ks[i] @ (x - self.xs[i])
            output.set_value(u)

    return Impl


Controller = Controller_[None]


class IterativeLQRAlgorithm:
    def __init__(self, prob):
        self.prob = prob
        self.N = prob.N
        self.nx = prob.nx
        self.nu = prob.nu

    def _Step(self, x, u):
        system, context = self.prob.system_ad, self.prob.context_ad
        nx, nu = self.nx, self.nu
        assert len(x) == nx
        assert len(u) == nu
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
        return xnext, dxnext_dx, dxnext_du

    def _Rollout(self, x0, us):
        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        xs = np.empty((N + 1, nx))
        xs[0] = x0
        As = np.empty((N, nx, nx))
        Bs = np.empty((N, nx, nu))
        cost = 0

        for i in range(N):
            xs[i + 1], As[i], Bs[i] = self._Step(xs[i], us[i])
            cost += self.prob.CalcStageCost(xs[i], us[i])
        cost += self.prob.CalcTerminalCost(xs[-1])

        return xs, As, Bs, cost

    def _ForwardPass(self, xs, us, ks, Ks, alpha):
        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert xs.shape == (N + 1, nx)
        assert us.shape == (N, nu)
        assert ks.shape == (N, nu)
        assert Ks.shape == (N, nu, nx)

        xs_new = np.empty_like(xs)
        xs_new[0] = xs[0]
        us_new = us + alpha * ks
        As = np.empty((N, nx, nx))
        Bs = np.empty((N, nx, nu))
        cost_new = 0

        for i in range(N):
            us_new[i] += Ks[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1], As[i], Bs[i] = self._Step(xs_new[i], us_new[i])
            cost_new += self.prob.CalcStageCost(xs_new[i], us_new[i])
        cost_new += self.prob.CalcTerminalCost(xs_new[-1])

        return xs_new, us_new, As, Bs, cost_new

    def _BackwardPass(self, xs, us, As, Bs, regu):
        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert xs.shape == (N + 1, nx)
        assert us.shape == (N, nu)
        assert As.shape == (N, nx, nx)
        assert Bs.shape == (N, nx, nu)

        ks = np.empty((N, nu))
        Ks = np.empty((N, nu, nx))
        delta_V = 0

        V_x, V_xx = self.prob.CalcTerminalCostDerivatives(xs[-1])
        for i in range(N - 1, -1, -1):
            A, B = As[i], Bs[i]
            l_x, l_u, l_xx, l_xu, l_uu = self.prob.CalcStageCostDerivatives(
                xs[i], us[i]
            )

            # Q_terms
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_xu = l_xu + A.T @ V_xx @ B
            Q_uu = l_uu + B.T @ V_xx @ B

            # gains
            Q_xu_regu = Q_xu + regu * A.T @ B
            Q_uu_regu = Q_uu + regu * B.T @ B
            Q_uu_inv = np.linalg.inv(Q_uu_regu)

            k = -Q_uu_inv @ Q_u
            K = -Q_uu_inv @ Q_xu_regu.T
            ks[i], Ks[i] = k, K

            # V_terms
            V_x = Q_x + K.T @ Q_u + Q_xu @ k + K.T @ Q_uu @ k
            V_xx = Q_xx + 2 * K.T @ Q_xu.T + K.T @ Q_uu @ K
            V_xx = 0.5 * (V_xx + V_xx.T)

            # expected cost reduction
            delta_V += Q_u.T @ k + 0.5 * k.T @ Q_uu @ k

        return ks, Ks, delta_V

    def Solve(
        self,
        us_init,
        max_iters=50,
        early_stop=True,
        alphas=0.5 ** np.arange(8),
        regu_init=20,
        max_regu=10000,
        min_regu=0.001,
    ):
        assert max_iters > 0
        assert np.all(np.diff(alphas) < 0)
        assert regu_init > 0
        assert max_regu > 0 and min_regu > 0 and max_regu >= min_regu

        x0 = np.array(self.prob.GetInitialState())
        us = np.array(us_init)

        N, nx, nu = self.N, self.nx, self.nu
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        xs, As, Bs, cost_old = self._Rollout(x0, us)

        regu = min(max(regu_init, min_regu), max_regu)
        for iter in range(max_iters):
            ks, Ks, exp_cost_redu = self._BackwardPass(xs, us, As, Bs, regu)

            if iter > 3 and early_stop and np.abs(exp_cost_redu) < 1e-5:
                break

            # Backtracking line search
            for alpha in alphas:
                xs_new, us_new, As, Bs, cost_new = self._ForwardPass(
                    xs, us, ks, Ks, alpha
                )
                if cost_new < cost_old:
                    # Accept new trajectories and lower regularization
                    cost_old = cost_new
                    xs = xs_new
                    us = us_new
                    regu *= 0.7
                    break
                else:
                    # Reject new trajectories and increase regularization
                    regu *= 2.0
            regu = min(max(regu, min_regu), max_regu)

        return xs, us, ks, Ks
