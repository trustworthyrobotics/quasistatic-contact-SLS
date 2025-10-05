import numpy as np

from pydrake.all import (
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiffTuple,
    MathematicalProgram,
    SolutionResult,
    Solve,
)


def IterativeTO(
    prob,
    us_init,
    trust_region_covar=None,
    use_contact_trust_region=True,
    use_true_dynamics_for_rollout=False,
    use_true_dynamics_for_constraint=False,
    **kwargs,
):
    alg = IterativeTOAlgorithm(
        prob,
        trust_region_covar=trust_region_covar,
        use_contact_trust_region=use_contact_trust_region,
        use_true_dynamics_for_rollout=use_true_dynamics_for_rollout,
        use_true_dynamics_for_constraint=use_true_dynamics_for_constraint,
    )
    xs, us = alg.Solve(us_init=us_init, **kwargs)

    class NominalTrajectory:
        def __init__(self, xs, us):
            self.xs = xs
            self.us = us

    return NominalTrajectory(xs, us)


class IterativeTOAlgorithm:
    def __init__(
        self,
        prob,
        trust_region_covar=None,
        use_contact_trust_region=True,
        use_true_dynamics_for_rollout=False,
        use_true_dynamics_for_constraint=False,
    ):
        self.prob = prob
        self.trust_region_covar = trust_region_covar
        self.use_contact_trust_region = use_contact_trust_region
        self.use_true_dynamics_for_rollout = use_true_dynamics_for_rollout
        self.use_true_dynamics_for_constraint = use_true_dynamics_for_constraint

        if use_contact_trust_region:
            self.contact_force_size = (
                prob.system.get_contact_force_output_port().size()
            )
            self.contact_dim = prob.system.contact_dim()

        self.N = prob.N
        self.nx = prob.nx
        self.nu = prob.nu

    def _Step(self, x, u, disable_smoothing=False):
        system, context = self.prob.system_ad, self.prob.context_ad
        nx, nu = self.nx, self.nu
        assert len(x) == nx
        assert len(u) == nu

        if disable_smoothing:
            dynamics_smoothing = system.get_dynamics_smoothing(context)
            geometry_smoothing = system.get_geometry_smoothing(context)
            system.set_dynamics_smoothing(context, 0.0)
            system.set_geometry_smoothing(context, 0.0)

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

        if self.use_contact_trust_region:
            lambda_ad = system.get_contact_force_output_port().Eval(context)
            lambda_ = np.squeeze(ExtractValue(lambda_ad))
            dlambda_dxu = ExtractGradient(lambda_ad)
            dlambda_dx = dlambda_dxu[:, 0:nx]
            dlambda_du = dlambda_dxu[:, -nu:]

        if disable_smoothing:
            system.set_dynamics_smoothing(context, dynamics_smoothing)
            system.set_geometry_smoothing(context, geometry_smoothing)

        if not self.use_contact_trust_region:
            return xnext, dxnext_dx, dxnext_du
        else:
            return xnext, dxnext_dx, dxnext_du, lambda_, dlambda_dx, dlambda_du

    def _Rollout(self, x0, us):
        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        xs = np.empty((N + 1, nx))
        xs[0] = x0
        As = np.empty((N, nx, nx))
        Bs = np.empty((N, nx, nu))
        if self.use_contact_trust_region:
            nl = self.contact_force_size
            lambdas = np.empty((N, nl))
            Cs = np.empty((N, nl, nx))
            Ds = np.empty((N, nl, nu))
        if self.use_true_dynamics_for_constraint:
            As_true = np.empty((N, nx, nx))
            Bs_true = np.empty((N, nx, nu))

        cost = 0

        for i in range(N):
            dynamics = self._Step(xs[i], us[i])
            if not self.use_contact_trust_region:
                (xs[i + 1], As[i], Bs[i]) = dynamics
            else:
                (xs[i + 1], As[i], Bs[i], lambdas[i], Cs[i], Ds[i]) = dynamics

            if not (
                self.use_true_dynamics_for_rollout
                or self.use_true_dynamics_for_constraint
            ):
                continue
            dynamics = self._Step(xs[i], us[i], disable_smoothing=True)
            if self.use_true_dynamics_for_rollout:
                if not self.use_contact_trust_region:
                    (xs[i + 1], _, _) = dynamics
                else:
                    (xs[i + 1], _, _, lambdas[i], _, _) = dynamics
            if self.use_true_dynamics_for_constraint:
                if not self.use_contact_trust_region:
                    (_, As_true[i], Bs_true[i]) = dynamics
                else:
                    (_, As_true[i], Bs_true[i], _, _, _) = dynamics

        cost = self.prob.CalcTotalCost(xs, us)

        if not self.use_contact_trust_region:
            if not self.use_true_dynamics_for_constraint:
                return (xs, As, Bs, None, None, None, None, None), cost
            else:
                return (xs, As, Bs, None, None, None, As_true, Bs_true), cost
        else:
            if not self.use_true_dynamics_for_constraint:
                return (xs, As, Bs, lambdas, Cs, Ds, None, None), cost
            else:
                return (xs, As, Bs, lambdas, Cs, Ds, As_true, Bs_true), cost

    def _BackwardPass(
        self,
        xs,
        As,
        Bs,
        lambdas=None,
        Cs=None,
        Ds=None,
        As_true=None,
        Bs_true=None,
        us=[],
    ):
        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert xs.shape == (N + 1, nx)
        assert us.shape == (N, nu)
        assert As.shape == (N, nx, nx)
        assert Bs.shape == (N, nx, nu)
        if self.use_contact_trust_region:
            nl = self.contact_force_size
            assert lambdas.shape == (N, nl)
            assert Cs.shape == (N, nl, nx)
            assert Ds.shape == (N, nl, nu)

        Sigma = self.trust_region_covar
        if Sigma is not None:
            assert Sigma.shape == (nx + nu, nx + nu)

        prog = MathematicalProgram()
        delta_xs = prog.NewContinuousVariables(N + 1, nx, "dx")
        delta_us = prog.NewContinuousVariables(N, nu, "du")

        cost = self.prob.CalcTotalCost(xs + delta_xs, us + delta_us)
        prog.AddCost(cost)

        prog.AddLinearEqualityConstraint(delta_xs[0], np.zeros(nx))
        for k in range(N):
            prog.AddLinearEqualityConstraint(
                delta_xs[k + 1] - As[k] @ delta_xs[k] - Bs[k] @ delta_us[k],
                np.zeros(nx),
            )

        if self.use_true_dynamics_for_constraint:
            delta_xs_true = prog.NewContinuousVariables(N + 1, nx, "dx_true")
            prog.AddLinearEqualityConstraint(delta_xs_true[0], np.zeros(nx))
            for k in range(N):
                prog.AddLinearEqualityConstraint(
                    delta_xs_true[k + 1]
                    - As_true[k] @ delta_xs_true[k]
                    - Bs_true[k] @ delta_us[k],
                    np.zeros(nx),
                )

        self.prob.ApplyConstraintToMathematicalProgram(
            prog=prog,
            xs=xs,
            us=us,
            delta_xs=(
                delta_xs_true
                if self.use_true_dynamics_for_constraint
                else delta_xs
            ),
            delta_us=delta_us,
        )

        if Sigma is not None:
            for k in range(N):
                if not self.use_true_dynamics_for_rollout:
                    prog.AddQuadraticConstraint(
                        Q=Sigma,
                        b=np.zeros(nx + nu),
                        lb=-np.inf,
                        ub=1.0,
                        vars=np.concatenate((delta_xs[k], delta_us[k])),
                    )
                else:
                    lim = np.diag(Sigma) ** -0.5
                    prog.AddBoundingBoxConstraint(
                        -lim,
                        lim,
                        np.concatenate((delta_xs[k], delta_us[k])),
                    )

        if self.use_contact_trust_region:
            dim = self.contact_dim
            for k in range(N):
                if dim == 1:
                    prog.AddLinearConstraint(
                        np.hstack((Cs[k], Ds[k])),
                        lb=-lambdas[k],
                        ub=np.full(nl, np.inf),
                        vars=np.concatenate((delta_xs[k], delta_us[k])),
                    )
                else:
                    for contact in range(nl // dim):
                        mask = range(dim * contact, dim * (contact + 1))
                        prog.AddLorentzConeConstraint(
                            np.hstack((Cs[k, mask, :], Ds[k, mask, :])),
                            lambdas[k, mask],
                            np.concatenate((delta_xs[k], delta_us[k])),
                        )

        result = Solve(prog)
        if not result.is_success():
            status = result.get_solution_result()
            if status != SolutionResult.kSolverSpecificError:
                reason = str(status).split(".")[-1]
            else:
                reason = str(result.get_solver_details().rescode)
            raise RuntimeError("Failed to solve backward pass due to " + reason)

        delta_us_val = result.GetSolution(delta_us).reshape(delta_us.shape)
        return delta_us_val

    def Solve(
        self,
        us_init,
        max_iters=50,
        tol=1e-6,
    ):
        assert max_iters > 0

        x0 = np.array(self.prob.GetInitialState())
        us = np.array(us_init)

        N = us.shape[0]
        nx, nu = self.nx, self.nu
        assert x0.shape == (nx,)
        assert us.shape == (N, nu)

        dynamics, cost_old = self._Rollout(x0, us)

        for iter in range(max_iters):
            delta_us = self._BackwardPass(*dynamics, us=us)
            us += delta_us

            dynamics, cost_new = self._Rollout(x0, us)

            if np.abs(cost_old - cost_new) < tol:
                break
            cost_old = cost_new

        xs = dynamics[0]
        return xs, us
