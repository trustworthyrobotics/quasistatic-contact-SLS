import numpy as np

from pydrake.all import (
    AutoDiffXd,
    LeafSystem_,
    TemplateSystem,
)


def ModelPredictiveController(
    prob,
    solver,
    us_init,
    presolve_callback=None,
    postsolve_callback=None,
    **kwargs,
):
    return Controller(
        prob=prob,
        solver=solver,
        us_init=us_init,
        presolve_callback=presolve_callback,
        postsolve_callback=postsolve_callback,
        kwargs=kwargs,
    )


@TemplateSystem.define("Controller_", T_list=[float, AutoDiffXd])
def Controller_(T):
    class Impl(LeafSystem_[T]):
        def _construct(
            self,
            prob,
            solver,
            us_init,
            presolve_callback,
            postsolve_callback,
            kwargs,
            converter=None,
        ):
            super().__init__(converter)
            self.prob = prob
            self.solver = solver
            self.us = us_init
            self.presolve_callback = presolve_callback
            self.postsolve_callback = postsolve_callback
            self.kwargs = kwargs

            self.DeclareVectorInputPort("x", prob.nx)
            self.DeclareVectorOutputPort("u", prob.nu, self.CalcOutput)

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self,
                prob=other.prob,
                solver=other.solver,
                us_init=other.us_init,
                presolve_callback=other.presolve_callback,
                postsolve_callback=other.postsolve_callback,
                kwargs=other.kwargs,
                converter=converter,
            )

        def CalcOutput(self, context, output):
            x = self.get_input_port().Eval(context)
            prob = self.prob
            prob.SetInitialState(x)

            if self.presolve_callback:
                if not hasattr(self, "u_prev"):
                    self.u_prev = self.us[0]
                self.presolve_callback(prob, self.u_prev)
            try:
                self.us = self.solver(prob, us_init=self.us, **self.kwargs).us
            except Exception:
                pass
            if self.postsolve_callback:
                self.postsolve_callback(prob)

            output.set_value(self.us[0])
            self.u_prev = self.us[0]
            self.us = np.concatenate((self.us[1:], [self.us[-1]]))

    return Impl


Controller = Controller_[None]
