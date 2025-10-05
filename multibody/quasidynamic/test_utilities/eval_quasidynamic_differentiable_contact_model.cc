#include "drake/multibody/quasidynamic/test_utilities/eval_quasidynamic_differentiable_contact_model.h"

#include <memory>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using systems::Context;
using systems::System;

template <typename T>
std::tuple<VectorXd, VectorXd, MatrixXd> EvalStep(
    const QuasidynamicDifferentiableContactModel<T>& model, Context<T>* context,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u) {
  model.ValidateContext(context);
  const int n_x = model.get_state_output_port().size();
  const int n_u = model.get_actuation_input_port().size();
  const int n_l = model.get_contact_force_output_port().size();
  DRAKE_THROW_UNLESS(x.size() == n_x);
  DRAKE_THROW_UNLESS(u.size() == n_u);

  model.SetState(context, x);
  model.get_actuation_input_port().FixValue(context, u);

  context->SetDiscreteState(model.EvalUniquePeriodicDiscreteUpdate(*context));

  const VectorXd xnext = model.get_state_output_port().Eval(*context);
  const VectorXd lambda = model.get_contact_force_output_port().Eval(*context);
  const MatrixXd E = model.GetDisturbanceMatrix(*context);
  DRAKE_DEMAND(xnext.size() == n_x);
  DRAKE_DEMAND(lambda.size() == n_l);
  DRAKE_DEMAND(E.rows() == n_x);

  return {xnext, lambda, E};
}

template <typename T>
std::tuple<MatrixXd, MatrixXd, MatrixXd> EvalJacobian(
    const QuasidynamicDifferentiableContactModel<T>& model_in,
    Context<T>* context_in, const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u) {
  model_in.ValidateContext(context_in);
  const int n_x = model_in.get_state_output_port().size();
  const int n_u = model_in.get_actuation_input_port().size();
  DRAKE_THROW_UNLESS(x.size() == n_x);
  DRAKE_THROW_UNLESS(u.size() == n_u);

  const QuasidynamicDifferentiableContactModel<AutoDiffXd>* model;
  Context<AutoDiffXd>* context;

  std::unique_ptr<System<AutoDiffXd>> model_ad;
  std::unique_ptr<Context<AutoDiffXd>> context_ad;
  if constexpr (!std::is_same_v<T, AutoDiffXd>) {
    model_ad = model_in.ToAutoDiffXd();
    model =
        dynamic_cast<const QuasidynamicDifferentiableContactModel<AutoDiffXd>*>(
            model_ad.get());
    context_ad = model_ad->CreateDefaultContext();
    context_ad->SetTimeStateAndParametersFrom(*context_in);
    context = context_ad.get();
  } else {
    model = &model_in;
    context = context_in;
  }

  auto autodiff_args = math::InitializeAutoDiffTuple(x, u);
  model->SetState(context, std::get<0>(autodiff_args));
  model->get_actuation_input_port().FixValue(context,
                                             std::get<1>(autodiff_args));

  context->SetDiscreteState(model->EvalUniquePeriodicDiscreteUpdate(*context));

  const VectorX<AutoDiffXd> xnext =
      model->get_state_output_port().Eval(*context);
  const VectorX<AutoDiffXd> lambda =
      model->get_contact_force_output_port().Eval(*context);
  const MatrixX<AutoDiffXd> E = model->GetDisturbanceMatrix(*context);

  const MatrixXd dxnext_dxu = math::ExtractGradient(xnext);
  const MatrixXd dlambda_dxu = math::ExtractGradient(lambda);
  const MatrixXd dE_dxu = math::ExtractGradient(E);

  return {dxnext_dxu, dlambda_dxu, dE_dxu};
}

template <typename T>
std::tuple<MatrixXd, MatrixXd, MatrixXd> ComputeFiniteDiffJacobian(
    const QuasidynamicDifferentiableContactModel<T>& model, Context<T>* context,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u, double epsilon) {
  model.ValidateContext(context);
  const int n_x = model.get_state_output_port().size();
  const int n_u = model.get_actuation_input_port().size();
  const int n_l = model.get_contact_force_output_port().size();
  const int n_E = model.GetDisturbanceMatrix(*context).size();
  DRAKE_THROW_UNLESS(x.size() == n_x);
  DRAKE_THROW_UNLESS(u.size() == n_u);

  MatrixXd dxnext_dxu(n_x, n_x + n_u);
  MatrixXd dlambda_dxu(n_l, n_x + n_u);
  MatrixXd dE_dxu(n_E, n_x + n_u);

  VectorXd xu(n_x + n_u);
  xu.head(n_x) = x;
  xu.tail(n_u) = u;

  VectorXd xnext1, xnext2;
  VectorXd lambda1, lambda2;
  MatrixXd E1, E2;
  for (int j = 0; j < n_x + n_u; ++j) {
    xu[j] -= epsilon;
    std::tie(xnext1, lambda1, E1) =
        EvalStep(model, context, xu.head(n_x), xu.tail(n_u));
    xu[j] += 2 * epsilon;
    std::tie(xnext2, lambda2, E2) =
        EvalStep(model, context, xu.head(n_x), xu.tail(n_u));
    xu[j] -= epsilon;

    dxnext_dxu.col(j) = (xnext2 - xnext1) / (2 * epsilon);
    dlambda_dxu.col(j) = (lambda2 - lambda1) / (2 * epsilon);
    dE_dxu.col(j) = (E2 - E1).reshaped() / (2 * epsilon);
  }

  return {dxnext_dxu, dlambda_dxu, dE_dxu};
}

MatrixXd PseudoInverse(const MatrixXd& A, double rcond) {
  DRAKE_THROW_UNLESS(rcond >= 0);
  Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& S = svd.singularValues();

  Eigen::VectorXd Sinv = S;
  const double cutoff = rcond * S.array().abs().maxCoeff();

  for (int i = 0; i < S.size(); ++i)
    Sinv(i) = (S(i) > cutoff) ? 1.0 / S(i) : 0.0;

  return svd.matrixV() * Sinv.asDiagonal() * svd.matrixU().transpose();
}

template std::tuple<VectorXd, VectorXd, MatrixXd> EvalStep<double>(
    const QuasidynamicDifferentiableContactModel<double>&, Context<double>*,
    const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::VectorXd>&);

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&EvalJacobian<T>));

template std::tuple<MatrixXd, MatrixXd, MatrixXd> ComputeFiniteDiffJacobian<
    double>(const QuasidynamicDifferentiableContactModel<double>&,
            Context<double>*, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&, double);

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake
