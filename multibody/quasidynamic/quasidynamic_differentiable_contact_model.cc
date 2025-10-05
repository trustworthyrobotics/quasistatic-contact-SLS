#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "drake/common/text_logging.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/gradient_util.h"
#include "drake/solvers/solve_conic_program.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using systems::ValueProducer;

MatrixXd LinearSolve(const MatrixXd& P, const MatrixXd& r) {
  Eigen::LLT<MatrixXd> solver(P);
  DRAKE_DEMAND(solver.info() == Eigen::Success);
  return solver.solve(r);
}

MatrixX<AutoDiffXd> LinearSolve(const MatrixX<AutoDiffXd>& P_ad,
                                const MatrixX<AutoDiffXd>& r_ad) {
  const MatrixXd P = math::ExtractValue(P_ad);
  Eigen::LDLT<MatrixXd> solver(P);
  DRAKE_DEMAND(solver.info() == Eigen::Success);

  const MatrixXd r = math::ExtractValue(r_ad);
  const MatrixXd x = solver.solve(r);

  MatrixXd dP = math::ExtractGradient(P_ad);
  MatrixXd dr = math::ExtractGradient(r_ad);
  const int num_derivs = std::max(dP.cols(), dr.cols());

  const auto resize_gradient = [num_derivs](MatrixXd& mat) {
    DRAKE_THROW_UNLESS(mat.cols() == 0 || mat.cols() == num_derivs);
    if (mat.cols() == 0) {
      mat.resize(mat.rows(), num_derivs);
      mat.setZero();
    }
  };
  resize_gradient(dP);
  resize_gradient(dr);

  const int num_rows = r.rows();
  const int num_cols = r.cols();

  const MatrixXd dx = solver
                          .solve((dr - math::matGradMult(dP, x))
                                     .reshaped(num_rows, num_cols * num_derivs))
                          .reshaped(num_rows * num_cols, num_derivs);
  const MatrixX<AutoDiffXd> x_ad = math::InitializeAutoDiff(x, dx);
  return x_ad;
}

}  // namespace

template <typename T>
QuasidynamicDifferentiableContactModel<
    T>::QuasidynamicDifferentiableContactModel(systems::SystemScalarConverter
                                                   converter,
                                               double dynamics_smoothing)
    : systems::LeafSystem<T>(std::move(converter)) {
  DRAKE_THROW_UNLESS(dynamics_smoothing >= 0);
  dynamics_smoothing_param_index_ =
      systems::NumericParameterIndex(this->DeclareNumericParameter(
          systems::BasicVector{T(dynamics_smoothing)}));
}

template <typename T>
double QuasidynamicDifferentiableContactModel<T>::get_dynamics_smoothing(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return ExtractDoubleOrThrow(
      context.get_numeric_parameter(dynamics_smoothing_param_index_)
          .value()[0]);
}

template <typename T>
void QuasidynamicDifferentiableContactModel<T>::set_dynamics_smoothing(
    systems::Context<T>* context, double kappa) const {
  this->ValidateContext(context);
  DRAKE_THROW_UNLESS(kappa >= 0);
  context->get_mutable_numeric_parameter(dynamics_smoothing_param_index_)
      .get_mutable_value()[0] = kappa;
}

template <typename T>
void QuasidynamicDifferentiableContactModel<T>::DeclareStateAndPorts(
    const Eigen::Ref<const VectorX<T>>& x, int num_inputs, int num_contacts,
    int contact_dim, double dt) {
  DRAKE_THROW_UNLESS(x.size() > 0);
  DRAKE_THROW_UNLESS(num_inputs > 0);
  DRAKE_THROW_UNLESS(num_contacts > 0);
  DRAKE_THROW_UNLESS(contact_dim == 1 || contact_dim == 2 || contact_dim == 3);
  DRAKE_THROW_UNLESS(dt > 0);

  num_contacts_ = num_contacts;
  contact_dim_ = contact_dim;

  x_state_index_ = this->DeclareDiscreteState(x);
  x_output_index_ =
      this->DeclareStateOutputPort("x", x_state_index_).get_index();
  u_input_index_ = this->DeclareVectorInputPort("u", num_inputs).get_index();
  this->DeclarePeriodicDiscreteUpdateEvent(
      dt, 0.0, &QuasidynamicDifferentiableContactModel<T>::UpdateState);
  lambda_output_index_ =
      this->DeclareVectorOutputPort(
              "lambda", num_contacts * contact_dim,
              &QuasidynamicDifferentiableContactModel<T>::CalcLambdaOutput,
              {this->discrete_state_ticket(x_state_index_)})
          .get_index();
  lambda_cache_index_ =
      this->DeclareCacheEntry(
              "contact_forces",
              ValueProducer(
                  VectorX<T>(VectorX<T>::Zero(num_contacts * contact_dim)),
                  &ValueProducer::NoopCalc),
              {this->nothing_ticket()})
          .cache_index();
  E_cache_index_ =
      this->DeclareCacheEntry("disturbance_matrix",
                              ValueProducer(MatrixX<T>(MatrixX<T>::Zero(
                                                x.size(), num_contacts)),
                                            &ValueProducer::NoopCalc),
                              {this->nothing_ticket()})
          .cache_index();
}

template <typename T>
const systems::InputPort<T>&
QuasidynamicDifferentiableContactModel<T>::get_actuation_input_port() const {
  return this->get_input_port(u_input_index_);
}

template <typename T>
const systems::OutputPort<T>&
QuasidynamicDifferentiableContactModel<T>::get_state_output_port() const {
  return this->get_output_port(x_output_index_);
}

template <typename T>
const systems::OutputPort<T>& QuasidynamicDifferentiableContactModel<
    T>::get_contact_force_output_port() const {
  return this->get_output_port(lambda_output_index_);
}

template <typename T>
const VectorX<T>& QuasidynamicDifferentiableContactModel<T>::GetState(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return context.get_discrete_state(x_state_index_).value();
}

template <typename T>
void QuasidynamicDifferentiableContactModel<T>::SetState(
    systems::Context<T>* context, const Eigen::Ref<const VectorX<T>>& x) const {
  this->ValidateContext(context);
  DRAKE_THROW_UNLESS(x.size() ==
                     context->get_discrete_state(x_state_index_).size());
  context->SetDiscreteState(x_state_index_, x);
}

template <typename T>
const MatrixX<T>&
QuasidynamicDifferentiableContactModel<T>::GetDisturbanceMatrix(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  const MatrixX<T>& E = this->get_cache_entry(E_cache_index_)
                            .get_cache_entry_value(context)
                            .template PeekValueOrThrow<MatrixX<T>>();
  return E;
}

template <typename T>
void QuasidynamicDifferentiableContactModel<T>::UpdateState(
    const systems::Context<T>& context,
    systems::DiscreteValues<T>* values) const {
  const VectorX<T>& x = GetState(context);
  const int n_x = x.size();
  const int n_l = contact_dim_ * num_contacts_;
  const int n_c = num_contacts_;
  const int dim = contact_dim_;

  const MatrixX<T>& P = EvalQuasidynamicPMatrix(context);
  const VectorX<T>& q = EvalQuasidynamicBVector(context);
  const MatrixX<T>& J = EvalContactJacobian(context);
  const VectorX<T>& phi = EvalContactSignedDistance(context);
  const VectorX<double>& mu = EvalContactFrictionCoefficient(context);
  DRAKE_DEMAND(x.size() == n_x);
  DRAKE_DEMAND(P.rows() == n_x && P.cols() == n_x);
  DRAKE_DEMAND(q.size() == n_x);
  DRAKE_DEMAND(J.rows() == n_l && J.cols() == n_x);
  DRAKE_DEMAND(phi.size() == n_c);
  DRAKE_DEMAND(mu.size() == n_c);

  MatrixX<double> d = MatrixX<double>::Ones(dim, n_c);
  d.row(0) = mu.cwiseInverse();
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> D =
      d.reshaped().asDiagonal();
  MatrixX<T> phi_padded = MatrixX<double>::Zero(dim, n_c);
  phi_padded.row(0) = phi;

  const MatrixX<T> A = D * -J;
  const VectorX<T> b = D * (-J * x + phi_padded.reshaped());

  std::vector<solvers::Cone> cones;
  if (dim == 1) {
    cones.push_back(solvers::NonnegativeCone(n_c));
  } else {
    cones = std::vector<solvers::Cone>(n_c, solvers::SecondOrderCone(dim));
  }

  const double kappa = get_dynamics_smoothing(context);
  VectorX<T> dxnext_dkappa, dlambda_dkappa;
  auto [xnext, lambda] = solvers::SolveConicProgram(
      P, q, A, b, cones, kappa, &dxnext_dkappa, &dlambda_dkappa);
  lambda = D * lambda;
  dlambda_dkappa = D * dlambda_dkappa;

  // Eⱼ = −P⁻¹ Jⱼᵀ (∂λⱼ/∂κ) κ
  MatrixX<T> E(n_x, n_c);
  for (int j = 0; j < n_c; ++j) {
    E.col(j) = -J.middleRows(dim * j, dim).transpose() *
               dlambda_dkappa.segment(dim * j, dim) * kappa;
  }
  E = LinearSolve(P, E);

  systems::BasicVector<T>& xnext_out =
      values->get_mutable_vector(x_state_index_);
  systems::CacheEntryValue& lambda_out =
      this->get_cache_entry(lambda_cache_index_)
          .get_mutable_cache_entry_value(context);
  lambda_out.mark_out_of_date();
  systems::CacheEntryValue& E_out = this->get_cache_entry(E_cache_index_)
                                        .get_mutable_cache_entry_value(context);
  E_out.mark_out_of_date();

  xnext_out.set_value(xnext);
  lambda_out.SetValueOrThrow(lambda);
  E_out.SetValueOrThrow(E);
}

template <typename T>
void QuasidynamicDifferentiableContactModel<T>::CalcLambdaOutput(
    const systems::Context<T>& context, systems::BasicVector<T>* output) const {
  const VectorX<T>& lambda = this->get_cache_entry(lambda_cache_index_)
                                 .get_cache_entry_value(context)
                                 .template PeekValueOrThrow<VectorX<T>>();
  output->set_value(lambda);
}

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::
        QuasidynamicDifferentiableContactModel);
