#include "drake/multibody/quasidynamic/quasidynamic_linear_pusher.h"

#include <utility>

#include "drake/geometry/shape_specification.h"
#include "drake/solvers/solve_conic_program.h"

namespace drake {

namespace systems {
namespace scalar_conversion {
template <>
struct Traits<multibody::quasidynamic::QuasidynamicLinearPusher>
    : public NonSymbolicTraits {};
}  // namespace scalar_conversion
}  // namespace systems

namespace multibody {
namespace quasidynamic {

template <typename T>
QuasidynamicLinearPusher<T>::QuasidynamicLinearPusher(double k_a, double m_o,
                                                      double w, double dt,
                                                      double dynamics_smoothing)
    : QuasidynamicDifferentiableContactModel<T>(
          systems::SystemTypeTag<QuasidynamicLinearPusher>{},
          dynamics_smoothing),
      k_a_(k_a),
      m_o_(m_o),
      w_(w),
      dt_(dt),
      P_(Vector2<T>{k_a, m_o / (dt * dt)}.asDiagonal()),
      J_(RowVector2<T>{-1, 1}) {
  DRAKE_THROW_UNLESS(k_a > 0);
  DRAKE_THROW_UNLESS(m_o > 0);
  DRAKE_THROW_UNLESS(w > 0);
  DRAKE_THROW_UNLESS(dt > 0);

  this->DeclareStateAndPorts(
      /* x */ Vector2<T>::Zero(), /* num_inputs */ 1,
      /* num_contacts */ 1, /* contact_dim */ 1, /* dt */ dt);

  b_cache_index_ =
      this->DeclareCacheEntry(
              "b", VectorX<T>(Vector2<T>::Zero()),
              &QuasidynamicLinearPusher<T>::CalcQuasidynamicBVector,
              {this->all_state_ticket(), this->all_input_ports_ticket()})
          .cache_index();

  phi_cache_index_ =
      this->DeclareCacheEntry(
              "phi", VectorX<T>(Vector1<T>::Zero()),
              &QuasidynamicLinearPusher<T>::CalcContactSignedDistance,
              {this->all_state_ticket()})
          .cache_index();

  this->DeclarePerStepPublishEvent(&QuasidynamicLinearPusher<T>::DrawMeshcat);
  this->DeclareForcedPublishEvent(&QuasidynamicLinearPusher<T>::DrawMeshcat);
}

template <typename T>
template <typename U>
QuasidynamicLinearPusher<T>::QuasidynamicLinearPusher(
    const QuasidynamicLinearPusher<U>& other)
    : QuasidynamicLinearPusher<T>(
          other.k_a_, other.m_o_, other.w_, other.dt_,
          other.get_dynamics_smoothing(*other.CreateDefaultContext())) {
  meshcat_ = other.meshcat_;
}

template <typename T>
const MatrixX<T>& QuasidynamicLinearPusher<T>::EvalQuasidynamicPMatrix(
    const systems::Context<T>&) const {
  return P_;
}

template <typename T>
void QuasidynamicLinearPusher<T>::CalcQuasidynamicBVector(
    const systems::Context<T>& context, VectorX<T>* b) const {
  const VectorX<T>& x = this->GetState(context);
  const VectorX<T>& u = this->get_actuation_input_port().Eval(context);
  *b = Vector2<T>{-k_a_ * u[0], -m_o_ * x[1] / (dt_ * dt_)};
}

template <typename T>
const VectorX<T>& QuasidynamicLinearPusher<T>::EvalQuasidynamicBVector(
    const systems::Context<T>& context) const {
  return this->get_cache_entry(b_cache_index_)
      .template Eval<VectorX<T>>(context);
}

template <typename T>
const MatrixX<T>& QuasidynamicLinearPusher<T>::EvalContactJacobian(
    const systems::Context<T>&) const {
  return J_;
}

template <typename T>
void QuasidynamicLinearPusher<T>::CalcContactSignedDistance(
    const systems::Context<T>& context, VectorX<T>* phi) const {
  const VectorX<T>& x = this->GetState(context);
  (*phi)[0] = x[1] - x[0] - w_;
}

template <typename T>
const VectorX<T>& QuasidynamicLinearPusher<T>::EvalContactSignedDistance(
    const systems::Context<T>& context) const {
  return this->get_cache_entry(phi_cache_index_)
      .template Eval<VectorX<T>>(context);
}

template <typename T>
const VectorX<double>&
QuasidynamicLinearPusher<T>::EvalContactFrictionCoefficient(
    const systems::Context<T>&) const {
  static VectorX<double> mu = Vector1<double>{1.0};
  return mu;
}

template <typename T>
void QuasidynamicLinearPusher<T>::SetMeshcat(geometry::Meshcat* meshcat) {
  meshcat_ = meshcat;
}

template <typename T>
systems::EventStatus QuasidynamicLinearPusher<T>::DrawMeshcat(
    const systems::Context<T>& context) const {
  if (meshcat_ == nullptr) return systems::EventStatus::DidNothing();
  const double t = ExtractDoubleOrThrow(context.get_time());
  const Vector2<double> x = ExtractDoubleOrThrow(this->GetState(context));
  constexpr std::string_view pusher_path = "/quasidynamic_linear_pusher/pusher";
  constexpr std::string_view object_path = "/quasidynamic_linear_pusher/object";
  meshcat_->SetObject(pusher_path, geometry::Sphere(w_ / 2),
                      geometry::Rgba(1, 0, 0, 1));
  meshcat_->SetObject(object_path, geometry::Box(w_, w_, w_),
                      geometry::Rgba(0, 0, 1, 1));
  meshcat_->SetTransform(pusher_path,
                         math::RigidTransform(Vector3<double>(x[0], 0, 0)), t);
  meshcat_->SetTransform(object_path,
                         math::RigidTransform(Vector3<double>(x[1], 0, 0)), t);
  return systems::EventStatus::Succeeded();
}

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::QuasidynamicLinearPusher);
