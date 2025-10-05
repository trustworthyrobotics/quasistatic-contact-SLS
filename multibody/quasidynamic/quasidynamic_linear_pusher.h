#pragma once

#include "drake/geometry/meshcat.h"
#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

/**
 @p QuasidynamicLinearPusher represents a 1D pusher controlled by a stiffness
 controller and pushes an object.
 */
template <typename T>
class QuasidynamicLinearPusher
    : public QuasidynamicDifferentiableContactModel<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasidynamicLinearPusher);

  /** Constructs a linear pusher.
   @param k_a Stiffness of the pusher controller.
   @param m_o Mass of the pushed object.
   @param w   Width of both the pusher and the object.
   @param dt  Time step size.
   @param dynamics_smoothing The dynamics smoothing parameter.
   @pre `k_a > 0`.
   @pre `m_o > 0`.
   @pre `w > 0`.
   @pre `dt > 0`.
   @pre `dynamics_smoothing >= 0`. */
  QuasidynamicLinearPusher(double k_a, double m_o, double w, double dt,
                           double dynamics_smoothing = 0);

  /* Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  template <typename U>
  explicit QuasidynamicLinearPusher(const QuasidynamicLinearPusher<U>& other);

  /** Evaluates the P(x) matrix governing the quasidynamic model. */
  const MatrixX<T>& EvalQuasidynamicPMatrix(
      const systems::Context<T>& context) const override;

  /** Evaluates the b(x,u) vector governing the quasidynamic model. */
  const VectorX<T>& EvalQuasidynamicBVector(
      const systems::Context<T>& context) const override;

  /** Evaluates the contact jacobian and signed distance. */
  const MatrixX<T>& EvalContactJacobian(
      const systems::Context<T>& context) const override;

  /** Evaluates the contact signed distance. */
  const VectorX<T>& EvalContactSignedDistance(
      const systems::Context<T>& context) const override;

  /** Evaluates the friction coefficient for the contact. */
  const VectorX<double>& EvalContactFrictionCoefficient(
      const systems::Context<T>& context) const override;

  /** Sets the meshcat for which to draw to. */
  void SetMeshcat(geometry::Meshcat* meshcat);

 private:
  /* Allows QuasidynamicLinearPusher<U> to access private members of
   QuasidynamicLinearPusher<T> for scalar conversion. */
  template <typename U>
  friend class QuasidynamicLinearPusher;

  /* Calculates the b(x,u) vector governing the quasidynamic model. */
  void CalcQuasidynamicBVector(const systems::Context<T>& context,
                               VectorX<T>* b) const;

  /** Calculates the contact signed distance. */
  void CalcContactSignedDistance(const systems::Context<T>& context,
                                 VectorX<T>* phi) const;

  systems::EventStatus DrawMeshcat(const systems::Context<T>& context) const;

  const double k_a_{};
  const double m_o_{};
  const double w_{};
  const double dt_{};
  const MatrixX<T> P_;
  const MatrixX<T> J_;
  systems::CacheIndex b_cache_index_;
  systems::CacheIndex phi_cache_index_;
  mutable geometry::Meshcat* meshcat_{};
};

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::QuasidynamicLinearPusher);
