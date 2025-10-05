#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/rgba.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/multibody/tree/spatial_inertia.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

// Forward declaration
template <typename T>
struct QuasidynamicMultibodyPlantSparseMatrix;

/**
 @p QuasidynamicMultibodyPlant represents a spatial quasidynamic model.
 The dynamics are governed by
    P(x) x⁺ + b(x,u) − J(x)ᵀ λ = 0.
 */
template <typename T>
class QuasidynamicMultibodyPlant
    : public QuasidynamicDifferentiableContactModel<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasidynamicMultibodyPlant);

  /** Constructs a quasidynamic planar plant.
   @param dt Time step size.
   @param dynamics_smoothing The dynamics smoothing parameter.
   @param geometry_smoothing The geometry smoothing parameter.
   @pre `dt > 0`.
   @pre `dynamics_smoothing >= 0`.
   @pre `geometry_smoothing >= 0`. */
  explicit QuasidynamicMultibodyPlant(double dt, double dynamics_smoothing = 0,
                                      double geometry_smoothing = 0);

  /* Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  template <typename U>
  explicit QuasidynamicMultibodyPlant(
      const QuasidynamicMultibodyPlant<U>& other);

  /** Adds a rigid body with the provided spatial inertia.
   @param M_BBo_B Spatial inertia of the rigid body, computed about the body
                  frame origin Bo and expressed in the body frame B.
   @throws std::exception if called post-finalize. */
  BodyIndex AddRigidBody(
      const SpatialInertia<double>& M_BBo_B = SpatialInertia<double>::Zero());

  /** Returns the index corresponding to the world body. */
  BodyIndex world_body() const;

  /** Adds a revolute joint.
   @param body_P Index of the parent body.
   @param X_PJp The fixed pose of frame Jp attached to the parent frame P.
   @param body_C Index of the child body.
   @param X_CJc The fixed pose of frame Jc attached to the child frame C.
   @param axis A vector specifying the axis of revolution for this joint.
   @param actuation_stiffness If not null, make this joint actuated with the
                              specified stiffness.
   @pre `body_P` and `body_C` are valid indexes.
   @pre `actuation_stiffness` is positive is not null.
   @throws std::exception if called post-finalize. */
  JointIndex AddRevoluteJoint(
      BodyIndex body_P, const math::RigidTransform<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform<double>& X_CJc,
      const Vector3<double>& axis,
      std::optional<double> actuation_stiffness = std::nullopt);

  /** Adds a prismatic joint.
   @param body_P Index of the parent body.
   @param X_PJp The fixed pose of frame Jp attached to the parent frame P.
   @param body_C Index of the child body.
   @param X_CJc The fixed pose of frame Jc attached to the child frame C.
   @param axis A vector specifying the translation axis for this joint.
   @param actuation_stiffness If not null, make this joint actuated with the
                              specified stiffness.
   @pre `body_P` and `body_C` are valid indexes.
   @pre `actuation_stiffness` is positive is not null.
   @throws std::exception if called post-finalize. */
  JointIndex AddPrismaticJoint(
      BodyIndex body_P, const math::RigidTransform<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform<double>& X_CJc,
      const Vector3<double>& axis,
      std::optional<double> actuation_stiffness = std::nullopt);

  /** Adds a Cartesian joint.
   @param body_P Index of the parent body.
   @param X_PJp The fixed pose of frame Jp attached to the parent frame P.
   @param body_C Index of the child body.
   @param X_CJc The fixed pose of frame Jc attached to the child frame C.
   @param actuation_stiffness If not null, make this joint actuated with the
                              specified stiffness.
   @pre `body_P` and `body_C` are valid indexes.
   @pre `actuation_stiffness` is positive is not null.
   @throws std::exception if called post-finalize. */
  JointIndex AddCartesianJoint(
      BodyIndex body_P, const math::RigidTransform<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform<double>& X_CJc,
      const std::optional<Vector3<double>>& actuation_stiffness = std::nullopt);

  /** Adds a roll-pitch-yaw floating joint.
   @param body_P Index of the parent body.
   @param X_PJp The fixed pose of frame Jp attached to the parent frame P.
   @param body_C Index of the child body.
   @param X_CJc The fixed pose of frame Jc attached to the child frame C.
   @param actuation_stiffness If not null, make this joint actuated with the
                              specified stiffness.
   @pre `body_P` and `body_C` are valid indexes.
   @pre `actuation_stiffness` is positive is not null.
   @throws std::exception if called post-finalize. */
  JointIndex AddRpyFloatingJoint(
      BodyIndex body_P, const math::RigidTransform<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform<double>& X_CJc,
      const std::optional<Vector6<double>>& actuation_stiffness = std::nullopt);

  /** Registers a geometry to be used for collision for a given body.
   @param body The body for which geometry is being registered.
   @param X_BG The fixed pose of the geometry frame G in the body frame B.
   @param shape The geometry used for collision.
   @param friction_coefficient The friction coefficient μ of the object.
   The effective coefficient of friction at the interface of two objects is
   computed as 2μ₁μ₂ / (μ₁+μ₂).
   @pre `body` is a valid index.
   @pre `friction_coefficient >= 0`.
   @throws std::exception if called post-finalize. */
  geometry::GeometryId RegisterCollisionGeometry(
      BodyIndex body, const math::RigidTransform<double>& X_BG,
      const geometry::Shape& shape, double friction_coefficient);

  /** Registers a geometry to be used for visualization for a given body.
   @param body The body for which geometry is being registered.
   @param X_BG The fixed pose of the geometry frame G in the body frame B.
   @param shape The geometry used for visualization.
   @param rgba The color for the geometry.
   @param name The name for the geometry.
   @pre `body` is a valid index.
   @throws std::exception if called post-finalize. */
  geometry::GeometryId RegisterVisualGeometry(
      BodyIndex body, const math::RigidTransform<double>& X_BG,
      const geometry::Shape& shape, const geometry::Rgba& rgba,
      std::optional<std::string_view> name = std::nullopt);

  /** Sets the acceleration of gravity vector, expressed in the world frame W.
   The default gravity vector is set to [0, 0, -9.80665].
   @throws std::exception if called post-finalize. */
  void SetGravityVector(const Vector3<double>& g_W);

  /** Finalizes the QuasidynamicMultibodyPlant. This method must be called after
   all elements in the model (bodies, joints, geometries) are added and before
   any computations are performed. */
  void Finalize();

  /** Gets the geometry smoothing parameter. */
  double get_geometry_smoothing(const systems::Context<T>& context) const;

  /** Sets the geometry smoothing parameter. (Zero means no smoothing and larger
   values mean higher smoothing.)
   @pre `geometry_smoothing >= 0`. */
  void set_geometry_smoothing(systems::Context<T>* context,
                              double geometry_smoothing) const;

  /** Evaluates the P(x) matrix governing the quasidynamic model.
   @throws std::exception if called pre-finalize. */
  const MatrixX<T>& EvalQuasidynamicPMatrix(
      const systems::Context<T>& context) const override;

  /** Evaluates the b(x,u) vector governing the quasidynamic model.
   @throws std::exception if called pre-finalize. */
  const VectorX<T>& EvalQuasidynamicBVector(
      const systems::Context<T>& context) const override;

  /** Evaluates the contact jacobian.
   @throws std::exception if called pre-finalize. */
  const MatrixX<T>& EvalContactJacobian(
      const systems::Context<T>& context) const override;

  /** Evaluates the contact signed distance.
   @throws std::exception if called pre-finalize. */
  const VectorX<T>& EvalContactSignedDistance(
      const systems::Context<T>& context) const override;

  /** Evaluates the friction coefficient for the contact.
   @throws std::exception if called pre-finalize. */
  const VectorX<double>& EvalContactFrictionCoefficient(
      const systems::Context<T>& context) const override;

  /** Evaluates the pose of a body in the world frame.
   @throws std::exception if called pre-finalize.
   @pre `body` is a valid index. */
  const math::RigidTransform<T>& EvalPoseInWorld(
      const systems::Context<T>& context, BodyIndex body) const;

  /** Evaluates the collision jacobian.
   @throws std::exception if called pre-finalize. */
  const MatrixX<T>& EvalCollisionJacobian(
      const systems::Context<T>& context) const;

  /** Evaluates the collision signed distance.
   @throws std::exception if called pre-finalize. */
  const VectorX<T>& EvalCollisionSignedDistance(
      const systems::Context<T>& context) const;

  /** Sets the meshcat for which to draw to. */
  void SetMeshcat(
      geometry::Meshcat* meshcat,
      std::string_view name_prefix = "quasidynamic_multibody_plant");

 private:
  /* Allows QuasidynamicMultibodyPlant<U> to access private members of
   QuasidynamicMultibodyPlant<T> for scalar conversion. */
  template <typename U>
  friend class QuasidynamicMultibodyPlant;

  /* Evaluates the matrix that maps q̇ to v.
   @throws std::exception if called pre-finalize. */
  const Eigen::SparseMatrix<T>& EvalQDotToVelocityMap(
      const systems::Context<T>& context) const;

  /* Calculation for EvalQDotToVelocityMap. */
  void CalcQDotToVelocityMap(
      const systems::Context<T>& context,
      QuasidynamicMultibodyPlantSparseMatrix<T>* N) const;

  /* Evaluates the unactuated mass matrix.
   @throws std::exception if called pre-finalize. */
  const MatrixX<T>& EvalUnactuatedMassMatrix(
      const systems::Context<T>& context) const;

  /* Calculation for EvalUnactuatedMassMatrix. */
  void CalcUnactuatedMassMatrix(const systems::Context<T>& context,
                                MatrixX<T>* M_unactuated) const;

  /* Calculation for EvalQuasidynamicPMatrix. */
  void CalcQuasidynamicPMatrix(const systems::Context<T>& context,
                               MatrixX<T>* P_quasidynamic) const;

  /* Calculation for EvalQuasidynamicBVector. */
  void CalcQuasidynamicBVector(const systems::Context<T>& context,
                               VectorX<T>* b_quasidynamic) const;

  /* Calculation for EvalContactJacobian and EvalContactSignedDistance. */
  void CalcContact(const systems::Context<T>& context,
                   std::tuple<MatrixX<T>, VectorX<T>>* contact) const;

  /* Calculation for EvalCollisionJacobian and EvalCollisionSignedDistance. */
  void CalcCollision(const systems::Context<T>& context,
                     std::tuple<MatrixX<T>, VectorX<T>>* collision) const;

  systems::EventStatus DrawMeshcat(const systems::Context<T>& context) const;

  class Impl;
  struct ImplDeleter {
    void operator()(Impl*);
  };

  const double dt_{};
  std::unique_ptr<Impl, ImplDeleter> impl_;
  systems::NumericParameterIndex geometry_smoothing_param_index_{};
  systems::CacheIndex qdot_to_vel_map_cache_index_{};
  systems::CacheIndex M_unactuated_cache_index_{};
  systems::CacheIndex P_quasidynamic_cache_index_{};
  systems::CacheIndex b_quasidynamic_cache_index_{};
  systems::CacheIndex contact_cache_index_{};
  systems::CacheIndex collision_cache_index_{};
  systems::CacheIndex scratch_cache_index_{};
};

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::QuasidynamicMultibodyPlant);
