#include "drake/multibody/quasidynamic/quasidynamic_planar_plant.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/overloaded.h"
#include "drake/common/pointer_cast.h"
#include "drake/geometry/proximity/calc_proximity.h"
#include "drake/geometry/shape_specification.h"
#include "drake/multibody/tree/multibody_tree-inl.h"
#include "drake/multibody/tree/multibody_tree.h"
#include "drake/multibody/tree/multibody_tree_system.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/rigid_body.h"

namespace drake {

namespace systems {
namespace scalar_conversion {
template <>
struct Traits<multibody::quasidynamic::QuasidynamicPlanarPlant>
    : public NonSymbolicTraits {};
}  // namespace scalar_conversion
}  // namespace systems

namespace multibody {
namespace quasidynamic {

namespace {

using geometry::GeometryId;

struct RigidBody {
  RigidBody() : mass(0.0), p_BBcm_B(Vector2<double>::Zero()), I_BBcm(0.0) {}

  RigidBody(double mass_in, const Vector2<double>& p_BBcm_B_in,
            double I_BBcm_in)
      : mass(mass_in), p_BBcm_B(p_BBcm_B_in), I_BBcm(I_BBcm_in) {
    DRAKE_THROW_UNLESS(mass >= 0);
    DRAKE_THROW_UNLESS(I_BBcm >= 0);
  }

  double mass{};
  Vector2<double> p_BBcm_B;
  double I_BBcm{};
  std::vector<JointIndex> children;
  std::optional<JointIndex> parent;
  std::vector<GeometryId> collision_geometries;
  std::vector<GeometryId> visual_geometries;
};

struct Joint {
  enum class Type { Revolute, Prismatic, Cartesian, Planar };

  template <typename Derived>
  Joint(Type type_in, BodyIndex body_P_in,
        const math::RigidTransform2d<double>& X_PJp_in, BodyIndex body_C_in,
        const math::RigidTransform2d<double>& X_CJc_in,
        const std::optional<Derived>& actuation_stiffness_in)
      : type(type_in),
        body_P(body_P_in),
        X_PJp(X_PJp_in),
        body_C(body_C_in),
        X_CJc(X_CJc_in) {
    if (actuation_stiffness_in) {
      if constexpr (std::is_same_v<Derived, double>)
        actuation_stiffness = Vector1d{*actuation_stiffness_in};
      else
        actuation_stiffness = *actuation_stiffness_in;
      DRAKE_THROW_UNLESS((actuation_stiffness->array() > 0).all());
    }
  }

  bool is_actuated() const { return actuation_stiffness.has_value(); }

  Type type{};
  BodyIndex body_P{};
  math::RigidTransform2d<double> X_PJp;
  BodyIndex body_C{};
  math::RigidTransform2d<double> X_CJc;
  std::optional<Eigen::VectorXd> actuation_stiffness;
};

struct CollisionGeometry {
  CollisionGeometry(BodyIndex body_in,
                    const math::RigidTransform2d<double>& X_BG_in,
                    const geometry::Shape2d& shape_in,
                    double friction_coefficient_in)
      : body(body_in),
        X_BG(X_BG_in),
        shape(shape_in.Clone()),
        friction_coefficient(friction_coefficient_in) {
    DRAKE_THROW_UNLESS(friction_coefficient >= 0);
  }

  BodyIndex body{};
  math::RigidTransform2d<double> X_BG;
  copyable_unique_ptr<geometry::Shape2d> shape;
  double friction_coefficient;
};

struct VisualGeometry {
  VisualGeometry(BodyIndex body_in,
                 const math::RigidTransform2d<double>& X_BG_in,
                 const geometry::Shape2d& shape_in,
                 const geometry::Rgba& rgba_in,
                 std::optional<std::string_view> name_in = std::nullopt)
      : body(body_in),
        X_BG(X_BG_in),
        shape_2d(shape_in.Clone()),
        rgba(rgba_in),
        name(name_in ? std::optional<std::string>(*name_in) : std::nullopt) {}

  VisualGeometry(BodyIndex body_in, const math::RigidTransform<double>& X_BG_in,
                 const geometry::Shape& shape_in, const geometry::Rgba& rgba_in,
                 std::optional<std::string_view> name_in = std::nullopt)
      : body(body_in),
        X_BG(X_BG_in),
        shape_3d(shape_in.Clone()),
        rgba(rgba_in),
        name(name_in ? std::optional<std::string>(*name_in) : std::nullopt) {}

  BodyIndex body{};
  std::variant<math::RigidTransform2d<double>, math::RigidTransform<double>>
      X_BG;
  copyable_unique_ptr<geometry::Shape2d> shape_2d;
  copyable_unique_ptr<geometry::Shape> shape_3d;
  geometry::Rgba rgba;
  std::optional<std::string> name;
};

template <typename T, int ColsAtCompileTime>
Eigen::Matrix<T, 3, ColsAtCompileTime> Convert2dTo3d(
    const Eigen::Matrix<T, 2, ColsAtCompileTime>& mat_2d) {
  Eigen::Matrix<T, 3, ColsAtCompileTime> mat_3d;
  if constexpr (ColsAtCompileTime == Eigen::Dynamic)
    mat_3d.resize(3, mat_2d.cols());
  mat_3d.setZero();
  mat_3d.template topRows<2>() = mat_2d;
  return mat_3d;
}

template <typename T>
math::RigidTransform<T> Convert2dTo3d(const math::RigidTransform2d<T>& X_2d) {
  return math::RigidTransform<T>(
      math::RotationMatrix<T>::MakeZRotation(X_2d.angle()),
      Convert2dTo3d(X_2d.translation()));
}

template <typename T, int ColsAtCompileTime>
auto Convert3dTo2d(const Eigen::Matrix<T, 3, ColsAtCompileTime>& mat_3d,
                   double check_tolerance = 1e-12) {
  DRAKE_ASSERT(
      (ExtractDoubleOrThrow(mat_3d).row(2).array().abs() < check_tolerance)
          .all());
  return mat_3d.template topRows<2>();
}

template <typename T>
math::RigidTransform2d<T> Convert3dTo2d(const math::RigidTransform<T>& X_3d,
                                        double check_tolerance = 1e-12) {
  const Matrix3<T>& R_3d = X_3d.rotation().matrix();
  DRAKE_ASSERT(((ExtractDoubleOrThrow(R_3d).col(2) - Vector3<double>::UnitZ())
                    .array()
                    .abs() < check_tolerance)
                   .all());
  const T angle = atan2(R_3d(1, 0), R_3d(0, 0));
  return math::RigidTransform2d<T>(angle, Convert3dTo2d(X_3d.translation()));
}

template <typename T>
Matrix2<T> MakeFromUnitXVector(const Vector2<T>& axis) {
  Matrix2<T> R;
  R << axis, Vector2<T>{-axis.y(), axis.x()};
  return R;
}

template <typename T>
T CalcEffectiveFrictionCoefficient(const T& friction_coefficient1,
                                   const T& friction_coefficient2) {
  const auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };
  return safe_divide(2 * friction_coefficient1 * friction_coefficient2,
                     friction_coefficient1 + friction_coefficient2);
}

void AppendCartesianProduct(
    const std::vector<GeometryId>& set1, const std::vector<GeometryId>& set2,
    std::vector<std::pair<GeometryId, GeometryId>>* output,
    bool skip_first_cartesian_pair = false) {
  DRAKE_THROW_UNLESS(output != nullptr);
  for (int i = 0; i < ssize(set1); ++i) {
    const GeometryId a = set1[i];
    for (int j = 0; j < ssize(set2); ++j) {
      const GeometryId b = set2[j];
      if (skip_first_cartesian_pair && i == 0 && j == 0) continue;
      output->emplace_back(a, b);
    }
  }
}

template <typename T>
class Scratch {
 public:
  explicit Scratch(
      const internal::MultibodyTreeSystem<T>* multibody_tree_system)
      : multibody_tree_system_(multibody_tree_system) {
    DRAKE_THROW_UNLESS(multibody_tree_system != nullptr);
    context_ = multibody_tree_system->CreateDefaultContext();
  }

  void SetState(const Eigen::Ref<const VectorX<T>>& x) {
    DRAKE_DEMAND(multibody_tree_system_ != nullptr);
    const internal::MultibodyTree<T>& tree =
        internal::GetInternalTree(*multibody_tree_system_);
    DRAKE_THROW_UNLESS(x.size() == tree.num_positions());
    tree.GetMutablePositionsAndVelocities(context_.get_mutable())
        .head(x.size()) = x;
  }

  const systems::Context<T>& context() const {
    DRAKE_DEMAND(context_ != nullptr);
    return *context_;
  }

 private:
  const internal::MultibodyTreeSystem<T>* multibody_tree_system_;
  copyable_unique_ptr<systems::Context<T>> context_;
};

struct Draw {
  mutable geometry::Meshcat* meshcat{};
  std::string name_prefix;
  double z_height{};
};

}  // namespace

template <typename T>
class QuasidynamicPlanarPlant<T>::Impl {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Impl);

  Impl() {
    // First body is world body.
    bodies_.emplace_back(std::make_unique<RigidBody>());
  }

  BodyIndex AddRigidBody(double mass, const Vector2<double>& p_BBcm_B,
                         double I_BBcm) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(mass > 0);
    DRAKE_THROW_UNLESS(I_BBcm > 0);
    bodies_.emplace_back(std::make_unique<RigidBody>(mass, p_BBcm_B, I_BBcm));
    return BodyIndex(bodies_.size() - 1);
  }

  BodyIndex AddRigidBody() {
    DRAKE_THROW_UNLESS(!finalized_);
    bodies_.emplace_back(std::make_unique<RigidBody>());
    return BodyIndex(bodies_.size() - 1);
  }

  BodyIndex world_body() const { return BodyIndex(0); }

  JointIndex AddRevoluteJoint(BodyIndex body_P,
                              const math::RigidTransform2d<double>& X_PJp,
                              BodyIndex body_C,
                              const math::RigidTransform2d<double>& X_CJc,
                              std::optional<double> actuation_stiffness) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(!actuation_stiffness || actuation_stiffness.value() > 0);
    return AddJoint(Joint(Joint::Type::Revolute, body_P, X_PJp, body_C, X_CJc,
                          actuation_stiffness));
  }

  JointIndex AddPrismaticJoint(BodyIndex body_P,
                               const math::RigidTransform2d<double>& X_PJp,
                               BodyIndex body_C,
                               const math::RigidTransform2d<double>& X_CJc,
                               std::optional<double> actuation_stiffness) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(!actuation_stiffness || actuation_stiffness.value() > 0);
    return AddJoint(Joint(Joint::Type::Prismatic, body_P, X_PJp, body_C, X_CJc,
                          actuation_stiffness));
  }

  JointIndex AddCartesianJoint(
      BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
      const std::optional<Vector2<double>>& actuation_stiffness) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(!actuation_stiffness ||
                       (actuation_stiffness->array() > 0).all());
    return AddJoint(Joint(Joint::Type::Cartesian, body_P, X_PJp, body_C, X_CJc,
                          actuation_stiffness));
  }

  JointIndex AddPlanarJoint(
      BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
      BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
      const std::optional<Vector3<double>>& actuation_stiffness) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(!actuation_stiffness ||
                       (actuation_stiffness->array() > 0).all());
    return AddJoint(Joint(Joint::Type::Planar, body_P, X_PJp, body_C, X_CJc,
                          actuation_stiffness));
  }

  GeometryId RegisterCollisionGeometry(
      BodyIndex body, const math::RigidTransform2d<double>& X_BG,
      const geometry::Shape2d& shape, double friction_coefficient) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(body < bodies_.size());
    DRAKE_THROW_UNLESS(friction_coefficient >= 0);
    GeometryId geometry_id = GeometryId ::get_new_id();
    bodies_[body]->collision_geometries.push_back(geometry_id);
    collision_geometries_.emplace(
        geometry_id,
        CollisionGeometry(body, X_BG, shape, friction_coefficient));
    return geometry_id;
  }

  GeometryId RegisterVisualGeometry(BodyIndex body,
                                    const math::RigidTransform2d<double>& X_BG,
                                    const geometry::Shape2d& shape,
                                    const geometry::Rgba& rgba,
                                    std::optional<std::string_view> name) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(body < bodies_.size());
    GeometryId geometry_id = GeometryId ::get_new_id();
    bodies_[body]->visual_geometries.push_back(geometry_id);
    visual_geometries_.emplace(geometry_id,
                               VisualGeometry(body, X_BG, shape, rgba, name));
    return geometry_id;
  }

  GeometryId RegisterVisualGeometry(BodyIndex body,
                                    const math::RigidTransform<double>& X_BG,
                                    const geometry::Shape& shape,
                                    const geometry::Rgba& rgba,
                                    std::optional<std::string_view> name) {
    DRAKE_THROW_UNLESS(!finalized_);
    DRAKE_THROW_UNLESS(body < bodies_.size());
    GeometryId geometry_id = GeometryId ::get_new_id();
    bodies_[body]->visual_geometries.push_back(geometry_id);
    visual_geometries_.emplace(geometry_id,
                               VisualGeometry(body, X_BG, shape, rgba, name));
    return geometry_id;
  }

  void SetGravityVector(const Vector2<double>& g_W) {
    DRAKE_THROW_UNLESS(!finalized_);
    gravity_ = g_W;
  }

  void Finalize() {
    DRAKE_THROW_UNLESS(!finalized_);
    /* For any body that does not have a parent, which implicitly means world
     body is its parent, explicitly add world body as its parent. */
    for (BodyIndex i{1}; i < ssize(bodies_); ++i) {
      if (!bodies_[i]->parent) {
        AddPlanarJoint(world_body(), math::RigidTransform2d<double>(), i,
                       math::RigidTransform2d<double>(), std::nullopt);
      }
    }
    /* Check that there are no loops in the kinematic chain, and each tree
     branch is either fully actuated or fully unactuated. */
    std::vector<bool> visited(bodies_.size(), false);
    const std::function<void(BodyIndex, bool)> recursive_check =
        [this, &visited, &recursive_check](BodyIndex body_index,
                                           bool actuated) {
          if (visited[body_index]) {
            throw std::runtime_error("There is a loop in the kinematic chain.");
          }
          visited[body_index] = true;

          const RigidBody& body = *this->bodies_[body_index];
          if (!actuated) {
            if (body.mass == 0) {
              throw std::runtime_error(fmt::format(
                  "Rigid body {} is unactuated but has zero mass", body_index));
            }
            if (body.I_BBcm == 0) {
              throw std::runtime_error(fmt::format(
                  "Rigid body {} is unactuated but has zero rotational inertia",
                  body_index));
            }
          }

          for (JointIndex joint_index : body.children) {
            const Joint& joint = *this->joints_[joint_index];
            if (joint.is_actuated() != actuated) {
              throw std::runtime_error(
                  fmt::format("There is a branch in the kinematic chain that "
                              "is not fully {}.",
                              actuated ? "actuated" : "unactuated"));
            }
            recursive_check(joint.body_C, actuated);
          }
        };
    visited[0] = true;
    for (JointIndex joint_index : bodies_[0]->children) {
      const Joint& joint = *this->joints_[joint_index];
      recursive_check(joint.body_C, joint.is_actuated());
    }
    /* Make the drake::multibody::internal::MultibodyTree,
     contact pairs, and collision pairs. */
    multibody_tree_system_ = MakeMultibodyTreeSystem();
    actuation_stiffness_ = MakeActuationStiffness();
    std::tie(contact_pairs_, friction_coefficients_) = MakeContactPairs();
    collision_pairs_ = MakeCollisionPairs();
    finalized_ = true;
  }

  int num_dofs() const {
    DRAKE_THROW_UNLESS(finalized_);
    return multibody_tree().num_positions();
  }

  int num_actuated_dofs() const {
    DRAKE_THROW_UNLESS(finalized_);
    return ssize(actuation_stiffness_);
  }

  int num_unactuated_dofs() const {
    DRAKE_THROW_UNLESS(finalized_);
    return num_dofs() - num_actuated_dofs();
  }

  int num_contacts() const {
    DRAKE_THROW_UNLESS(finalized_);
    return ssize(contact_pairs_);
  }

  int num_collisions() const {
    DRAKE_THROW_UNLESS(finalized_);
    return ssize(collision_pairs_);
  }

  std::unique_ptr<Scratch<T>> CreateScratch() const {
    DRAKE_THROW_UNLESS(finalized_);
    return std::make_unique<Scratch<T>>(multibody_tree_system_.get());
  }

  const VectorX<double>& GetActuationStiffnessVector() const {
    DRAKE_THROW_UNLESS(finalized_);
    return actuation_stiffness_;
  }

  const VectorX<double>& GetFrictionCoefficientVector() const {
    DRAKE_THROW_UNLESS(finalized_);
    return friction_coefficients_;
  }

  math::RigidTransform2d<T> CalcPoseInWorld(const Scratch<T>& scratch,
                                            BodyIndex body_index) const {
    DRAKE_THROW_UNLESS(finalized_);
    DRAKE_THROW_UNLESS(body_index < bodies_.size());
    return Convert3dTo2d(multibody_tree()
                             .get_body(body_index)
                             .EvalPoseInWorld(scratch.context()));
  }

  void CalcUnactuatedMassMatrix(const Scratch<T>& scratch,
                                EigenPtr<MatrixX<T>> M_unactuated) const {
    DRAKE_THROW_UNLESS(finalized_);
    DRAKE_THROW_UNLESS(M_unactuated != nullptr);
    DRAKE_THROW_UNLESS(M_unactuated->rows() == num_unactuated_dofs() &&
                       M_unactuated->cols() == num_unactuated_dofs());
    MatrixX<T> M(num_dofs(), num_dofs());
    multibody_tree().CalcMassMatrix(scratch.context(), &M);
    *M_unactuated =
        M.bottomRightCorner(num_unactuated_dofs(), num_unactuated_dofs());
  }

  VectorX<T> CalcGravityGeneralizedForces(const Scratch<T>& scratch) const {
    DRAKE_THROW_UNLESS(finalized_);
    return multibody_tree().CalcGravityGeneralizedForces(scratch.context());
  }

  void CalcContact(const Scratch<T>& scratch, EigenPtr<MatrixX<T>> J,
                   EigenPtr<VectorX<T>> phi, double smoothing) const {
    DRAKE_THROW_UNLESS(finalized_);
    DRAKE_THROW_UNLESS(J != nullptr);
    DRAKE_THROW_UNLESS(J->rows() == 2 * num_contacts() &&
                       J->cols() == num_dofs());
    DRAKE_THROW_UNLESS(phi != nullptr);
    DRAKE_THROW_UNLESS(phi->size() == num_contacts());
    DRAKE_THROW_UNLESS(smoothing >= 0);

    Matrix3X<T> Js_v_W(3, num_dofs());
    const auto& frame_W = multibody_tree().world_frame();

    for (int i = 0; i < num_contacts(); ++i) {
      const CollisionGeometry& geometry_A =
          collision_geometries_.at(contact_pairs_[i].first);
      const CollisionGeometry& geometry_B =
          collision_geometries_.at(contact_pairs_[i].second);
      const geometry::Shape2d& shape_A = *geometry_A.shape;
      const geometry::Shape2d& shape_B = *geometry_B.shape;
      const math::RigidTransform2d<T> X_WA =
          CalcPoseInWorld(scratch, geometry_A.body) * geometry_A.X_BG.cast<T>();
      const math::RigidTransform2d<T> X_WB =
          CalcPoseInWorld(scratch, geometry_B.body) * geometry_B.X_BG.cast<T>();
      const geometry::SignedDistancePair2d<T> res =
          CalcProximity2d(shape_A, X_WA, shape_B, X_WB, smoothing);

      (*phi)[i] = res.distance;

      const auto& frame_A =
          multibody_tree().get_body(geometry_A.body).body_frame();
      multibody_tree().CalcJacobianTranslationalVelocity(
          scratch.context(), JacobianWrtVariable::kV, frame_A, frame_A,
          Convert2dTo3d(res.p_ACa), frame_W, frame_W, &Js_v_W);
      const Matrix2<T> R_WCa = MakeFromUnitXVector(res.nhat_BA_W);
      J->template middleRows<2>(2 * i).leftCols(num_actuated_dofs()) =
          R_WCa.transpose() *
          Convert3dTo2d(Js_v_W).leftCols(num_actuated_dofs());

      const auto& frame_B =
          multibody_tree().get_body(geometry_B.body).body_frame();
      multibody_tree().CalcJacobianTranslationalVelocity(
          scratch.context(), JacobianWrtVariable::kV, frame_B, frame_B,
          Convert2dTo3d(res.p_BCb), frame_W, frame_W, &Js_v_W);
      /* The contact force has equal magnitude and opposite direction, so simply
       switch the sign of each axis. */
      const Matrix2<T> R_WCb = -R_WCa;
      J->template middleRows<2>(2 * i).rightCols(num_unactuated_dofs()) =
          R_WCb.transpose() *
          Convert3dTo2d(Js_v_W).rightCols(num_unactuated_dofs());
    }
  }

  void CalcCollision(const Scratch<T>& scratch, EigenPtr<MatrixX<T>> J,
                     EigenPtr<VectorX<T>> phi) const {
    DRAKE_THROW_UNLESS(finalized_);
    DRAKE_THROW_UNLESS(J != nullptr);
    DRAKE_THROW_UNLESS(J->rows() == num_collisions() &&
                       J->cols() == num_dofs());
    DRAKE_THROW_UNLESS(phi != nullptr);
    DRAKE_THROW_UNLESS(phi->size() == num_collisions());

    Matrix3X<T> Js_v_W(3, num_dofs());
    const auto& frame_W = multibody_tree().world_frame();

    J->setZero();
    for (int i = 0; i < num_collisions(); ++i) {
      const CollisionGeometry& geometry_A =
          collision_geometries_.at(collision_pairs_[i].first);
      const CollisionGeometry& geometry_B =
          collision_geometries_.at(collision_pairs_[i].second);
      const geometry::Shape2d& shape_A = *geometry_A.shape;
      const geometry::Shape2d& shape_B = *geometry_B.shape;
      const math::RigidTransform2d<T> X_WA =
          CalcPoseInWorld(scratch, geometry_A.body) * geometry_A.X_BG.cast<T>();
      const math::RigidTransform2d<T> X_WB =
          CalcPoseInWorld(scratch, geometry_B.body) * geometry_B.X_BG.cast<T>();
      const geometry::SignedDistancePair2d<T> res =
          CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

      (*phi)[i] = res.distance;
      const Matrix2<T> R_WC = MakeFromUnitXVector(res.nhat_BA_W);

      const auto& frame_A =
          multibody_tree().get_body(geometry_A.body).body_frame();
      multibody_tree().CalcJacobianTranslationalVelocity(
          scratch.context(), JacobianWrtVariable::kV, frame_A, frame_A,
          Convert2dTo3d(res.p_ACa), frame_W, frame_W, &Js_v_W);
      J->row(i) += R_WC.transpose().row(0) * Convert3dTo2d(Js_v_W);

      const auto& frame_B =
          multibody_tree().get_body(geometry_B.body).body_frame();
      multibody_tree().CalcJacobianTranslationalVelocity(
          scratch.context(), JacobianWrtVariable::kV, frame_B, frame_B,
          Convert2dTo3d(res.p_BCb), frame_W, frame_W, &Js_v_W);
      J->row(i) -= R_WC.transpose().row(0) * Convert3dTo2d(Js_v_W);
    }
  }

  void SetMeshcat(geometry::Meshcat* meshcat, std::string_view name_prefix,
                  double z_height) {
    draw_ = Draw{
        .meshcat = meshcat,
        .name_prefix = std::string(name_prefix),
        .z_height = z_height,
    };
  }

  systems::EventStatus DrawMeshcat(const Scratch<T>& scratch,
                                   double time_in_recording) const {
    if (draw_.meshcat == nullptr) return systems::EventStatus::DidNothing();

    geometry::Meshcat& meshcat = *draw_.meshcat;
    const std::string& name_prefix = draw_.name_prefix;
    const double h = draw_.z_height;

    for (BodyIndex body_index{0}; body_index < ssize(bodies_); ++body_index) {
      const RigidBody& body = *bodies_[body_index];
      const math::RigidTransform2d<T> pose =
          CalcPoseInWorld(scratch, body_index);
      const math::RigidTransform2d<double> X_WB(
          ExtractDoubleOrThrow(pose.angle()),
          ExtractDoubleOrThrow(pose.translation()));
      for (int i = 0; i < ssize(body.visual_geometries); ++i) {
        const GeometryId geom_id = body.visual_geometries[i];
        const VisualGeometry& geom = visual_geometries_.at(geom_id);
        const std::string name =
            geom.name ? fmt::format("/{}/{}", name_prefix, geom.name.value())
                      : fmt::format("/{}/body{}/geometry{}", name_prefix,
                                    body_index, i);
        const geometry::Rgba& rgba = geom.rgba;

        if (geom.shape_2d != nullptr) {
          const math::RigidTransform2d<double> X_WG =
              X_WB * std::get<math::RigidTransform2d<double>>(geom.X_BG);
          geom.shape_2d->Visit(overloaded{
              [&](const geometry::Circle& circle) {
                meshcat.SetObject(name, geometry::Cylinder(circle.radius(), h),
                                  rgba);
                meshcat.SetTransform(name, Convert2dTo3d(X_WG),
                                     time_in_recording);
              },
              [&](const geometry::Rectangle& rect) {
                meshcat.SetObject(
                    name, geometry::Box(rect.width(), rect.height(), h), rgba);
                meshcat.SetTransform(name, Convert2dTo3d(X_WG),
                                     time_in_recording);
              },
              [&](const geometry::Obround& obround) {
                meshcat.SetObject(
                    name + "/0",
                    geometry::Box(obround.length(), obround.radius() * 2, h),
                    rgba);
                meshcat.SetTransform(name + "/0", Convert2dTo3d(X_WG),
                                     time_in_recording);
                meshcat.SetObject(
                    name + "/1", geometry::Cylinder(obround.radius(), h), rgba);
                meshcat.SetTransform(
                    name + "/1",
                    Convert2dTo3d(X_WG * math::RigidTransform2d(Vector2<double>{
                                             obround.length() / 2, 0.0})),
                    time_in_recording);
                meshcat.SetObject(
                    name + "/2", geometry::Cylinder(obround.radius(), h), rgba);
                meshcat.SetTransform(
                    name + "/2",
                    Convert2dTo3d(X_WG * math::RigidTransform2d(Vector2<double>{
                                             -obround.length() / 2, 0.0})),
                    time_in_recording);
              },
          });
        } else if (geom.shape_3d != nullptr) {
          const math::RigidTransform<double> X_WG =
              Convert2dTo3d(X_WB) *
              std::get<math::RigidTransform<double>>(geom.X_BG);
          meshcat.SetObject(name, *geom.shape_3d, rgba);
          meshcat.SetTransform(name, X_WG, time_in_recording);
        } else {
          DRAKE_UNREACHABLE();
        }
      }
    }
    return systems::EventStatus::Succeeded();
  }

  template <typename U>
  void CopyFrom(const typename QuasidynamicPlanarPlant<U>::Impl& other) {
    // Copy finalized_ flag.
    finalized_ = other.finalized_;
    // Copy gravity_.
    gravity_ = other.gravity_;
    // Copy bodies_.
    bodies_ = other.bodies_;
    // Copy joints_.
    joints_ = other.joints_;
    // Copy collision_geometries_.
    collision_geometries_ = other.collision_geometries_;
    // Copy visual_geometries_.
    visual_geometries_ = other.visual_geometries_;
    // Copy multibody_tree_system_.
    if (other.multibody_tree_system_ != nullptr) {
      if constexpr (std::is_same_v<T, U>) {
        multibody_tree_system_ =
            dynamic_pointer_cast<internal::MultibodyTreeSystem<T>>(
                other.multibody_tree_system_->Clone());
      } else {
        multibody_tree_system_ =
            dynamic_pointer_cast<internal::MultibodyTreeSystem<T>>(
                other.multibody_tree_system_->template ToScalarType<T>());
      }
    } else {
      multibody_tree_system_ = nullptr;
    }
    // Copy actuation_stiffness_.
    actuation_stiffness_ = other.actuation_stiffness_;
    // Copy contact_pairs_.
    contact_pairs_ = other.contact_pairs_;
    // Copy friction_coefficients_.
    friction_coefficients_ = other.friction_coefficients_;
    // Copy collision_pairs_.
    collision_pairs_ = other.collision_pairs_;
    // Copy draw_.
    draw_ = other.draw_;
  }

  bool is_finalized() const { return finalized_; }

 private:
  /* Allows QuasidynamicPlanarPlant<U>::Impl to access private members of
   QuasidynamicPlanarPlant<T>::Impl for copying. */
  template <typename U>
  friend class QuasidynamicPlanarPlant;

  JointIndex AddJoint(Joint joint_in) {
    DRAKE_THROW_UNLESS(joint_in.body_P < bodies_.size());
    DRAKE_THROW_UNLESS(joint_in.body_C < bodies_.size());
    joints_.emplace_back(std::make_unique<Joint>(std::move(joint_in)));
    const Joint& joint = *joints_.back();
    const JointIndex joint_index(joints_.size() - 1);
    // Register joint as child in body P.
    bodies_[joint.body_P]->children.push_back(joint_index);
    // Register joint as parent in body C.
    if (bodies_[joint.body_C]->parent)
      throw std::runtime_error(
          fmt::format("Rigid body {} alread has a parent.", joint.body_C));
    bodies_[joint.body_C]->parent = joint_index;
    return joint_index;
  }

  std::tuple<std::vector<std::pair<GeometryId, GeometryId>>, VectorX<double>>
  MakeContactPairs() const {
    std::vector<GeometryId> actuated_geometries;
    std::vector<GeometryId> unactuated_geometries;

    /* Collect actuated and unactuated geometries. */
    const std::function<void(BodyIndex)> add_geometries =
        [this, &actuated_geometries, &unactuated_geometries,
         &add_geometries](BodyIndex body_index) {
          const RigidBody& body = *this->bodies_[body_index];
          const bool is_actuated =
              (body_index != 0)
                  ? this->joints_[body.parent.value()]->is_actuated()
                  : true;  // World body is treated as actuated because its
                           // attached geometries such as floor or wall makes
                           // contact with unactuated bodies.
          std::vector<GeometryId>& geometries =
              is_actuated ? actuated_geometries : unactuated_geometries;
          geometries.insert(geometries.end(),
                            body.collision_geometries.cbegin(),
                            body.collision_geometries.cend());

          for (JointIndex joint_index : body.children) {
            const Joint& joint = *this->joints_[joint_index];
            add_geometries(joint.body_C);
          }
        };
    add_geometries(BodyIndex(0));

    /* Aggregate pairs. */
    std::vector<std::pair<GeometryId, GeometryId>> pairs;
    AppendCartesianProduct(actuated_geometries, unactuated_geometries, &pairs);

    /* Compute effective friction coefficients. */
    VectorX<double> mu(pairs.size());
    for (int i = 0; i < ssize(pairs); ++i) {
      const double mu1 =
          collision_geometries_.at(pairs[i].first).friction_coefficient;
      const double mu2 =
          collision_geometries_.at(pairs[i].second).friction_coefficient;
      mu[i] = CalcEffectiveFrictionCoefficient(mu1, mu2);
    }

    return {pairs, mu};
  }

  std::vector<std::pair<GeometryId, GeometryId>> MakeCollisionPairs() const {
    std::vector<std::vector<GeometryId>> actuated_branches;

    /* Root is its own branch. */
    actuated_branches.emplace_back(this->bodies_[0]->collision_geometries);

    /* For each branch that is actuated, add its geometries. */
    const std::function<void(BodyIndex, std::vector<GeometryId>*)>
        add_geometries =
            [this, &add_geometries](BodyIndex body_index,
                                    std::vector<GeometryId>* geometries) {
              const RigidBody& body = *this->bodies_[body_index];
              geometries->insert(geometries->end(),
                                 body.collision_geometries.cbegin(),
                                 body.collision_geometries.cend());
              for (JointIndex joint_index : body.children) {
                const Joint& joint = *this->joints_[joint_index];
                add_geometries(joint.body_C, geometries);
              }
            };
    for (JointIndex joint_index : this->bodies_[0]->children) {
      const Joint& joint = *this->joints_[joint_index];
      if (joint.is_actuated()) {
        actuated_branches.emplace_back();
        add_geometries(joint.body_C, &actuated_branches.back());
      }
    }

    /* Aggregate pairs. */
    std::vector<std::pair<GeometryId, GeometryId>> pairs;
    int i = 0;
    for (int j = i + 1; j < ssize(actuated_branches); ++j) {
      std::vector<GeometryId> actuated_branches_j;
      if (actuated_branches[j].size() > 1) {
        actuated_branches_j.insert(actuated_branches_j.end(),
                                   actuated_branches[j].cbegin() + 1,
                                   actuated_branches[j].cend());
      }
      AppendCartesianProduct(actuated_branches[i], actuated_branches_j, &pairs,
                             /* skip_first_cartesian_pair */ false);
    }
    for (i = 1; i < ssize(actuated_branches); ++i) {
      for (int j = i + 1; j < ssize(actuated_branches); ++j) {
        AppendCartesianProduct(actuated_branches[i], actuated_branches[j],
                               &pairs, /* skip_first_cartesian_pair */ true);
      }
    }

    return pairs;
  }

  std::unique_ptr<internal::MultibodyTreeSystem<T>> MakeMultibodyTreeSystem()
      const {
    std::unique_ptr<internal::MultibodyTree<T>> tree =
        std::make_unique<internal::MultibodyTree<T>>();

    /* Add bodies to MultibodyTree. */
    for (BodyIndex body_index{1}; body_index < ssize(bodies_); ++body_index) {
      const RigidBody& body = *bodies_[body_index];

      SpatialInertia<double> I_BBcm_B = SpatialInertia<double>::NaN();
      if (body.mass != 0) {
        const Vector3<double> p_BBcm_B = Convert2dTo3d(body.p_BBcm_B);
        const RotationalInertia<double> I_BBcm(/* Ixx */ body.I_BBcm,
                                               /* Iyy */ body.I_BBcm,
                                               /* Ixx */ body.I_BBcm);
        I_BBcm_B = SpatialInertia<double>::MakeFromCentralInertia(
            body.mass, p_BBcm_B, I_BBcm);
      } else {
        I_BBcm_B = SpatialInertia<double>::Zero();
      }

      const auto& res =
          tree->AddRigidBody(fmt::format("RigidBody{}", body_index), I_BBcm_B);
      DRAKE_DEMAND(res.index() == body_index);
    }

    /* Order joints. */
    std::vector<JointIndex> joint_indexes;
    joint_indexes.reserve(joints_.size());
    const std::function<void(const RigidBody&)> dfs =
        [this, &joint_indexes, &dfs](const RigidBody& body) {
          for (JointIndex joint_index : body.children) {
            const Joint& joint = *this->joints_[joint_index];
            joint_indexes.push_back(joint_index);
            dfs(*this->bodies_[joint.body_C]);
          }
        };

    RigidBody body0 = *bodies_[0];
    std::sort(body0.children.begin(), body0.children.end(),
              [this](JointIndex i1, JointIndex i2) {
                // Actuated ones appear first.
                return joints_[i1]->is_actuated() > joints_[i2]->is_actuated();
              });
    dfs(body0);
    DRAKE_DEMAND(joint_indexes.size() == joints_.size());

    /* Add joints to MultibodyTree. */
    int total_dofs = 0;
    for (JointIndex joint_index : joint_indexes) {
      const Joint& joint = *joints_[joint_index];
      const std::string name = fmt::format("Joint{}", joint_index);

      const auto& parent = tree->get_body(joint.body_P);
      const math::RigidTransform<double> X_PJp = Convert2dTo3d(joint.X_PJp);
      const auto& child = tree->get_body(joint.body_C);
      const math::RigidTransform<double> X_CJc = Convert2dTo3d(joint.X_CJc);

      int dof = -1;
      switch (joint.type) {
        case Joint::Type::Revolute: {
          dof = 1;
          tree->template AddJoint<RevoluteJoint>(
              name, parent, X_PJp, child, X_CJc, Vector3<double>::UnitZ());
          break;
        }
        case Joint::Type::Prismatic: {
          dof = 1;
          tree->template AddJoint<PrismaticJoint>(
              name, parent, X_PJp, child, X_CJc, Vector3<double>::UnitX());
          break;
        }
        case Joint::Type::Cartesian: {
          dof = 2;
          const auto& intermediate = tree->AddRigidBody(
              fmt::format("IntermediateRigidBody{}", joint_index),
              SpatialInertia<double>::Zero());
          tree->template AddJoint<PrismaticJoint>(
              name, parent, X_PJp, intermediate, {}, Vector3<double>::UnitX());
          tree->template AddJoint<PrismaticJoint>(name + "_2", intermediate, {},
                                                  child, X_CJc,
                                                  Vector3<double>::UnitY());
          break;
        }
        case Joint::Type::Planar: {
          dof = 3;
          tree->template AddJoint<PlanarJoint>(
              name, parent, X_PJp, child, X_CJc,
              /* damping */ Vector3<double>::Zero());
          break;
        }
        default:
          DRAKE_UNREACHABLE();
      }

      total_dofs += dof;
      if (joint.is_actuated())
        DRAKE_DEMAND(joint.actuation_stiffness->size() == dof);
    }

    /* Finalize the MultibodyTree. */
    tree->mutable_gravity_field().set_gravity_vector(Convert2dTo3d(gravity_));
    tree->Finalize();
    DRAKE_DEMAND(tree->num_positions() == total_dofs);
    DRAKE_DEMAND(tree->IsVelocityEqualToQDot());

    return std::make_unique<internal::MultibodyTreeSystem<T>>(std::move(tree));
  }

  VectorX<double> MakeActuationStiffness() const {
    VectorX<double> storage =
        VectorX<double>::Constant(multibody_tree().num_positions(), NAN);
    int num_actuated_dofs = 0;

    /* Fill in the actuation stiffness. */
    std::regex re(R"(Joint(\d+))");
    for (int i = 0; i < multibody_tree().num_joints(); ++i) {
      const auto& res = multibody_tree().get_joint(JointIndex(i));
      const int dof_begin = res.position_start();

      std::smatch match;
      if (!std::regex_match(res.name(), match, re)) continue;
      JointIndex joint_index(std::stoi(match[1]));
      const Joint& joint = *joints_[joint_index];

      if (!joint.is_actuated()) continue;
      const VectorX<double>& stiffness = joint.actuation_stiffness.value();
      storage.segment(dof_begin, stiffness.size()) = stiffness;
      num_actuated_dofs += stiffness.size();
    }

    /* Check that the actuated DoFs come before the unactuated DoFs. */
    for (int i = 0; i < multibody_tree().num_joints(); ++i) {
      const auto& res = multibody_tree().get_joint(JointIndex(i));
      const int dof_begin = res.position_start();
      const int dof_end = res.position_start() + res.num_positions();

      std::smatch match;
      DRAKE_DEMAND(std::regex_search(res.name(), match, re));
      JointIndex joint_index(std::stoi(match[1]));
      const Joint& joint = *joints_[joint_index];

      if (joint.is_actuated())
        DRAKE_DEMAND(dof_end <= num_actuated_dofs);
      else
        DRAKE_DEMAND(dof_begin >= num_actuated_dofs);
    }

    const VectorX<double> actuation_stiffness = storage.head(num_actuated_dofs);
    DRAKE_DEMAND((actuation_stiffness.array() != NAN).all());
    return actuation_stiffness;
  }

  const internal::MultibodyTree<T>& multibody_tree() const {
    DRAKE_DEMAND(multibody_tree_system_ != nullptr);
    return internal::GetInternalTree(*multibody_tree_system_);
  }

  bool finalized_{false};
  Vector2<double> gravity_{Vector2<double>::Zero()};
  std::vector<copyable_unique_ptr<RigidBody>> bodies_;
  std::vector<copyable_unique_ptr<Joint>> joints_;
  std::unordered_map<GeometryId, CollisionGeometry> collision_geometries_;
  std::unordered_map<GeometryId, VisualGeometry> visual_geometries_;
  copyable_unique_ptr<internal::MultibodyTreeSystem<T>> multibody_tree_system_;
  VectorX<double> actuation_stiffness_;
  std::vector<std::pair<GeometryId, GeometryId>> contact_pairs_;
  VectorX<double> friction_coefficients_;
  std::vector<std::pair<GeometryId, GeometryId>> collision_pairs_;
  Draw draw_;
};

template <typename T>
void QuasidynamicPlanarPlant<T>::ImplDeleter::operator()(Impl* impl) {
  delete impl;
}

template <typename T>
QuasidynamicPlanarPlant<T>::QuasidynamicPlanarPlant(double dt,
                                                    double dynamics_smoothing,
                                                    double geometry_smoothing)
    : QuasidynamicDifferentiableContactModel<T>(
          systems::SystemTypeTag<QuasidynamicPlanarPlant>{},
          dynamics_smoothing),
      dt_(dt),
      impl_{std::unique_ptr<Impl, ImplDeleter>(new Impl())} {
  DRAKE_THROW_UNLESS(dt > 0);
  DRAKE_THROW_UNLESS(dynamics_smoothing >= 0);
  DRAKE_THROW_UNLESS(geometry_smoothing >= 0);

  geometry_smoothing_param_index_ =
      systems::NumericParameterIndex(this->DeclareNumericParameter(
          systems::BasicVector{T(geometry_smoothing)}));
}

template <typename T>
template <typename U>
QuasidynamicPlanarPlant<T>::QuasidynamicPlanarPlant(
    const QuasidynamicPlanarPlant<U>& other)
    : QuasidynamicPlanarPlant(
          other.dt_,
          other.get_dynamics_smoothing(*other.CreateDefaultContext()),
          other.get_geometry_smoothing(*other.CreateDefaultContext())) {
  DRAKE_DEMAND(other.impl_ != nullptr);
  impl_->template CopyFrom<U>(*other.impl_);
  if (impl_->is_finalized()) Finalize();
}

template <typename T>
BodyIndex QuasidynamicPlanarPlant<T>::AddRigidBody(
    double mass, const Vector2<double>& p_BBcm_B, double I_BBcm) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddRigidBody(mass, p_BBcm_B, I_BBcm);
}

template <typename T>
BodyIndex QuasidynamicPlanarPlant<T>::AddRigidBody() {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddRigidBody();
}

template <typename T>
BodyIndex QuasidynamicPlanarPlant<T>::world_body() const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->world_body();
}

template <typename T>
JointIndex QuasidynamicPlanarPlant<T>::AddRevoluteJoint(
    BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
    BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
    std::optional<double> actuation_stiffness) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddRevoluteJoint(body_P, X_PJp, body_C, X_CJc,
                                 actuation_stiffness);
}

template <typename T>
JointIndex QuasidynamicPlanarPlant<T>::AddPrismaticJoint(
    BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
    BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
    std::optional<double> actuation_stiffness) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddPrismaticJoint(body_P, X_PJp, body_C, X_CJc,
                                  actuation_stiffness);
}

template <typename T>
JointIndex QuasidynamicPlanarPlant<T>::AddCartesianJoint(
    BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
    BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
    const std::optional<Vector2<double>>& actuation_stiffness) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddCartesianJoint(body_P, X_PJp, body_C, X_CJc,
                                  actuation_stiffness);
}

template <typename T>
JointIndex QuasidynamicPlanarPlant<T>::AddPlanarJoint(
    BodyIndex body_P, const math::RigidTransform2d<double>& X_PJp,
    BodyIndex body_C, const math::RigidTransform2d<double>& X_CJc,
    const std::optional<Vector3<double>>& actuation_stiffness) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->AddPlanarJoint(body_P, X_PJp, body_C, X_CJc,
                               actuation_stiffness);
}

template <typename T>
GeometryId QuasidynamicPlanarPlant<T>::RegisterCollisionGeometry(
    BodyIndex body, const math::RigidTransform2d<double>& X_BG,
    const geometry::Shape2d& shape, double friction_coefficient) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->RegisterCollisionGeometry(body, X_BG, shape,
                                          friction_coefficient);
}

template <typename T>
GeometryId QuasidynamicPlanarPlant<T>::RegisterVisualGeometry(
    BodyIndex body, const math::RigidTransform2d<double>& X_BG,
    const geometry::Shape2d& shape, const geometry::Rgba& rgba,
    std::optional<std::string_view> name) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->RegisterVisualGeometry(body, X_BG, shape, rgba, name);
}

template <typename T>
GeometryId QuasidynamicPlanarPlant<T>::RegisterVisualGeometry(
    BodyIndex body, const math::RigidTransform<double>& X_BG,
    const geometry::Shape& shape, const geometry::Rgba& rgba,
    std::optional<std::string_view> name) {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->RegisterVisualGeometry(body, X_BG, shape, rgba, name);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::SetGravityVector(const Vector2<double>& g_W) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->SetGravityVector(g_W);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::Finalize() {
  DRAKE_DEMAND(impl_ != nullptr);
  if (!impl_->is_finalized()) impl_->Finalize();

  const int num_dofs = impl_->num_dofs();
  const int num_actuated_dofs = impl_->num_actuated_dofs();
  const int num_unactuated_dofs = impl_->num_unactuated_dofs();
  const int num_contacts = impl_->num_contacts();
  const int num_collisions = impl_->num_collisions();

  this->DeclareStateAndPorts(
      /* x */ VectorX<T>::Zero(num_dofs),
      /* num_inputs */ num_actuated_dofs,
      /* num_contacts */ num_contacts,
      /* contact_dim */ 2,
      /* dt */ dt_);

  M_unactuated_cache_index_ =
      this->DeclareCacheEntry(
              "M_unactuated",
              MatrixX<T>(
                  MatrixX<T>::Zero(num_unactuated_dofs, num_unactuated_dofs)),
              &QuasidynamicPlanarPlant<T>::CalcUnactuatedMassMatrix,
              {this->all_state_ticket()})
          .cache_index();

  P_quasidynamic_cache_index_ =
      this->DeclareCacheEntry(
              "P_quasidynamic",
              MatrixX<T>(MatrixX<T>::Zero(num_dofs, num_dofs)),
              &QuasidynamicPlanarPlant<T>::CalcQuasidynamicPMatrix,
              {this->all_state_ticket()})
          .cache_index();

  b_quasidynamic_cache_index_ =
      this->DeclareCacheEntry(
              "b_quasidynamic", VectorX<T>(VectorX<T>::Zero(num_dofs)),
              &QuasidynamicPlanarPlant<T>::CalcQuasidynamicBVector,
              {this->all_state_ticket(), this->all_input_ports_ticket()})
          .cache_index();

  contact_cache_index_ =
      this->DeclareCacheEntry(
              "contact",
              std::make_tuple(
                  MatrixX<T>(MatrixX<T>::Zero(2 * num_contacts, num_dofs)),
                  VectorX<T>(VectorX<T>::Zero(num_contacts))),
              &QuasidynamicPlanarPlant<T>::CalcContact,
              {this->all_state_ticket(),
               this->numeric_parameter_ticket(geometry_smoothing_param_index_)})
          .cache_index();

  collision_cache_index_ =
      this->DeclareCacheEntry(
              "collision",
              std::make_tuple(
                  MatrixX<T>(MatrixX<T>::Zero(num_collisions, num_dofs)),
                  VectorX<T>(VectorX<T>::Zero(num_collisions))),
              &QuasidynamicPlanarPlant<T>::CalcCollision,
              {this->all_state_ticket()})
          .cache_index();

  std::unique_ptr<Scratch<T>> scratch = impl_->CreateScratch();
  scratch_cache_index_ =
      this->DeclareCacheEntry(
              "scratch",
              systems::ValueProducer(
                  *scratch,
                  std::function<void(const systems::Context<T>&, Scratch<T>*)>(
                      [this](const systems::Context<T>& context,
                             Scratch<T>* scratch_out) {
                        scratch_out->SetState(this->GetState(context));
                      })),
              {this->all_state_ticket()})
          .cache_index();

  this->DeclarePerStepPublishEvent(&QuasidynamicPlanarPlant<T>::DrawMeshcat);
  this->DeclareForcedPublishEvent(&QuasidynamicPlanarPlant<T>::DrawMeshcat);
}

template <typename T>
double QuasidynamicPlanarPlant<T>::get_geometry_smoothing(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return ExtractDoubleOrThrow(
      context.get_numeric_parameter(geometry_smoothing_param_index_)
          .value()[0]);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::set_geometry_smoothing(
    systems::Context<T>* context, double geometry_smoothing) const {
  this->ValidateContext(context);
  DRAKE_THROW_UNLESS(geometry_smoothing >= 0);
  context->get_mutable_numeric_parameter(geometry_smoothing_param_index_)
      .get_mutable_value()[0] = geometry_smoothing;
}

template <typename T>
static const Scratch<T>& EvalScratch(const QuasidynamicPlanarPlant<T>& system,
                                     const systems::Context<T>& context,
                                     systems::CacheIndex scratch_cache_index) {
  return system.get_cache_entry(scratch_cache_index)
      .template Eval<Scratch<T>>(context);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::CalcUnactuatedMassMatrix(
    const systems::Context<T>& context, MatrixX<T>* M_unactuated) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  impl_->CalcUnactuatedMassMatrix(scratch, M_unactuated);
}

template <typename T>
const MatrixX<T>& QuasidynamicPlanarPlant<T>::EvalUnactuatedMassMatrix(
    const systems::Context<T>& context) const {
  return this->get_cache_entry(M_unactuated_cache_index_)
      .template Eval<MatrixX<T>>(context);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::CalcQuasidynamicPMatrix(
    const systems::Context<T>& context, MatrixX<T>* P_quasidynamic) const {
  DRAKE_DEMAND(impl_ != nullptr);

  const int num_actuated_dofs = impl_->num_actuated_dofs();
  const int num_unactuated_dofs = impl_->num_unactuated_dofs();
  const MatrixX<T>& M_unactuated = EvalUnactuatedMassMatrix(context);

  P_quasidynamic->setZero();
  P_quasidynamic->bottomRightCorner(num_unactuated_dofs, num_unactuated_dofs) =
      M_unactuated / (dt_ * dt_);
  P_quasidynamic->diagonal().head(num_actuated_dofs) =
      impl_->GetActuationStiffnessVector();
}

template <typename T>
const MatrixX<T>& QuasidynamicPlanarPlant<T>::EvalQuasidynamicPMatrix(
    const systems::Context<T>& context) const {
  return this->get_cache_entry(P_quasidynamic_cache_index_)
      .template Eval<MatrixX<T>>(context);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::CalcQuasidynamicBVector(
    const systems::Context<T>& context, VectorX<T>* b_quasidynamic) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const int num_actuated_dofs = impl_->num_actuated_dofs();
  const int num_unactuated_dofs = impl_->num_unactuated_dofs();
  const VectorX<T>& u = this->get_actuation_input_port().Eval(context);
  const VectorX<T>& x = this->GetState(context);
  const VectorX<T>& k = impl_->GetActuationStiffnessVector();
  const MatrixX<T>& Mo = EvalUnactuatedMassMatrix(context);

  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  *b_quasidynamic = -impl_->CalcGravityGeneralizedForces(scratch);
  b_quasidynamic->head(num_actuated_dofs) -= k.cwiseProduct(u);
  b_quasidynamic->tail(num_unactuated_dofs) -=
      Mo * x.tail(num_unactuated_dofs) / (dt_ * dt_);
}

template <typename T>
const VectorX<T>& QuasidynamicPlanarPlant<T>::EvalQuasidynamicBVector(
    const systems::Context<T>& context) const {
  return this->get_cache_entry(b_quasidynamic_cache_index_)
      .template Eval<VectorX<T>>(context);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::CalcContact(
    const systems::Context<T>& context,
    std::tuple<MatrixX<T>, VectorX<T>>* contact) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  impl_->CalcContact(scratch, &std::get<0>(*contact), &std::get<1>(*contact),
                     get_geometry_smoothing(context));
}

template <typename T>
const MatrixX<T>& QuasidynamicPlanarPlant<T>::EvalContactJacobian(
    const systems::Context<T>& context) const {
  return std::get<0>(
      this->get_cache_entry(contact_cache_index_)
          .template Eval<std::tuple<MatrixX<T>, VectorX<T>>>(context));
}

template <typename T>
const VectorX<T>& QuasidynamicPlanarPlant<T>::EvalContactSignedDistance(
    const systems::Context<T>& context) const {
  return std::get<1>(
      this->get_cache_entry(contact_cache_index_)
          .template Eval<std::tuple<MatrixX<T>, VectorX<T>>>(context));
}

template <typename T>
const VectorX<double>&
QuasidynamicPlanarPlant<T>::EvalContactFrictionCoefficient(
    const systems::Context<T>&) const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->GetFrictionCoefficientVector();
}

template <typename T>
math::RigidTransform2d<T> QuasidynamicPlanarPlant<T>::EvalPoseInWorld(
    const systems::Context<T>& context, BodyIndex body) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  return impl_->CalcPoseInWorld(scratch, body);
}

template <typename T>
void QuasidynamicPlanarPlant<T>::CalcCollision(
    const systems::Context<T>& context,
    std::tuple<MatrixX<T>, VectorX<T>>* collision) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  impl_->CalcCollision(scratch, &std::get<0>(*collision),
                       &std::get<1>(*collision));
}

template <typename T>
const MatrixX<T>& QuasidynamicPlanarPlant<T>::EvalCollisionJacobian(
    const systems::Context<T>& context) const {
  return std::get<0>(
      this->get_cache_entry(collision_cache_index_)
          .template Eval<std::tuple<MatrixX<T>, VectorX<T>>>(context));
}

template <typename T>
const VectorX<T>& QuasidynamicPlanarPlant<T>::EvalCollisionSignedDistance(
    const systems::Context<T>& context) const {
  return std::get<1>(
      this->get_cache_entry(collision_cache_index_)
          .template Eval<std::tuple<MatrixX<T>, VectorX<T>>>(context));
}

template <typename T>
void QuasidynamicPlanarPlant<T>::SetMeshcat(geometry::Meshcat* meshcat,
                                            std::string_view name_prefix,
                                            double z_height) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->SetMeshcat(meshcat, name_prefix, z_height);
}

template <typename T>
systems::EventStatus QuasidynamicPlanarPlant<T>::DrawMeshcat(
    const systems::Context<T>& context) const {
  DRAKE_DEMAND(impl_ != nullptr);
  const Scratch<T>& scratch = EvalScratch(*this, context, scratch_cache_index_);
  const double t = ExtractDoubleOrThrow(context.get_time());
  return impl_->DrawMeshcat(scratch, t);
}

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::quasidynamic::QuasidynamicPlanarPlant);
