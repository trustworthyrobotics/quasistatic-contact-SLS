#include "drake/geometry/proximity/calc_proximity.h"

#include <vector>

#include "drake/common/unused.h"
#include "drake/solvers/solve_conic_program.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Matrix;
using Eigen::Vector;

/* Solves the QP:
      minimize    ½ xᵀ P x + qᵀ x
      subject to  A x ≼ b
   to complementary slackness
      x ⊙ z = κ 1
   where z is the dual variable and κ≥0 is a relaxation parameter.
   The returned value is x - (∂x/∂κ) 2κ. */
template <typename T, int XSize, int ZSize>
Vector<T, XSize> SolveQP(const Matrix<T, XSize, XSize>& P,
                         const Vector<T, XSize>& q,
                         const Matrix<T, ZSize, XSize>& A,
                         const Vector<T, ZSize>& b, double kappa) {
  const std::vector<solvers::Cone> cones{solvers::NonnegativeCone(b.size())};
  VectorX<T> x, dx_dkappa;
  if (kappa == 0) {
    std::tie(x, std::ignore) =
        solvers::SolveConicProgram<T>(P, q, A, b, cones, kappa);
    return x;
  } else {
    std::tie(x, std::ignore) =
        solvers::SolveConicProgram<T>(P, q, A, b, cones, kappa, &dx_dkappa);
    return x - dx_dkappa * (2 * kappa);
  }
}

#define DEFINE_SYMMETRIC_CALC_PROXIMITY_2D_IMPL(TypeA, TypeB)      \
  template <typename T>                                            \
  SignedDistancePair2d<T> CalcProximity2dImpl(                     \
      const TypeA& shape_A, const math::RigidTransform2d<T>& X_WA, \
      const TypeB& shape_B, const math::RigidTransform2d<T>& X_WB, \
      double kappa) {                                              \
    SignedDistancePair2d<T> result =                               \
        CalcProximity2dImpl(shape_B, X_WB, shape_A, X_WA, kappa);  \
    result.SwapAAndB();                                            \
    return result;                                                 \
  }

/* An H-Polygon defines the set { x ∈ ℝ² | A x ≤ b }. */
class HPolygon {
 public:
  // NOLINTNEXTLINE(runtime/explicit): Conversion from Rectangle to HPolygon.
  HPolygon(const Rectangle& rectangle) : A_(4, 2), b_(4) {
    // clang-format off
    A_ << 1,  0,
         -1,  0,
          0,  1,
          0, -1;
    // clang-format on
    b_ << rectangle.width() / 2, rectangle.width() / 2, rectangle.height() / 2,
        rectangle.height() / 2;
  }

  const Eigen::MatrixX2<double>& A() const { return A_; }
  const VectorX<double>& b() const { return b_; }

 private:
  Eigen::MatrixX2<double> A_;
  VectorX<double> b_;
};

/* Computes proximity between circle and circle. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const Circle& shape_A, const math::RigidTransform2d<T>& X_WA,
    const Circle& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  unused(kappa);
  const Vector2<T> nhat_BA_W =
      (X_WA.translation() - X_WB.translation()).normalized();
  const Vector2<T> p_WCa = X_WA.translation() - nhat_BA_W * shape_A.radius();
  const Vector2<T> p_WCb = X_WB.translation() + nhat_BA_W * shape_B.radius();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between circle and obround. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const Circle& shape_A, const math::RigidTransform2d<T>& X_WA,
    const Obround& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  const Vector2<T> p_WA = X_WA.translation();
  const Vector2<T> p_WB1 = X_WB * Vector2<T>{-shape_B.length() / 2, 0.0};
  const Vector2<T> p_WB2 = X_WB * Vector2<T>{+shape_B.length() / 2, 0.0};

  const Matrix<T, 1, 1> Q = (p_WB2 - p_WB1).transpose() * (p_WB2 - p_WB1);
  const Vector<T, 1> q = (p_WB2 - p_WB1).transpose() * (p_WB1 - p_WA);
  const Matrix<T, 2, 1> A{1, -1};
  const Vector2<T> b{1, 0};

  unused(kappa);  // Sommthing is not needed since obround is smooth.
  const Vector<T, 1> beta = SolveQP(Q, q, A, b, 0);
  const Vector2<T> p_WB =
      p_WB1 + (p_WB2 - p_WB1) * beta;  // Point on line segment.
  const Vector2<T> nhat_BA_W = (p_WA - p_WB).normalized();
  const Vector2<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();
  const Vector2<T> p_WCb = p_WB + nhat_BA_W * shape_B.radius();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between obround and circle. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_2D_IMPL(Obround, Circle)

/* Computes proximity between circle and H-polygon. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const Circle& shape_A, const math::RigidTransform2d<T>& X_WA,
    const HPolygon& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  const Vector2<T> p_WA = X_WA.translation();
  const Matrix2<T> R_BW = X_WB.rotation_matrix().transpose();

  const Matrix2<T> P = Matrix2<T>::Identity();
  const Vector2<T> q = -p_WA;
  const Eigen::MatrixX2<T> A = shape_B.A() * R_BW;
  const VectorX<T> b = shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector2<T> p_WCb = SolveQP(P, q, A, b, kappa);
  const Vector2<T> nhat_BA_W = (p_WA - p_WCb).normalized();
  const Vector2<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between H-polygon and circle. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_2D_IMPL(HPolygon, Circle)

/* Computes proximity between obround and obround. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const Obround& shape_A, const math::RigidTransform2d<T>& X_WA,
    const Obround& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  const Vector2<T> p_WA1 = X_WA * Vector2<T>{-shape_A.length() / 2, 0.0};
  const Vector2<T> p_WA2 = X_WA * Vector2<T>{+shape_A.length() / 2, 0.0};
  const Vector2<T> p_WB1 = X_WB * Vector2<T>{-shape_B.length() / 2, 0.0};
  const Vector2<T> p_WB2 = X_WB * Vector2<T>{+shape_B.length() / 2, 0.0};

  Matrix2<T> S;
  S << p_WA2 - p_WA1, p_WB1 - p_WB2;

  const Matrix2<T> P = S.transpose() * S;
  const Vector2<T> q = S.transpose() * (p_WA1 - p_WB1);
  Matrix<T, 4, 2> A = Matrix<T, 4, 2>::Zero();
  A.template topLeftCorner<2, 1>() = Vector2<T>{1, -1};
  A.template bottomRightCorner<2, 1>() = Vector2<T>{1, -1};
  const Vector<T, 4> b{1, 0, 1, 0};

  unused(kappa);  // Sommthing is not needed since obround is smooth.
  const Vector2<T> primal = SolveQP(P, q, A, b, 0);
  const T& alpha = primal[0];
  const T& beta = primal[1];
  // Point on line segment.
  const Vector2<T> p_WA = p_WA1 + (p_WA2 - p_WA1) * alpha;
  const Vector2<T> p_WB = p_WB1 + (p_WB2 - p_WB1) * beta;

  const Vector2<T> nhat_BA_W = (p_WA - p_WB).normalized();
  const Vector2<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();
  const Vector2<T> p_WCb = p_WB + nhat_BA_W * shape_B.radius();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between obround and H-polygon. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const Obround& shape_A, const math::RigidTransform2d<T>& X_WA,
    const HPolygon& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  const Vector2<T> p_WA1 = X_WA * Vector2<T>{-shape_A.length() / 2, 0.0};
  const Vector2<T> p_WA2 = X_WA * Vector2<T>{+shape_A.length() / 2, 0.0};
  const Matrix2<T> R_BW = X_WB.rotation_matrix().transpose();

  Matrix<T, 2, 3> S;
  S << p_WA2 - p_WA1, -Matrix2<T>::Identity();

  const Matrix3<T> P = S.transpose() * S;
  const Vector3<T> q = S.transpose() * p_WA1;

  const int num_constraints = 2 + shape_B.b().size();
  Eigen::MatrixX3<T> A = Eigen::MatrixX3<T>::Zero(num_constraints, 3);
  A.template topLeftCorner<2, 1>() = Vector2<T>{1, -1};
  A.bottomRightCorner(num_constraints - 2, 2) = shape_B.A() * R_BW;
  VectorX<T> b(num_constraints);
  b << 1, 0, shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector3<T> primal = SolveQP(P, q, A, b, kappa);
  const T& alpha = primal[0];
  const Vector2<T> p_WCb = primal.template tail<2>();

  const Vector2<T> p_WA =
      p_WA1 + (p_WA2 - p_WA1) * alpha;  // Point on line segment.
  const Vector2<T> nhat_BA_W = (p_WA - p_WCb).normalized();
  const Vector2<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between H-polygon and obround. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_2D_IMPL(HPolygon, Obround)

/* Computes proximity between H-polygon and H-polygon. */
template <typename T>
SignedDistancePair2d<T> CalcProximity2dImpl(
    const HPolygon& shape_A, const math::RigidTransform2d<T>& X_WA,
    const HPolygon& shape_B, const math::RigidTransform2d<T>& X_WB,
    double kappa) {
  const Matrix2<T> R_AW = X_WA.rotation_matrix().transpose();
  const Matrix2<T> R_BW = X_WB.rotation_matrix().transpose();

  Matrix<T, 2, 4> S;
  S << Matrix2<T>::Identity(), -Matrix2<T>::Identity();

  const Matrix4<T> P = S.transpose() * S;
  const Vector4<T> q = Vector4<T>::Zero();

  const int num_constraints_A = shape_A.b().size();
  const int num_constraints_B = shape_B.b().size();
  Eigen::MatrixX4<T> A =
      Eigen::MatrixX4<T>::Zero(num_constraints_A + num_constraints_B, 4);
  A.topLeftCorner(num_constraints_A, 2) = shape_A.A() * R_AW;
  A.bottomRightCorner(num_constraints_B, 2) = shape_B.A() * R_BW;
  VectorX<T> b(num_constraints_A + num_constraints_B);
  b << shape_A.b() + shape_A.A() * R_AW * X_WA.translation(),
      shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector4<T> primal = SolveQP(P, q, A, b, kappa);
  const Vector2<T> p_WCa = primal.template head<2>();
  const Vector2<T> p_WCb = primal.template tail<2>();
  // TODO(wei-chen): Handle computing normal when p_WCa is close to p_WCb.
  const Vector2<T> nhat_BA_W = (p_WCa - p_WCb).normalized();

  SignedDistancePair2d<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

#undef DEFINE_SYMMETRIC_CALC_PROXIMITY_2D_IMPL

#define DEFINE_SYMMETRIC_CALC_PROXIMITY_IMPL(TypeA, TypeB)       \
  template <typename T>                                          \
  SignedDistancePair<T> CalcProximityImpl(                       \
      const TypeA& shape_A, const math::RigidTransform<T>& X_WA, \
      const TypeB& shape_B, const math::RigidTransform<T>& X_WB, \
      double kappa) {                                            \
    SignedDistancePair<T> result =                               \
        CalcProximityImpl(shape_B, X_WB, shape_A, X_WA, kappa);  \
    result.SwapAAndB();                                          \
    return result;                                               \
  }

/* An H-Polyhedron defines the set { x ∈ ℝ² | A x ≤ b }. */
class HPolyhedron {
 public:
  // NOLINTNEXTLINE(runtime/explicit): Conversion from Box to HPolygon.
  HPolyhedron(const Box& box) : A_(6, 3), b_(6) {
    // clang-format off
    A_ << 1,  0,  0,
         -1,  0,  0,
          0,  1,  0,
          0, -1,  0,
          0,  0,  1,
          0,  0, -1;
    // clang-format on
    b_ << box.width() / 2, box.width() / 2, box.depth() / 2, box.depth() / 2,
        box.height() / 2, box.height() / 2;
  }

  // NOLINTNEXTLINE(runtime/explicit): Conversion from HalfSpace to HPolygon.
  HPolyhedron(const HalfSpace&) : A_(1, 3), b_(1) {
    A_ << 0, 0, 1;
    b_ << 0;
  }

  const Eigen::MatrixX3<double>& A() const { return A_; }
  const VectorX<double>& b() const { return b_; }

 private:
  Eigen::MatrixX3<double> A_;
  VectorX<double> b_;
};

/* Computes proximity between sphere and sphere. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const Sphere& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const Sphere& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  unused(kappa);
  const Vector3<T> nhat_BA_W =
      (X_WA.translation() - X_WB.translation()).normalized();
  const Vector3<T> p_WCa = X_WA.translation() - nhat_BA_W * shape_A.radius();
  const Vector3<T> p_WCb = X_WB.translation() + nhat_BA_W * shape_B.radius();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between sphere and capsule. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const Sphere& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const Capsule& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  const Vector3<T> p_WA = X_WA.translation();
  const Vector3<T> p_WB1 = X_WB * Vector3<T>{0.0, 0.0, -shape_B.length() / 2};
  const Vector3<T> p_WB2 = X_WB * Vector3<T>{0.0, 0.0, +shape_B.length() / 2};

  const Matrix<T, 1, 1> Q = (p_WB2 - p_WB1).transpose() * (p_WB2 - p_WB1);
  const Vector<T, 1> q = (p_WB2 - p_WB1).transpose() * (p_WB1 - p_WA);
  const Matrix<T, 2, 1> A{1, -1};
  const Vector2<T> b{1, 0};

  unused(kappa);  // Sommthing is not needed since capsule is smooth.
  const Vector<T, 1> beta = SolveQP(Q, q, A, b, 0);
  const Vector3<T> p_WB =
      p_WB1 + (p_WB2 - p_WB1) * beta;  // Point on line segment.
  const Vector3<T> nhat_BA_W = (p_WA - p_WB).normalized();
  const Vector3<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();
  const Vector3<T> p_WCb = p_WB + nhat_BA_W * shape_B.radius();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between capsule and sphere. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_IMPL(Capsule, Sphere);

/* Computes proximity between sphere and H-polyhedron. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const Sphere& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const HPolyhedron& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  const Vector3<T> p_WA = X_WA.translation();
  const Matrix3<T> R_BW = X_WB.rotation().inverse().matrix();

  const Matrix3<T> P = Matrix3<T>::Identity();
  const Vector3<T> q = -p_WA;
  const Eigen::MatrixX3<T> A = shape_B.A() * R_BW;
  const VectorX<T> b = shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector3<T> p_WCb = SolveQP(P, q, A, b, kappa);
  const Vector3<T> nhat_BA_W = (p_WA - p_WCb).normalized();
  const Vector3<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between H-polyhedron and sphere. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_IMPL(HPolyhedron, Sphere);

/* Computes proximity between capsule and capsule. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const Capsule& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const Capsule& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  const Vector3<T> p_WA1 = X_WA * Vector3<T>{0.0, 0.0, -shape_A.length() / 2};
  const Vector3<T> p_WA2 = X_WA * Vector3<T>{0.0, 0.0, +shape_A.length() / 2};
  const Vector3<T> p_WB1 = X_WB * Vector3<T>{0.0, 0.0, -shape_B.length() / 2};
  const Vector3<T> p_WB2 = X_WB * Vector3<T>{0.0, 0.0, +shape_B.length() / 2};

  Matrix<T, 3, 2> S;
  S << p_WA2 - p_WA1, p_WB1 - p_WB2;

  const Matrix2<T> P = S.transpose() * S;
  const Vector2<T> q = S.transpose() * (p_WA1 - p_WB1);
  Matrix<T, 4, 2> A = Matrix<T, 4, 2>::Zero();
  A.template topLeftCorner<2, 1>() = Vector2<T>{1, -1};
  A.template bottomRightCorner<2, 1>() = Vector2<T>{1, -1};
  const Vector<T, 4> b{1, 0, 1, 0};

  unused(kappa);  // Sommthing is not needed since capsule is smooth.
  const Vector2<T> primal = SolveQP(P, q, A, b, 0);
  const T& alpha = primal[0];
  const T& beta = primal[1];
  // Point on line segment.
  const Vector3<T> p_WA = p_WA1 + (p_WA2 - p_WA1) * alpha;
  const Vector3<T> p_WB = p_WB1 + (p_WB2 - p_WB1) * beta;

  const Vector3<T> nhat_BA_W = (p_WA - p_WB).normalized();
  const Vector3<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();
  const Vector3<T> p_WCb = p_WB + nhat_BA_W * shape_B.radius();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between capsule and H-polyhedron. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const Capsule& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const HPolyhedron& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  const Vector3<T> p_WA1 = X_WA * Vector3<T>{0.0, 0.0, -shape_A.length() / 2};
  const Vector3<T> p_WA2 = X_WA * Vector3<T>{0.0, 0.0, +shape_A.length() / 2};
  const Matrix3<T> R_BW = X_WB.rotation().inverse().matrix();

  Matrix<T, 3, 4> S;
  S << p_WA2 - p_WA1, -Matrix3<T>::Identity();

  const Matrix4<T> P = S.transpose() * S;
  const Vector4<T> q = S.transpose() * p_WA1;

  const int num_constraints = 2 + shape_B.b().size();
  Eigen::MatrixX4<T> A = Eigen::MatrixX4<T>::Zero(num_constraints, 4);
  A.template topLeftCorner<2, 1>() = Vector2<T>{1, -1};
  A.bottomRightCorner(num_constraints - 2, 3) = shape_B.A() * R_BW;
  VectorX<T> b(num_constraints);
  b << 1, 0, shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector4<T> primal = SolveQP(P, q, A, b, kappa);
  const T& alpha = primal[0];
  const Vector3<T> p_WCb = primal.template tail<3>();

  const Vector3<T> p_WA =
      p_WA1 + (p_WA2 - p_WA1) * alpha;  // Point on line segment.
  const Vector3<T> nhat_BA_W = (p_WA - p_WCb).normalized();
  const Vector3<T> p_WCa = p_WA - nhat_BA_W * shape_A.radius();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

/* Computes proximity between H-polyhedron and capsule. */
DEFINE_SYMMETRIC_CALC_PROXIMITY_IMPL(HPolyhedron, Capsule);

/* Computes proximity between H-polyhedron and H-polyhedron. */
template <typename T>
SignedDistancePair<T> CalcProximityImpl(const HPolyhedron& shape_A,
                                        const math::RigidTransform<T>& X_WA,
                                        const HPolyhedron& shape_B,
                                        const math::RigidTransform<T>& X_WB,
                                        double kappa) {
  const Matrix3<T> R_AW = X_WA.rotation().inverse().matrix();
  const Matrix3<T> R_BW = X_WB.rotation().inverse().matrix();

  Matrix<T, 3, 6> S;
  S << Matrix3<T>::Identity(), -Matrix3<T>::Identity();

  const Matrix6<T> P = S.transpose() * S;
  const Vector6<T> q = Vector6<T>::Zero();

  const int num_constraints_A = shape_A.b().size();
  const int num_constraints_B = shape_B.b().size();
  Matrix<T, Eigen::Dynamic, 6> A = Matrix<T, Eigen::Dynamic, 6>::Zero(
      num_constraints_A + num_constraints_B, 6);
  A.topLeftCorner(num_constraints_A, 3) = shape_A.A() * R_AW;
  A.bottomRightCorner(num_constraints_B, 3) = shape_B.A() * R_BW;
  VectorX<T> b(num_constraints_A + num_constraints_B);
  b << shape_A.b() + shape_A.A() * R_AW * X_WA.translation(),
      shape_B.b() + shape_B.A() * R_BW * X_WB.translation();

  const Vector6<T> primal = SolveQP(P, q, A, b, kappa);
  const Vector3<T> p_WCa = primal.template head<3>();
  const Vector3<T> p_WCb = primal.template tail<3>();
  // TODO(wei-chen): Handle computing normal when p_WCa is close to p_WCb.
  const Vector3<T> nhat_BA_W = (p_WCa - p_WCb).normalized();

  SignedDistancePair<T> result;
  result.p_ACa = X_WA.inverse() * p_WCa;
  result.p_BCb = X_WB.inverse() * p_WCb;
  result.nhat_BA_W = nhat_BA_W;
  result.distance = (p_WCa - p_WCb).dot(nhat_BA_W);
  return result;
}

#undef DEFINE_SYMMETRIC_CALC_PROXIMITY_IMPL

}  // namespace

template <typename T>
SignedDistancePair2d<T> CalcProximity2d(const Shape2d& shape_A,
                                        const math::RigidTransform2d<T>& X_WA,
                                        const Shape2d& shape_B,
                                        const math::RigidTransform2d<T>& X_WB,
                                        double kappa) {
  DRAKE_THROW_UNLESS(kappa >= 0);
  return shape_A.Visit([&](const auto& shape_a) {
    return shape_B.Visit([&](const auto& shape_b) {
      return CalcProximity2dImpl<T>(shape_a, X_WA, shape_b, X_WB, kappa);
    });
  });
}

template <typename T>
SignedDistancePair<T> CalcProximity(const Shape& shape_A,
                                    const math::RigidTransform<T>& X_WA,
                                    const Shape& shape_B,
                                    const math::RigidTransform<T>& X_WB,
                                    double kappa) {
  DRAKE_THROW_UNLESS(kappa >= 0);
  return shape_A.Visit([&](const auto& shape_a) -> SignedDistancePair<T> {
    return shape_B.Visit([&](const auto& shape_b) -> SignedDistancePair<T> {
#define ARE_ANY(Shape)                                      \
  std::is_same_v<Shape, std::decay_t<decltype(shape_a)>> || \
      std::is_same_v<Shape, std::decay_t<decltype(shape_b)>>
      if constexpr (ARE_ANY(Convex)) {
        throw std::logic_error("CalcProximity() does not support Convex");
      } else if constexpr (ARE_ANY(Cylinder)) {
        throw std::logic_error("CalcProximity() does not support Cylinder");
      } else if constexpr (ARE_ANY(Ellipsoid)) {
        throw std::logic_error("CalcProximity() does not support Ellipsoid");
      } else if constexpr (ARE_ANY(Mesh)) {
        throw std::logic_error("CalcProximity() does not support Mesh");
      } else if constexpr (ARE_ANY(MeshcatCone)) {
        throw std::logic_error("CalcProximity() does not support MeshcatCone");
      } else {
        return CalcProximityImpl<T>(shape_a, X_WA, shape_b, X_WB, kappa);
      }
#undef ARE_ANY
    });
  });
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&CalcProximity2d<T>, &CalcProximity<T>));

}  // namespace geometry
}  // namespace drake
