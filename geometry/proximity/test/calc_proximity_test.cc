#include "drake/geometry/proximity/calc_proximity.h"

#include <tuple>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;
using math::RigidTransform;
using math::RigidTransform2d;

struct SignedDistancePair2dJacobian {
  Eigen::Matrix<double, 2, 6> grad_p_ACa;
  Eigen::Matrix<double, 2, 6> grad_p_BCb;
  Eigen::Matrix<double, 2, 6> grad_nhat_BA_W;
  Eigen::Matrix<double, 1, 6> grad_distance;
};

SignedDistancePair2dJacobian ComputeJacobianImplicitDiff(
    const Shape2d& shape_A, const RigidTransform2d<double>& X_WA,
    const Shape2d& shape_B, const RigidTransform2d<double>& X_WB,
    double kappa) {
  using Vector1d = Eigen::Vector<double, 1>;
  const auto ad_tuple = math::InitializeAutoDiffTuple(
      Vector1d(X_WA.angle()), Eigen::VectorXd(X_WA.translation()),
      Vector1d(X_WB.angle()), Eigen::VectorXd(X_WB.translation()));
  const RigidTransform2d<AutoDiffXd> X_WA_ad(std::get<0>(ad_tuple)[0],
                                             std::get<1>(ad_tuple));
  const RigidTransform2d<AutoDiffXd> X_WB_ad(std::get<2>(ad_tuple)[0],
                                             std::get<3>(ad_tuple));
  const SignedDistancePair2d<AutoDiffXd> r =
      CalcProximity2d(shape_A, X_WA_ad, shape_B, X_WB_ad, kappa);

  return SignedDistancePair2dJacobian{
      .grad_p_ACa = math::ExtractGradient(r.p_ACa),
      .grad_p_BCb = math::ExtractGradient(r.p_BCb),
      .grad_nhat_BA_W = math::ExtractGradient(r.nhat_BA_W),
      .grad_distance = r.distance.derivatives(),
  };
}

SignedDistancePair2dJacobian ComputeJacobianFiniteDiff(
    const Shape2d& shape_A, const RigidTransform2d<double>& X_WA_in,
    const Shape2d& shape_B, const RigidTransform2d<double>& X_WB_in,
    double kappa, double epsilon = 1e-8) {
  Eigen::Matrix<double, 2, 6> grad_p_ACa;
  Eigen::Matrix<double, 2, 6> grad_p_BCb;
  Eigen::Matrix<double, 2, 6> grad_nhat_BA_W;
  Eigen::Matrix<double, 1, 6> grad_distance;

  Eigen::Vector<double, 6> param;
  param << X_WA_in.angle(), X_WA_in.translation(), X_WB_in.angle(),
      X_WB_in.translation();

  const auto calc_proximity = [&](const Eigen::Vector<double, 6>& vectorized) {
    const RigidTransform2d<double> X_WA(param[0], param.segment<2>(1));
    const RigidTransform2d<double> X_WB(param[3], param.segment<2>(4));
    return CalcProximity2d(shape_A, X_WA, shape_B, X_WB, kappa);
  };

  for (int j = 0; j < param.size(); ++j) {
    param[j] -= epsilon;
    const SignedDistancePair2d<double> r1 = calc_proximity(param);
    param[j] += epsilon * 2;
    const SignedDistancePair2d<double> r2 = calc_proximity(param);
    param[j] -= epsilon;

    grad_p_ACa.col(j) = (r2.p_ACa - r1.p_ACa) / (2 * epsilon);
    grad_p_BCb.col(j) = (r2.p_BCb - r1.p_BCb) / (2 * epsilon);
    grad_nhat_BA_W.col(j) = (r2.nhat_BA_W - r1.nhat_BA_W) / (2 * epsilon);
    grad_distance[j] = (r2.distance - r1.distance) / (2 * epsilon);
  }

  return SignedDistancePair2dJacobian{
      .grad_p_ACa = grad_p_ACa,
      .grad_p_BCb = grad_p_BCb,
      .grad_nhat_BA_W = grad_nhat_BA_W,
      .grad_distance = grad_distance,
  };
}

void CheckJacobians(const Shape2d& shape_A,
                    const RigidTransform2d<double>& X_WA,
                    const Shape2d& shape_B,
                    const RigidTransform2d<double>& X_WB, double kappa,
                    double tolerance = 0.0, double finite_diff_epsilon = 1e-8) {
  const auto ad =
      ComputeJacobianImplicitDiff(shape_A, X_WA, shape_B, X_WB, kappa);
  const auto fd = ComputeJacobianFiniteDiff(shape_A, X_WA, shape_B, X_WB, kappa,
                                            finite_diff_epsilon);
  EXPECT_TRUE(CompareMatrices(ad.grad_p_ACa, fd.grad_p_ACa, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_p_BCb, fd.grad_p_BCb, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_nhat_BA_W, fd.grad_nhat_BA_W, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_distance, fd.grad_distance, tolerance));
}

GTEST_TEST(CalcProximity2dTest, CircleCircle) {
  {
    const Circle shape_A(2);
    const RigidTransform2d<double> X_WA(0, Vector2d{0, 0});
    const Circle shape_B(1);
    const RigidTransform2d<double> X_WB(0, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 0.0;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximity2dTest, CircleObround) {
  {
    const Circle shape_A(1);
    const RigidTransform2d<double> X_WA(0, Vector2d{0, 0});
    const Obround shape_B(1, 2);
    const RigidTransform2d<double> X_WB(0, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
  {
    const Obround shape_A(1, 2);
    const RigidTransform2d<double> X_WA(M_PI / 2, Vector2d{0, 0});
    const Circle shape_B(1);
    const RigidTransform2d<double> X_WB(0, Vector2d{3, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{0, -1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximity2dTest, CircleRectangle) {
  {
    const Circle shape_A(1);
    const RigidTransform2d<double> X_WA(0, Vector2d{0, 0});
    const Rectangle shape_B(4, 2);
    const RigidTransform2d<double> X_WB(0, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 5e-7,
                   /*finite_diff_epsilon*/ 1e-4);
  }
  {
    const Rectangle shape_A(2 * sqrt(2), 2 * sqrt(2));
    const RigidTransform2d<double> X_WA(-M_PI / 4, Vector2d{0, 0});
    const Circle shape_B(1);
    const RigidTransform2d<double> X_WB(0, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{sqrt(2), sqrt(2)}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

GTEST_TEST(CalcProximity2dTest, ObroundObround) {
  {
    const Obround shape_A(1, 3);
    const RigidTransform2d<double> X_WA(M_PI / 2, Vector2d{0, 0});
    const Obround shape_B(1, 2);
    const RigidTransform2d<double> X_WB(0, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{0, -1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximity2dTest, ObroundRectangle) {
  {
    const Obround shape_A(1, 2);
    const RigidTransform2d<double> X_WA(0, Vector2d{0, 0});
    const Rectangle shape_B(4, 2);
    const RigidTransform2d<double> X_WB(0, Vector2d{5, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{-2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
  {
    const Rectangle shape_A(2 * sqrt(2), 2 * sqrt(2));
    const RigidTransform2d<double> X_WA(-M_PI / 4, Vector2d{0, 0});
    const Obround shape_B(1, 2);
    const RigidTransform2d<double> X_WB(M_PI / 2, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{sqrt(2), sqrt(2)}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{0, 1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

GTEST_TEST(CalcProximity2dTest, RectangleRectangle) {
  {
    const Rectangle shape_A(2 * sqrt(2), 2 * sqrt(2));
    const RigidTransform2d<double> X_WA(-M_PI / 4, Vector2d{0, 0});
    const Rectangle shape_B(4, 2);
    const RigidTransform2d<double> X_WB(M_PI / 2, Vector2d{4, 0});

    const auto r = CalcProximity2d(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector2d{sqrt(2), sqrt(2)}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector2d{0, 1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector2d{-1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

struct SignedDistancePairJacobian {
  Eigen::Matrix<double, 3, 12> grad_p_ACa;
  Eigen::Matrix<double, 3, 12> grad_p_BCb;
  Eigen::Matrix<double, 3, 12> grad_nhat_BA_W;
  Eigen::Matrix<double, 1, 12> grad_distance;
};

SignedDistancePairJacobian ComputeJacobianImplicitDiff(
    const Shape& shape_A, const RigidTransform<double>& X_WA,
    const Shape& shape_B, const RigidTransform<double>& X_WB, double kappa) {
  const auto ad_tuple =
      math::InitializeAutoDiffTuple(X_WA.rotation().ToRollPitchYaw().vector(),
                                    Eigen::VectorXd(X_WA.translation()),
                                    X_WB.rotation().ToRollPitchYaw().vector(),
                                    Eigen::VectorXd(X_WB.translation()));
  const RigidTransform<AutoDiffXd> X_WA_ad(
      math::RotationMatrix(math::RollPitchYaw(std::get<0>(ad_tuple))),
      std::get<1>(ad_tuple));
  const RigidTransform<AutoDiffXd> X_WB_ad(
      math::RotationMatrix(math::RollPitchYaw(std::get<2>(ad_tuple))),
      std::get<3>(ad_tuple));
  const SignedDistancePair<AutoDiffXd> r =
      CalcProximity(shape_A, X_WA_ad, shape_B, X_WB_ad, kappa);

  return SignedDistancePairJacobian{
      .grad_p_ACa = math::ExtractGradient(r.p_ACa),
      .grad_p_BCb = math::ExtractGradient(r.p_BCb),
      .grad_nhat_BA_W = math::ExtractGradient(r.nhat_BA_W),
      .grad_distance = r.distance.derivatives(),
  };
}

SignedDistancePairJacobian ComputeJacobianFiniteDiff(
    const Shape& shape_A, const RigidTransform<double>& X_WA_in,
    const Shape& shape_B, const RigidTransform<double>& X_WB_in, double kappa,
    double epsilon = 1e-8) {
  Eigen::Matrix<double, 3, 12> grad_p_ACa;
  Eigen::Matrix<double, 3, 12> grad_p_BCb;
  Eigen::Matrix<double, 3, 12> grad_nhat_BA_W;
  Eigen::Matrix<double, 1, 12> grad_distance;

  Eigen::Vector<double, 12> param;
  param << X_WA_in.rotation().ToRollPitchYaw().vector(), X_WA_in.translation(),
      X_WB_in.rotation().ToRollPitchYaw().vector(), X_WB_in.translation();

  const auto calc_proximity = [&](const Eigen::Vector<double, 12>& vectorized) {
    const RigidTransform<double> X_WA(
        math::RotationMatrix(
            math::RollPitchYaw<double>(vectorized.segment<3>(0))),
        vectorized.segment<3>(3));
    const RigidTransform<double> X_WB(
        math::RotationMatrix(
            math::RollPitchYaw<double>(vectorized.segment<3>(6))),
        vectorized.segment<3>(9));
    return CalcProximity(shape_A, X_WA, shape_B, X_WB, kappa);
  };

  for (int j = 0; j < param.size(); ++j) {
    param[j] -= epsilon;
    const SignedDistancePair<double> r1 = calc_proximity(param);
    param[j] += epsilon * 2;
    const SignedDistancePair<double> r2 = calc_proximity(param);
    param[j] -= epsilon;

    grad_p_ACa.col(j) = (r2.p_ACa - r1.p_ACa) / (2 * epsilon);
    grad_p_BCb.col(j) = (r2.p_BCb - r1.p_BCb) / (2 * epsilon);
    grad_nhat_BA_W.col(j) = (r2.nhat_BA_W - r1.nhat_BA_W) / (2 * epsilon);
    grad_distance[j] = (r2.distance - r1.distance) / (2 * epsilon);
  }

  return SignedDistancePairJacobian{
      .grad_p_ACa = grad_p_ACa,
      .grad_p_BCb = grad_p_BCb,
      .grad_nhat_BA_W = grad_nhat_BA_W,
      .grad_distance = grad_distance,
  };
}

void CheckJacobians(const Shape& shape_A, const RigidTransform<double>& X_WA,
                    const Shape& shape_B, const RigidTransform<double>& X_WB,
                    double kappa, double tolerance = 0.0,
                    double finite_diff_epsilon = 1e-8) {
  const auto ad =
      ComputeJacobianImplicitDiff(shape_A, X_WA, shape_B, X_WB, kappa);
  const auto fd = ComputeJacobianFiniteDiff(shape_A, X_WA, shape_B, X_WB, kappa,
                                            finite_diff_epsilon);
  EXPECT_TRUE(CompareMatrices(ad.grad_p_ACa, fd.grad_p_ACa, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_p_BCb, fd.grad_p_BCb, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_nhat_BA_W, fd.grad_nhat_BA_W, tolerance));
  EXPECT_TRUE(CompareMatrices(ad.grad_distance, fd.grad_distance, tolerance));
}

GTEST_TEST(CalcProximityTest, SphereSphere) {
  {
    const Sphere shape_A(2);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Sphere shape_B(1);
    const RigidTransform<double> X_WB(Vector3d{4, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 0.0;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{2, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{-1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximityTest, SphereCapsule) {
  {
    const Sphere shape_A(1);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Capsule shape_B(1, 2);
    const RigidTransform<double> X_WB(Vector3d{0, 0, 4});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{0, 0, 1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{0, 0, -2}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{0, 0, -1}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
  {
    const Capsule shape_A(1, 2);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Sphere shape_B(1);
    const RigidTransform<double> X_WB(Vector3d{3, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{-1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximityTest, SphereBox) {
  {
    const Sphere shape_A(1);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Box shape_B(4, 3, 2);
    const RigidTransform<double> X_WB(Vector3d{4, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-10;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{-2, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
  {
    const Box shape_A(3, 4, 2);
    const RigidTransform<double> X_WA(
        math::RotationMatrixd::MakeZRotation(M_PI / 2), Vector3d{0, 0, 0});
    const Sphere shape_B(1);
    const RigidTransform<double> X_WB(Vector3d{4, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-8;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{0, -2, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{-1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 3e-6,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

GTEST_TEST(CalcProximityTest, SphereHalfspace) {
  {
    const HalfSpace shape_A;
    const RigidTransform<double> X_WA(
        math::RotationMatrixd::MakeXRotation(M_PI / 2));
    const Sphere shape_B(1);
    const RigidTransform<double> X_WB(Vector3d{3, -2, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-8;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{3, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{0, 1, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{0, 1, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 5e-7,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

GTEST_TEST(CalcProximityTest, CapsuleCapsule) {
  {
    const Capsule shape_A(1, 3);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Capsule shape_B(1, 2);
    const RigidTransform<double> X_WB(
        math::RotationMatrixd::MakeYRotation(-M_PI / 2), Vector3d{4, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{1, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{0, 0, 2}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
}

GTEST_TEST(CalcProximityTest, CapsuleBox) {
  {
    const Capsule shape_A(1, 2);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Box shape_B(4, 3, 2);
    const RigidTransform<double> X_WB(Vector3d{0, 0, 4});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{0, 0, 2}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{0, 0, -1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{0, 0, -1}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 1e-7);
  }
  {
    const Box shape_A(4, 3, 2);
    const RigidTransform<double> X_WA(Vector3d{0, 0, 0});
    const Capsule shape_B(1, 2);
    const RigidTransform<double> X_WB(
        math::RotationMatrixd::MakeYRotation(-M_PI / 2), Vector3d{5, 0, 0});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{2, 0, 0}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{0, 0, 2}, kTol));
    EXPECT_TRUE(CompareMatrices(r.nhat_BA_W, Vector3d{-1, 0, 0}, kTol));
    EXPECT_NEAR(r.distance, 1, kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 2e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 5e-6,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

GTEST_TEST(CalcProximityTest, BoxBox) {
  {
    const Box shape_A(4, 3, 2);
    const RigidTransform<double> X_WA(
        math::RotationMatrixd::MakeZRotation(M_PI / 2), Vector3d{0, 0, 0});
    const Box shape_B(2, 2, 2);
    const RigidTransform<double> X_WB(Vector3d{3.5, 4, 3});

    const auto r = CalcProximity(shape_A, X_WA, shape_B, X_WB);

    constexpr double kTol = 1e-9;
    EXPECT_TRUE(CompareMatrices(r.p_ACa, Vector3d{2, -1.5, 1}, kTol));
    EXPECT_TRUE(CompareMatrices(r.p_BCb, Vector3d{-1, -1, -1}, kTol));
    EXPECT_TRUE(
        CompareMatrices(r.nhat_BA_W, Vector3d{-1, -1, -1}.normalized(), kTol));
    EXPECT_NEAR(r.distance, sqrt(3), kTol);

    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 0.0, /*tol*/ 1e-7);
    CheckJacobians(shape_A, X_WA, shape_B, X_WB, /*kappa*/ 1e-3, /*tol*/ 2e-6,
                   /*finite_diff_epsilon*/ 1e-4);
  }
}

}  // namespace
}  // namespace geometry
}  // namespace drake
