#include "drake/math/rigid_transform_2d.h"

#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace math {
namespace {

using Eigen::Vector2d;

constexpr double kTol = std::numeric_limits<double>::epsilon();

GTEST_TEST(RigidTransform2dTest, ConstructionANdSet) {
  RigidTransform2d<double> X1;
  EXPECT_EQ(X1.angle(), 0);
  EXPECT_EQ(X1.translation(), Vector2d::Zero());

  RigidTransform2d<double> X2(1, Vector2d{2, 3});
  EXPECT_EQ(X2.angle(), 1);
  EXPECT_EQ(X2.translation(), Vector2d(2, 3));

  RigidTransform2d<double> X3(1);
  EXPECT_EQ(X3.angle(), 1);
  EXPECT_EQ(X3.translation(), Vector2d::Zero());

  RigidTransform2d<double> X4(Vector2d{2, 3});
  EXPECT_EQ(X4.angle(), 0);
  EXPECT_EQ(X4.translation(), Vector2d(2, 3));

  RigidTransform2d<double> X5(Eigen::Translation<double, 2>(Vector2d{2, 3}));
  EXPECT_EQ(X5.angle(), 0);
  EXPECT_EQ(X5.translation(), Vector2d(2, 3));

  X1.set(1, Vector2d{2, 3});
  EXPECT_EQ(X1.angle(), 1);
  EXPECT_EQ(X1.translation(), Vector2d(2, 3));

  X3.set_translation(Vector2d{2, 3});
  EXPECT_EQ(X3.translation(), Vector2d(2, 3));

  X4.set_angle(1);
  EXPECT_EQ(X4.angle(), 1);

  X1 = RigidTransform2d<double>::Identity();
  EXPECT_EQ(X1.angle(), 0);
  EXPECT_EQ(X1.translation(), Vector2d::Zero());

  X2.SetIdentity();
  EXPECT_EQ(X2.angle(), 0);
  EXPECT_EQ(X2.translation(), Vector2d::Zero());
}

GTEST_TEST(RigidTransform2dTest, Cast) {
  RigidTransform2d<double> X(1, Vector2d{2, 3});

  RigidTransform2d<AutoDiffXd> X_ad = X.cast<AutoDiffXd>();
  EXPECT_EQ(ExtractDoubleOrThrow(X_ad.angle()), 1);
  EXPECT_EQ(ExtractDoubleOrThrow(X_ad.translation()), Vector2d(2, 3));

  RigidTransform2d<symbolic::Expression> X_expr =
      X.cast<symbolic::Expression>();
  EXPECT_EQ(ExtractDoubleOrThrow(X_expr.angle()), 1);
  EXPECT_EQ(ExtractDoubleOrThrow(X_expr.translation()), Vector2d(2, 3));
}

GTEST_TEST(RigidTransform2dTest, GetRotationMatrix) {
  RigidTransform2d<double> X(M_PI / 6);

  Eigen::Matrix2d R;
  // clang-format off
  R << sqrt(3) / 2,    -1.0 / 2,
           1.0 / 2, sqrt(3) / 2;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(X.rotation_matrix(), R, kTol));
}

GTEST_TEST(RigidTransform2dTest, GetAsMatrix3) {
  RigidTransform2d<double> X(M_PI / 6, Vector2d{2, 3});

  Eigen::Matrix3d M;
  // clang-format off
  M << sqrt(3) / 2,    -1.0 / 2, 2,
           1.0 / 2, sqrt(3) / 2, 3,
                 0,           0, 1;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(X.GetAsMatrix3(), M, kTol));
  EXPECT_TRUE(CompareMatrices(X.GetAsMatrix23(), M.topRows<2>(), kTol));
}

GTEST_TEST(RigidTransform2dTest, GetAsIsometry2) {
  RigidTransform2d<double> X(M_PI / 6, Vector2d{2, 3});
  Eigen::Isometry2d isometry = X.GetAsIsometry2();

  Eigen::Matrix2d R;
  // clang-format off
  R << sqrt(3) / 2,    -1.0 / 2,
           1.0 / 2, sqrt(3) / 2;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(isometry.rotation(), R, kTol));
  EXPECT_TRUE(CompareMatrices(isometry.translation(), Vector2d(2, 3)));
}

GTEST_TEST(RigidTransform2dTest, Inverse) {
  RigidTransform2d<double> X(1, Vector2d{2, 3});
  RigidTransform2d<double> invX = X.inverse();

  EXPECT_TRUE(CompareMatrices(invX.GetAsMatrix3(), X.GetAsMatrix3().inverse()));
  EXPECT_TRUE(CompareMatrices(invX.GetAsMatrix3() * X.GetAsMatrix3(),
                              Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(CompareMatrices(X.GetAsMatrix3() * invX.GetAsMatrix3(),
                              Eigen::Matrix3d::Identity()));
}

GTEST_TEST(RigidTransform2dTest, InvertAndCompose) {
  const RigidTransform2d<double> X_WA(1, Vector2d{2, 3});
  const RigidTransform2d<double> X_WB(4, Vector2d{5, 6});
  const RigidTransform2d<double> X_AB = X_WA.InvertAndCompose(X_WB);

  EXPECT_TRUE(CompareMatrices(
      X_AB.GetAsMatrix3(), X_WA.GetAsMatrix3().inverse() * X_WB.GetAsMatrix3(),
      kTol));
}

GTEST_TEST(RigidTransform2dTest, MultiplyAssign) {
  const RigidTransform2d<double> X_WA(1, Vector2d{2, 3});
  const RigidTransform2d<double> X_AB(4, Vector2d{5, 6});

  RigidTransform2d<double> X_WB = X_WA;
  X_WB *= X_AB;

  EXPECT_TRUE(CompareMatrices(X_WB.GetAsMatrix3(),
                              X_WA.GetAsMatrix3() * X_AB.GetAsMatrix3(), kTol));
}

GTEST_TEST(RigidTransform2dTest, MultiplyRigidTransform2d) {
  const RigidTransform2d<double> X_WA(1, Vector2d{2, 3});
  const RigidTransform2d<double> X_AB(4, Vector2d{5, 6});

  const RigidTransform2d<double> X_WB = X_WA * X_AB;

  EXPECT_TRUE(CompareMatrices(X_WB.GetAsMatrix3(),
                              X_WA.GetAsMatrix3() * X_AB.GetAsMatrix3(), kTol));
}

GTEST_TEST(RigidTransform2dTest, MultiplyVector) {
  const RigidTransform2d<double> X_AB(1, Vector2d{2, 3});
  Vector2d p_BoQ_B{4, 5};

  const Vector2d p_AoQ_A = X_AB * p_BoQ_B;
  EXPECT_TRUE(CompareMatrices(
      p_AoQ_A, X_AB.GetAsMatrix23() * p_BoQ_B.homogeneous().eval()));
}

GTEST_TEST(RigidTransform2dTest, AutoDiff) {
  Eigen::VectorXd theta_AB(1);
  theta_AB << M_PI / 6;
  const Vector2d p_AoBo_A{2, 3};
  const Vector2d p_BoQ_B{2, 0};
  const auto autodiff_tuple =
      math::InitializeAutoDiffTuple(theta_AB, p_AoBo_A, p_BoQ_B);

  const RigidTransform2d<AutoDiffXd> X_AB_ad(std::get<0>(autodiff_tuple)[0],
                                             std::get<1>(autodiff_tuple));
  const Vector2<AutoDiffXd> p_BoQ_B_ad = std::get<2>(autodiff_tuple);

  const Vector2<AutoDiffXd> p_AoQ_A_ad = X_AB_ad * p_BoQ_B_ad;
  const Eigen::MatrixXd jacobian = math::ExtractGradient(p_AoQ_A_ad);

  Eigen::Matrix<double, 2, 5> jacobian_expected;
  // clang-format off
  jacobian_expected <<   -1.0, 1, 0, sqrt(3) / 2,    -1.0 / 2,
                      sqrt(3), 0, 1,     1.0 / 2, sqrt(3) / 2;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(jacobian, jacobian_expected, kTol));
}

GTEST_TEST(RigidTransform2dTest, MultiplyMultipleVector) {
  const RigidTransform2d<double> X_AB(1, Vector2d{2, 3});
  Eigen::Matrix2Xd p_BoQ_B(2, 4);
  p_BoQ_B << 1, 2, 3, 4, 5, 6, 7, 8;

  const Eigen::Matrix2Xd p_AoQ_A = X_AB * p_BoQ_B;
  for (int j = 0; j < p_BoQ_B.cols(); ++j) {
    EXPECT_TRUE(CompareMatrices(
        p_AoQ_A.col(j),
        X_AB.GetAsMatrix23() * p_BoQ_B.col(j).homogeneous().eval()));
  }
}

GTEST_TEST(RigidTransform2dTest, Formatting) {
  RigidTransform2d<double> X_AB(1, Vector2d{2, 3});
  std::string str = fmt::format("{}", X_AB);
  EXPECT_EQ(str, "theta = 1 xy = 2 3");
}

}  // namespace
}  // namespace math
}  // namespace drake
