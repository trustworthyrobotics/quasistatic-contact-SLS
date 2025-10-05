#include "drake/solvers/solve_conic_program.h"

#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace solvers {
namespace {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> ComputeJacobianImplicitDiff(
    const MatrixXd& P, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const std::vector<Cone>& cones, double mu_target) {
  const auto [P_ad, q_ad, A_ad, b_ad] =
      math::InitializeAutoDiffTuple(P, q, A, b);

  VectorX<AutoDiffXd> dx_dmu_ad, dz_dmu_ad;
  const auto [x_ad, z_ad] = SolveConicProgram(
      P_ad, q_ad, A_ad, b_ad, cones, mu_target, &dx_dmu_ad, &dz_dmu_ad);

  const MatrixXd grad_x = math::ExtractGradient(x_ad);
  const MatrixXd grad_z = math::ExtractGradient(z_ad);
  const MatrixXd grad_dx_dmu = math::ExtractGradient(dx_dmu_ad);
  const MatrixXd grad_dz_dmu = math::ExtractGradient(dz_dmu_ad);

  return {grad_x, grad_z, grad_dx_dmu, grad_dz_dmu};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> ComputeJacobianFiniteDiff(
    const MatrixXd& P, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const std::vector<Cone>& cones, double mu_target, double epsilon = 1e-4) {
  VectorXd params(P.size() + q.size() + A.size() + b.size());
  params << P.reshaped(), q.reshaped(), A.reshaped(), b.reshaped();

  const auto solve_conic_program = [&](const VectorXd& vectorized) {
    int offset = 0;
    Eigen::Map<const MatrixXd> P_view(vectorized.data() + offset, P.rows(),
                                      P.cols());
    offset += P.size();
    Eigen::Map<const VectorXd> q_view(vectorized.data() + offset, q.size());
    offset += q.size();
    Eigen::Map<const MatrixXd> A_view(vectorized.data() + offset, A.rows(),
                                      A.cols());
    offset += A.size();
    Eigen::Map<const VectorXd> b_view(vectorized.data() + offset, b.size());

    VectorXd dx_dmu, dz_dmu;
    const auto [x, z] = SolveConicProgram<double>(
        P_view, q_view, A_view, b_view, cones, mu_target, &dx_dmu, &dz_dmu);
    return std::make_tuple(x, z, dx_dmu, dz_dmu);
  };

  MatrixXd grad_x(q.size(), params.size());
  MatrixXd grad_z(b.size(), params.size());
  MatrixXd grad_dx_dmu(q.size(), params.size());
  MatrixXd grad_dz_dmu(b.size(), params.size());
  for (int j = 0; j < params.size(); ++j) {
    params[j] -= epsilon;
    const auto [x1, z1, dx1_dmu, dz1_dmu] = solve_conic_program(params);
    params[j] += epsilon * 2;
    const auto [x2, z2, dx2_dmu, dz2_dmu] = solve_conic_program(params);
    params[j] -= epsilon;

    grad_x.col(j) = (x2 - x1) / (2 * epsilon);
    grad_z.col(j) = (z2 - z1) / (2 * epsilon);
    grad_dx_dmu.col(j) = (dx2_dmu - dx1_dmu) / (2 * epsilon);
    grad_dz_dmu.col(j) = (dz2_dmu - dz1_dmu) / (2 * epsilon);
  }

  return {grad_x, grad_z, grad_dx_dmu, grad_dz_dmu};
}

class SolveConicProgramTest : public ::testing::Test {
 private:
  void SetUp() override {
    P_.resize(3, 3);
    // clang-format off
    P_ << 3, 1, -1,
          1, 4,  2,
         -1, 2,  5;
    // clang-format on

    q_.resize(3);
    q_ << 1, 2, -3;

    A_.resize(5, 3);
    // clang-format off
    A_ << 0,  1,  0,
          0,  0,  1,
         -1,  0,  0,
          0, -1,  0,
          0,  0, -1;
    // clang-format on

    b_.resize(5);
    b_ << 2, 2, 0, 0, 0;
  }

 protected:
  MatrixXd P_;
  VectorXd q_;
  MatrixXd A_;
  VectorXd b_;
};

TEST_F(SolveConicProgramTest, Solve) {
  std::vector<Cone> cones{NonnegativeCone(2), SecondOrderCone(3)};
  const double mu_target = 0.0;
  const auto [x, z] = SolveConicProgram(P_, q_, A_, b_, cones, mu_target);

  /* Check stationarity. */
  const VectorXd stationarity = P_ * x + q_ + A_.transpose() * z;
  EXPECT_TRUE(CompareMatrices(stationarity, VectorXd::Zero(x.size()), 1e-10));

  /* Check primal feasibility. */
  const VectorXd s = b_ - A_ * x;
  const VectorXd s_pos = s.head(2);
  const VectorXd s_soc = s.tail(3);
  EXPECT_TRUE((s_pos.array() > 0).all());
  EXPECT_TRUE(s_soc[0] > s_soc.tail(2).norm());

  /* Check dual feasibility. */
  const VectorXd z_pos = z.head(2);
  const VectorXd z_soc = z.tail(3);
  EXPECT_TRUE((z_pos.array() > 0).all());
  EXPECT_TRUE(z_soc[0] > z_soc.tail(2).norm());

  /* Check complementarity. */
  EXPECT_NEAR(s.dot(z) / 4, 0.0, 1e-9);
}

TEST_F(SolveConicProgramTest, SolveRelaxed) {
  std::vector<Cone> cones{NonnegativeCone(2), SecondOrderCone(3)};
  const double mu_target = 1e-4;
  VectorXd dx_dmu, dz_dmu;
  const auto [x, z] =
      SolveConicProgram(P_, q_, A_, b_, cones, mu_target, &dx_dmu, &dz_dmu);

  /* Check stationarity. */
  const VectorXd stationarity = P_ * x + q_ + A_.transpose() * z;
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(CompareMatrices(stationarity, VectorXd::Zero(x.size()), kTol));

  /* Check primal feasibility. */
  const VectorXd s = b_ - A_ * x;
  const VectorXd s_pos = s.head(2);
  const VectorXd s_soc = s.tail(3);
  EXPECT_TRUE((s_pos.array() > 0).all());
  EXPECT_TRUE(s_soc[0] > s_soc.tail(2).norm());

  /* Check dual feasibility. */
  const VectorXd z_pos = z.head(2);
  const VectorXd z_soc = z.tail(3);
  EXPECT_TRUE((z_pos.array() > 0).all());
  EXPECT_TRUE(z_soc[0] > z_soc.tail(2).norm());

  /* Check complementarity. */
  const int degrees = cones[0].degree() + cones[1].degree();
  EXPECT_NEAR(s.dot(z) / degrees, mu_target, kTol);
  EXPECT_NEAR(s_pos.dot(z_pos), cones[0].degree() * mu_target, kTol);
  EXPECT_NEAR(s_soc.dot(z_soc), cones[1].degree() * mu_target, kTol);

  /* Check dx_dmu and dz_dmu. */
  const double dmu = 1e-6;
  const auto [x2, z2] =
      SolveConicProgram(P_, q_, A_, b_, cones, mu_target + dmu);
  EXPECT_TRUE(CompareMatrices(dx_dmu, (x2 - x) / dmu, 2e-8));
  EXPECT_TRUE(CompareMatrices(dz_dmu, (z2 - z) / dmu, 2e-4));

  /* Compare implicit differentiation Jacobian with finite difference. */
  const auto [grad_x, grad_z, grad_dx_dmu, grad_dz_dmu] =
      ComputeJacobianImplicitDiff(P_, q_, A_, b_, cones, mu_target);
  const auto [grad_x_fd, grad_z_fd, grad_dx_dmu_fd, grad_dz_dmu_fd] =
      ComputeJacobianFiniteDiff(P_, q_, A_, b_, cones, mu_target);
  EXPECT_TRUE(CompareMatrices(grad_x, grad_x_fd, 2e-8));
  EXPECT_TRUE(CompareMatrices(grad_z, grad_z_fd, 3e-4));
  EXPECT_TRUE(CompareMatrices(grad_dx_dmu, grad_dx_dmu_fd, 0.04));
  EXPECT_TRUE(CompareMatrices(grad_dz_dmu, grad_dz_dmu_fd, 0.04));
}

/* This test should trigger Clarabel to give a non-feasible warm x, and the
 log-barrier solver should call other solvers to find a feasible x. */
TEST_F(SolveConicProgramTest, SolveInfeasibleWarmX) {
  using Eigen::Vector2d;

  MatrixXd P = MatrixXd::Identity(2, 2);
  VectorXd q = Vector2d{-0.5, 0.0};
  MatrixXd A = MatrixXd::Zero(4, 2);
  A.topLeftCorner<2, 1>() = Vector2d{1, -1};
  A.bottomRightCorner<2, 1>() = Vector2d{1, -1};
  VectorXd b = VectorXd::Ones(4) * 0.1;

  const double kappa = 1e-3;
  const auto [x, z] =
      SolveConicProgram(P, q, A, b, {solvers::NonnegativeCone(4)}, kappa);
  EXPECT_TRUE(CompareMatrices(x, Vector2d{0.09754611, 0.0}, 1e-8));
}

GTEST_TEST(ConeTest, NonnegativeCone) {
  Cone cone = NonnegativeCone(5);
  EXPECT_EQ(cone.tag(), Cone::Tag::NonnegativeCone);
  EXPECT_EQ(cone.dimension(), 5);
  EXPECT_EQ(cone.degree(), 5);

  cone = NonnegativeCone(3);
  EXPECT_EQ(cone.tag(), Cone::Tag::NonnegativeCone);
  EXPECT_EQ(cone.dimension(), 3);
  EXPECT_EQ(cone.degree(), 3);

  /* Check log-barrier value, gradient, and hessian. */
  const Vector3d s{5, 6, 7};

  const double phi = cone.CalcLogBarrier(s);
  const double phi_expected = -s.array().log().sum();
  EXPECT_DOUBLE_EQ(phi, phi_expected);

  const VectorXd grad_phi = cone.CalcLogBarrierGradient(s);
  const VectorXd grad_phi_expected = -s.cwiseInverse();
  EXPECT_TRUE(CompareMatrices(grad_phi, grad_phi_expected));

  const MatrixXd hess_phi = cone.CalcLogBarrierHessian(s);
  const MatrixXd hess_phi_expected = s.cwiseAbs2().cwiseInverse().asDiagonal();
  EXPECT_TRUE(CompareMatrices(hess_phi, hess_phi_expected));

  /* Check the Jordan algebra. */
  EXPECT_TRUE(CompareMatrices(cone.JordanIdentity(), Vector3d::Ones()));
  EXPECT_EQ(cone.JordanIdentity().dot(cone.JordanIdentity()), cone.degree());

  const Vector3d z = -cone.CalcLogBarrierGradient(s);
  EXPECT_TRUE(CompareMatrices(cone.JordanLeftMul<double>(s) * z,
                              cone.JordanIdentity()));
  EXPECT_TRUE(CompareMatrices(
      cone.JordanLeftMulInverse<double>(s) * cone.JordanIdentity(), z));
}

GTEST_TEST(ConeTest, SecondOrderCone) {
  Cone cone = SecondOrderCone(2);
  EXPECT_EQ(cone.tag(), Cone::Tag::SecondOrderCone);
  EXPECT_EQ(cone.dimension(), 2);
  EXPECT_EQ(cone.degree(), 1);

  cone = SecondOrderCone(3);
  EXPECT_EQ(cone.tag(), Cone::Tag::SecondOrderCone);
  EXPECT_EQ(cone.dimension(), 3);
  EXPECT_EQ(cone.degree(), 1);

  /* Check log-barrier value, gradient, and hessian. */
  const Vector3d s{6, 5, 3};

  const MatrixXd J = Vector3d{1, -1, -1}.asDiagonal();

  const double phi = cone.CalcLogBarrier(s);
  const double phi_expected = -0.5 * std::log(s.transpose() * J * s);
  EXPECT_DOUBLE_EQ(phi, phi_expected);

  using Eigen::Vector2d;
  const VectorXd grad_phi = cone.CalcLogBarrierGradient(s);
  const VectorXd grad_phi_expected = -1 / (s.transpose() * J * s) * (J * s);
  EXPECT_TRUE(CompareMatrices(grad_phi, grad_phi_expected));

  const MatrixXd hess_phi = cone.CalcLogBarrierHessian(s);
  const MatrixXd hess_phi_expected =
      2 / pow(s.transpose() * J * s, 2) * (J * s * s.transpose() * J) -
      1 / (s.transpose() * J * s) * J;
  EXPECT_TRUE(CompareMatrices(hess_phi, hess_phi_expected));

  /* Check the Jordan algebra. */
  EXPECT_TRUE(CompareMatrices(cone.JordanIdentity(), Vector3d{1, 0, 0}));
  EXPECT_EQ(cone.JordanIdentity().dot(cone.JordanIdentity()), cone.degree());

  const Vector3d z = -cone.CalcLogBarrierGradient(s);
  EXPECT_TRUE(CompareMatrices(cone.JordanLeftMul<double>(s) * z,
                              cone.JordanIdentity()));
  EXPECT_TRUE(CompareMatrices(
      cone.JordanLeftMulInverse<double>(s) * cone.JordanIdentity(), z));
}

}  // namespace
}  // namespace solvers
}  // namespace drake
