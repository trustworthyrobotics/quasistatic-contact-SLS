#include "drake/multibody/quasidynamic/quasidynamic_linear_pusher.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/quasidynamic/test_utilities/eval_quasidynamic_differentiable_contact_model.h"

namespace drake {
namespace multibody {
namespace quasidynamic {
namespace {

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::RowVector2d;
using Eigen::Vector2d;
using Eigen::VectorXd;
using systems::Context;
using systems::System;
using Vector1d = Eigen::Vector<double, 1>;

class QuasidynamicLinearPusherTest : public ::testing::Test {
 private:
  void SetUp() override {
    model_ =
        std::make_unique<QuasidynamicLinearPusher<double>>(k_a_, m_o_, w_, dt_);
  }

 protected:
  const double k_a_ = 1.0;
  const double m_o_ = 0.01;
  const double w_ = 0.1;
  const double dt_ = 0.01;
  std::unique_ptr<QuasidynamicDifferentiableContactModel<double>> model_;
};

TEST_F(QuasidynamicLinearPusherTest, ScalarConversion) {
  std::unique_ptr<System<AutoDiffXd>> model_ad = model_->ToAutoDiffXdMaybe();
  EXPECT_NE(model_ad, nullptr);
  std::unique_ptr<System<double>> model = model_ad->ToScalarType<double>();
  EXPECT_NE(model, nullptr);
  EXPECT_NO_THROW(model_->Clone());
}

TEST_F(QuasidynamicLinearPusherTest, DynamicsMatrix) {
  auto context = model_->CreateDefaultContext();
  const Vector2d x{0.0, 0.2};
  const Vector1d u{0.15};
  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);

  const MatrixXd& P = model_->EvalQuasidynamicPMatrix(*context);
  const Matrix2d P_expected = Vector2d{k_a_, m_o_ / pow(dt_, 2)}.asDiagonal();
  EXPECT_TRUE(CompareMatrices(P, P_expected));

  const VectorXd& b = model_->EvalQuasidynamicBVector(*context);
  const Vector2d b_expected{-k_a_ * u[0], -m_o_ * x[1] / pow(dt_, 2)};
  EXPECT_TRUE(CompareMatrices(b, b_expected));

  const MatrixXd& J = model_->EvalContactJacobian(*context);
  const RowVector2d J_expected{-1, 1};
  EXPECT_TRUE(CompareMatrices(J, J_expected));

  const VectorXd& phi = model_->EvalContactSignedDistance(*context);
  const Vector1d phi_expected{x[1] - x[0] - w_};
  EXPECT_TRUE(CompareMatrices(phi, phi_expected));
}

TEST_F(QuasidynamicLinearPusherTest, NonsmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 0.0;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector2d x{0.0, 0.2};
  const Vector1d u{0.15};

  VectorXd xnext, lambda;
  MatrixXd E;
  std::tie(xnext, lambda, E) = EvalStep(*model_, context.get(), x, u);

  const Vector2d xnext_expected{0.1004950, 0.2004950};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  const Matrix2d P = Vector2d{k_a_, m_o_ / (dt_ * dt_)}.asDiagonal();
  const Vector2d b = Vector2d{-k_a_ * u[0], -m_o_ * x[1] / (dt_ * dt_)};
  const Vector1d lambda_expected = Vector1d{(P * xnext + b)[1]};
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-7));

  const Vector2d E_expected = Vector2d::Zero();
  EXPECT_TRUE(CompareMatrices(E, E_expected));

  const auto [dxnext_dxu, dlambda_dxu, dE_dxu] =
      EvalJacobian(*model_, context.get(), x, u);
  const auto [dxnext_dxu_fd, dlambda_dxu_fd, dE_dxu_fd] =
      ComputeFiniteDiffJacobian(*model_, context.get(), x, u);
  constexpr double kTol = 1e-5;
  EXPECT_TRUE(CompareMatrices(dxnext_dxu, dxnext_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu, dlambda_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dE_dxu, dE_dxu_fd, kTol));
}

TEST_F(QuasidynamicLinearPusherTest, SmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 1e-2;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector2d x{0.0, 0.2};
  Vector1d u{0.1};

  VectorXd xnext, lambda;
  MatrixXd E;
  std::tie(xnext, lambda, E) = EvalStep(*model_, context.get(), x, u);

  const Vector2d xnext_expected{0.0004963, 0.2009950};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  const RowVector2d J{-1, 1};
  const double nu = J * xnext - w_;
  const Vector1d lambda_expected{kappa / nu};
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-7));

  const Matrix2d invP = Vector2d{1 / k_a_, (dt_ * dt_) / m_o_}.asDiagonal();
  const Vector2d E_expected = -invP * J.transpose() /
                              (nu + lambda[0] * J * invP * J.transpose()) *
                              kappa;
  EXPECT_TRUE(CompareMatrices(E, E_expected, 1e-10));

  u = Vector1d{0.2};
  const auto [dxnext_dxu, dlambda_dxu, dE_dxu] =
      EvalJacobian(*model_, context.get(), x, u);
  const auto [dxnext_dxu_fd, dlambda_dxu_fd, dE_dxu_fd] =
      ComputeFiniteDiffJacobian(*model_, context.get(), x, u, 1e-8);
  constexpr double kTol = 1e-8;
  EXPECT_TRUE(CompareMatrices(dxnext_dxu, dxnext_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu, dlambda_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dE_dxu, dE_dxu_fd, kTol));
}

}  // namespace
}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake
