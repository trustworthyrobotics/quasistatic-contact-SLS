#include "drake/multibody/quasidynamic/quasidynamic_planar_plant.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/pointer_cast.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/text_logging.h"
#include "drake/multibody/quasidynamic/test_utilities/eval_quasidynamic_differentiable_contact_model.h"

namespace drake {
namespace multibody {
namespace quasidynamic {
namespace {

using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Vector5d = Eigen::Vector<double, 5>;
using math::RigidTransform2d;
using systems::Context;
using systems::System;

class QuasidynamicPlanarPlantTest : public ::testing::Test {
 private:
  void SetUp() override {
    QuasidynamicPlanarPlant<double> plant(dt_);
    object_ = plant.AddRigidBody(m_o_, Vector2d::Zero(), I_o_);
    plant.RegisterCollisionGeometry(object_, RigidTransform2d<double>(),
                                    geometry::Rectangle(w_o_, w_o_), mu_);

    pusher_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher_, RigidTransform2d<double>(),
                                    geometry::Circle(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform2d<double>(),
                            pusher_, RigidTransform2d<double>(), k_a_);

    plant.Finalize();

    model_ =
        dynamic_pointer_cast<QuasidynamicPlanarPlant<double>>(plant.Clone());
  }

 protected:
  const Vector2d k_a_ = Vector2d{1.0, 0.9};
  const double m_o_ = 0.01;
  const double I_o_ = 0.0003;
  const double w_a_ = 0.1;
  const double w_o_ = 0.4;
  const double mu_ = 0.05;
  const double dt_ = 0.01;
  BodyIndex object_;
  BodyIndex pusher_;
  std::unique_ptr<QuasidynamicPlanarPlant<double>> model_;
};

TEST_F(QuasidynamicPlanarPlantTest, Construction) {
  EXPECT_EQ(model_->get_actuation_input_port().size(), 2);
  EXPECT_EQ(model_->get_state_output_port().size(), 5);
  EXPECT_EQ(model_->get_contact_force_output_port().size(), 2 * 1);

  const auto context = model_->CreateDefaultContext();
  model_->set_geometry_smoothing(context.get(), 1e-4);
  EXPECT_EQ(model_->get_geometry_smoothing(*context), 1e-4);
}

TEST_F(QuasidynamicPlanarPlantTest, ScalarConversion) {
  std::unique_ptr<System<AutoDiffXd>> model_ad = model_->ToAutoDiffXdMaybe();
  EXPECT_NE(model_ad, nullptr);
  std::unique_ptr<System<double>> model = model_ad->ToScalarType<double>();
  EXPECT_NE(model, nullptr);
  EXPECT_NO_THROW(model_->Clone());
}

TEST_F(QuasidynamicPlanarPlantTest, DynamicsMatrices) {
  const auto context = model_->CreateDefaultContext();
  const Vector5d x{0.0, 0.0, 0.35, 0.0, 0.0};
  const Vector2d u{0.15, 0.01};
  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);

  const MatrixXd& P = model_->EvalQuasidynamicPMatrix(*context);
  const MatrixXd P_expected = Vector5d{k_a_[0], k_a_[1], m_o_ / pow(dt_, 2),
                                       m_o_ / pow(dt_, 2), I_o_ / pow(dt_, 2)}
                                  .asDiagonal();
  EXPECT_TRUE(CompareMatrices(P, P_expected));

  const VectorXd& b = model_->EvalQuasidynamicBVector(*context);
  const Vector5d b_expected{
      -k_a_[0] * u[0], -k_a_[1] * u[1], -m_o_ / pow(dt_, 2) * x[2],
      -m_o_ / pow(dt_, 2) * x[3], -I_o_ / pow(dt_, 2) * x[4]};
  EXPECT_TRUE(CompareMatrices(b, b_expected, 1e-14));

  const MatrixXd& J = model_->EvalContactJacobian(*context);
  Eigen::Matrix<double, 2, 5> J_expected;
  // clang-format off
  J_expected << -1, 0, 1, 0, 0,
                0, -1, 0, 1, -0.2;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(J, J_expected, 1e-8));

  const VectorXd& phi = model_->EvalContactSignedDistance(*context);
  Vector1d phi_expected{0.1};
  EXPECT_TRUE(CompareMatrices(phi, phi_expected, 1e-8));

  const VectorXd& mu = model_->EvalContactFrictionCoefficient(*context);
  const Vector1d mu_expected{mu_};
  EXPECT_TRUE(CompareMatrices(mu, mu_expected, 1e-16));

  const RigidTransform2d<double> X_WB =
      model_->EvalPoseInWorld(*context, object_);
  EXPECT_EQ(X_WB.translation(), x.segment<2>(2));
  EXPECT_EQ(X_WB.angle(), x[4]);
}

TEST_F(QuasidynamicPlanarPlantTest, NonsmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 0.0;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector5d x{0.0, 0.0, 0.35, 0.0, 0.0};
  Vector2d u{0.15, 0.0};

  const auto [xnext, lambda, E] = EvalStep(*model_, context.get(), x, u);

  const Vector5d xnext_expected{0.1004950, 0.0, 0.3504950, 0.0, 0.0};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);
  const Matrix5d P = model_->EvalQuasidynamicPMatrix(*context);
  const Vector5d b = model_->EvalQuasidynamicBVector(*context);
  const Matrix<double, 2, 5> J = model_->EvalContactJacobian(*context);
  const Vector2d lambda_expected =
      J.transpose().completeOrthogonalDecomposition().solve(P * xnext + b);
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-7));

  const Vector5d E_expected = Vector5d::Zero();
  EXPECT_TRUE(CompareMatrices(E, E_expected, 1e-7));

  u = Vector2d{0.15, 0.01};
  const auto [dxnext_dxu, dlambda_dxu, dE_dxu] =
      EvalJacobian(*model_, context.get(), x, u);
  const auto [dxnext_dxu_fd, dlambda_dxu_fd, dE_dxu_fd] =
      ComputeFiniteDiffJacobian(*model_, context.get(), x, u, 1e-6);

  constexpr double kTol = 2e-3;
  EXPECT_TRUE(CompareMatrices(dxnext_dxu, dxnext_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu, dlambda_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dE_dxu, dE_dxu_fd, kTol));
}

TEST_F(QuasidynamicPlanarPlantTest, SmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 1e-2;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector5d x{0.0, 0.0, 0.35, 0.0, 0.0};
  Vector2d u{0.1, 0.0};

  const auto [xnext, lambda, E] = EvalStep(*model_, context.get(), x, u);

  const Vector5d xnext_expected{0.0004963, 0.0, 0.3509950, 0.0, 0.0};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);
  const Matrix<double, 2, 5> J = model_->EvalContactJacobian(*context);
  const double phi = model_->EvalContactSignedDistance(*context)[0];
  const Vector2d nu = J * (xnext - x) + Vector2d{phi, 0.0};
  const Vector2d lambda_expected = kappa /
                                   (pow(nu[0] / mu_, 2) - pow(nu[1], 2)) *
                                   Vector2d{nu[0] / pow(mu_, 2), -nu[1]};
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-16));

  const Matrix5d P = model_->EvalQuasidynamicPMatrix(*context);
  const Vector5d E_expected =
      -P.inverse() * J.transpose() *
      PseudoInverse(nu.asDiagonal().toDenseMatrix() +
                    (nu.asDiagonal() * lambda * lambda.transpose() * 2 / kappa -
                     lambda.asDiagonal().toDenseMatrix()) *
                        J * P.inverse() * J.transpose()) *
      (nu.asDiagonal() * lambda / kappa) * kappa;
  EXPECT_TRUE(CompareMatrices(E, E_expected, 1e-16));

  u = Vector2d{0.2, 0.01};
  const auto [dxnext_dxu, dlambda_dxu, dE_dxu] =
      EvalJacobian(*model_, context.get(), x, u);
  const auto [dxnext_dxu_fd, dlambda_dxu_fd, dE_dxu_fd] =
      ComputeFiniteDiffJacobian(*model_, context.get(), x, u, 1e-4);

  constexpr double kTol = 2e-5;
  EXPECT_TRUE(CompareMatrices(dxnext_dxu, dxnext_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu, dlambda_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dE_dxu, dE_dxu_fd, kTol));
}

class QuasidynamicPlanarPlantCollisionTest
    : public QuasidynamicPlanarPlantTest {
 private:
  void SetUp() override {
    QuasidynamicPlanarPlant<double> plant(dt_);
    object_ = plant.AddRigidBody(m_o_, Vector2d::Zero(), I_o_);
    plant.RegisterCollisionGeometry(object_, RigidTransform2d<double>(),
                                    geometry::Rectangle(w_o_, w_o_), mu_);

    pusher_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher_, RigidTransform2d<double>(),
                                    geometry::Circle(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform2d<double>(),
                            pusher_, RigidTransform2d<double>(), k_a_);

    pusher2_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher2_, RigidTransform2d<double>(),
                                    geometry::Circle(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform2d<double>(),
                            pusher2_, RigidTransform2d<double>(), k_a_);

    plant.Finalize();

    model_ =
        dynamic_pointer_cast<QuasidynamicPlanarPlant<double>>(plant.Clone());
  }

 protected:
  BodyIndex pusher2_;
};

TEST_F(QuasidynamicPlanarPlantCollisionTest,
       CollisionJacobianAndSignedDistance) {
  const auto context = model_->CreateDefaultContext();
  constexpr double kTol = 3e-16;
  {
    const Vector<double, 7> x{0.0, 0.0, 2.0, 0.0, 10.0, 0.0, 0.0};
    model_->SetState(context.get(), x);

    const MatrixXd& J = model_->EvalCollisionJacobian(*context);
    Eigen::Matrix<double, 1, 7> J_expected;
    J_expected << -1, 0, 1, 0, 0, 0, 0;
    // EXPECT_TRUE(CompareMatrices(J, J_expected, kTol));
    unused(J, J_expected, kTol);

    const VectorXd& phi = model_->EvalCollisionSignedDistance(*context);
    Vector1d phi_expected{1.9};
    // EXPECT_TRUE(CompareMatrices(phi, phi_expected, kTol));
    unused(phi, phi_expected, kTol);
  }
  {
    const Vector<double, 7> x{0.0, 0.0, sqrt(3), 1.0, 10.0, 0.0, 0.0};
    model_->SetState(context.get(), x);

    const MatrixXd& J = model_->EvalCollisionJacobian(*context);
    Eigen::Matrix<double, 1, 7> J_expected;
    J_expected << -sqrt(3) / 2, -0.5, sqrt(3) / 2, 0.5, 0, 0, 0;
    // EXPECT_TRUE(CompareMatrices(J, J_expected, kTol));
    unused(J, J_expected, kTol);

    const VectorXd& phi = model_->EvalCollisionSignedDistance(*context);
    Vector1d phi_expected{1.9};
    // EXPECT_TRUE(CompareMatrices(phi, phi_expected, kTol));
    unused(phi, phi_expected, kTol);
  }
}

}  // namespace
}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake
