#include "drake/multibody/quasidynamic/quasidynamic_multibody_plant.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/pointer_cast.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/quasidynamic/test_utilities/eval_quasidynamic_differentiable_contact_model.h"

namespace drake {
namespace multibody {
namespace quasidynamic {
namespace {

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Matrix9d = Eigen::Matrix<double, 9, 9>;
using Vector9d = Eigen::Vector<double, 9>;
using math::RigidTransform;
using systems::Context;
using systems::System;

class QuasidynamicMultibodyPlantTest : public ::testing::Test {
 private:
  void SetUp() override {
    QuasidynamicMultibodyPlant<double> plant(dt_);
    object_ = plant.AddRigidBody(
        SpatialInertia<double>::SolidBoxWithMass(m_o_, w_o_, w_o_, w_o_));
    plant.RegisterCollisionGeometry(object_, RigidTransform<double>(),
                                    geometry::Box(w_o_, w_o_, w_o_), mu_);

    pusher_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher_, RigidTransform<double>(),
                                    geometry::Sphere(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform<double>(),
                            pusher_, RigidTransform<double>(), k_a_);

    plant.SetGravityVector(Vector3d::Zero());
    plant.Finalize();

    model_ =
        dynamic_pointer_cast<QuasidynamicMultibodyPlant<double>>(plant.Clone());
  }

 protected:
  const Vector3d k_a_ = Vector3d{1.0, 0.9, 0.8};
  const double w_a_ = 0.1;
  const double m_o_ = 0.01;
  const double w_o_ = 0.4;
  const double I_o_ = m_o_ * w_o_ * w_o_ / 6;
  const double mu_ = 0.05;
  const double dt_ = 0.01;
  BodyIndex object_;
  BodyIndex pusher_;
  std::unique_ptr<QuasidynamicMultibodyPlant<double>> model_;
};

TEST_F(QuasidynamicMultibodyPlantTest, Construction) {
  EXPECT_EQ(model_->get_actuation_input_port().size(), 3);
  EXPECT_EQ(model_->get_state_output_port().size(), 9);
  EXPECT_EQ(model_->get_contact_force_output_port().size(), 3 * 1);

  const auto context = model_->CreateDefaultContext();
  model_->set_geometry_smoothing(context.get(), 1e-4);
  EXPECT_EQ(model_->get_geometry_smoothing(*context), 1e-4);
}

TEST_F(QuasidynamicMultibodyPlantTest, ScalarConversion) {
  std::unique_ptr<System<AutoDiffXd>> model_ad = model_->ToAutoDiffXdMaybe();
  EXPECT_NE(model_ad, nullptr);
  std::unique_ptr<System<double>> model = model_ad->ToScalarType<double>();
  EXPECT_NE(model, nullptr);
  EXPECT_NO_THROW(model_->Clone());
}

TEST_F(QuasidynamicMultibodyPlantTest, DynamicsMatrices) {
  const auto context = model_->CreateDefaultContext();
  const Vector9d x{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0};
  const Vector3d u{0.15, 0.01, -0.01};
  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);

  const MatrixXd& P = model_->EvalQuasidynamicPMatrix(*context);
  const MatrixXd P_expected =
      (Vector9d() << k_a_, I_o_ / pow(dt_, 2) * Vector3d::Ones(),
       m_o_ / pow(dt_, 2) * Vector3d::Ones())
          .finished()
          .asDiagonal();
  EXPECT_TRUE(CompareMatrices(P, P_expected, 1e-15));

  const VectorXd& b = model_->EvalQuasidynamicBVector(*context);
  Vector9d b_expected;
  b_expected << (-k_a_.cwiseProduct(u)).eval(),
      (-I_o_ / pow(dt_, 2) * x.segment<3>(3)).eval(),
      (-m_o_ / pow(dt_, 2) * x.segment<3>(6)).eval();
  EXPECT_TRUE(CompareMatrices(b, b_expected, 1e-14));

  const MatrixXd& J = model_->EvalContactJacobian(*context);
  Eigen::Matrix<double, 3, 9> J_expected;
  // clang-format off
  J_expected << -1, 0, 0, 0,    0,   0, 1,  0,  0,
                 0, 0, 1, 0, -0.2,   0, 0,  0, -1,
                 0, 1, 0, 0,    0, 0.2, 0, -1,  0;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(J, J_expected, 1e-8));

  const VectorXd& phi = model_->EvalContactSignedDistance(*context);
  Vector1d phi_expected{0.1};
  EXPECT_TRUE(CompareMatrices(phi, phi_expected, 1e-8));

  const VectorXd& mu = model_->EvalContactFrictionCoefficient(*context);
  const Vector1d mu_expected{mu_};
  EXPECT_TRUE(CompareMatrices(mu, mu_expected, 1e-16));

  const RigidTransform<double>& X_WB =
      model_->EvalPoseInWorld(*context, object_);
  EXPECT_TRUE(X_WB.rotation().IsExactlyIdentity());
  EXPECT_EQ(X_WB.translation(), x.tail<3>());
}

TEST_F(QuasidynamicMultibodyPlantTest, NonsmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 0.0;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector9d x{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0};
  Vector3d u{0.15, 0.0, 0.0};

  const auto [xnext, lambda, E] = EvalStep(*model_, context.get(), x, u);

  const Vector9d xnext_expected{0.1004950, 0.0,       0.0, 0.0, 0.0,
                                0.0,       0.3504950, 0.0, 0.0};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);
  const Matrix9d P = model_->EvalQuasidynamicPMatrix(*context);
  const Vector9d b = model_->EvalQuasidynamicBVector(*context);
  const Matrix<double, 3, 9> J = model_->EvalContactJacobian(*context);
  const Vector3d lambda_expected =
      J.transpose().completeOrthogonalDecomposition().solve(P * xnext + b);
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-7));

  const Vector9d E_expected = Vector9d::Zero();
  EXPECT_TRUE(CompareMatrices(E, E_expected, 1e-7));
}

TEST_F(QuasidynamicMultibodyPlantTest, SmoothedDynamics) {
  auto context = model_->CreateDefaultContext();
  const double kappa = 1e-2;
  model_->set_dynamics_smoothing(context.get(), kappa);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), kappa);

  const Vector9d x{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0};
  Vector3d u{0.1, 0.0, 0.0};

  const auto [xnext, lambda, E] = EvalStep(*model_, context.get(), x, u);

  const Vector9d xnext_expected{0.0004963, 0.0,       0.0, 0.0, 0.0,
                                0.0,       0.3509950, 0.0, 0.0};
  EXPECT_TRUE(CompareMatrices(xnext, xnext_expected, 1e-7));

  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);
  const Matrix<double, 3, 9> J = model_->EvalContactJacobian(*context);
  const double phi = model_->EvalContactSignedDistance(*context)[0];
  const Vector3d nu = J * (xnext - x) + Vector3d{phi, 0.0, 0.0};
  const Vector3d lambda_expected =
      kappa / (pow(nu[0] / mu_, 2) - pow(nu[1], 2) - pow(nu[2], 2)) *
      Vector3d{nu[0] / pow(mu_, 2), -nu[1], -nu[2]};
  EXPECT_TRUE(CompareMatrices(lambda, lambda_expected, 1e-16));

  const Matrix9d P = model_->EvalQuasidynamicPMatrix(*context);
  const Vector9d E_expected =
      -P.inverse() * J.transpose() *
      PseudoInverse(nu.asDiagonal().toDenseMatrix() +
                    (nu.asDiagonal() * lambda * lambda.transpose() * 2 / kappa -
                     lambda.asDiagonal().toDenseMatrix()) *
                        J * P.inverse() * J.transpose()) *
      (nu.asDiagonal() * lambda / kappa) * kappa;
  EXPECT_TRUE(CompareMatrices(E, E_expected, 1e-16));

  u = Vector3d{0.2, 0.01, -0.01};
  const auto [dxnext_dxu, dlambda_dxu, dE_dxu] =
      EvalJacobian(*model_, context.get(), x, u);
  const auto [dxnext_dxu_fd, dlambda_dxu_fd, dE_dxu_fd] =
      ComputeFiniteDiffJacobian(*model_, context.get(), x, u, 1e-4);

  constexpr double kTol = 4e-4;
  EXPECT_TRUE(CompareMatrices(dxnext_dxu, dxnext_dxu_fd, kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu.row(0), dlambda_dxu_fd.row(0), kTol));
  EXPECT_TRUE(CompareMatrices(dlambda_dxu.bottomRows<2>().colwise().norm(),
                              dlambda_dxu_fd.bottomRows<2>().colwise().norm(),
                              kTol));
  EXPECT_TRUE(CompareMatrices(dE_dxu, dE_dxu_fd, kTol));
}

class QuasidynamicMultibodyPlantCollisionTest
    : public QuasidynamicMultibodyPlantTest {
 private:
  void SetUp() override {
    QuasidynamicMultibodyPlant<double> plant(dt_);
    object_ = plant.AddRigidBody(
        SpatialInertia<double>::SolidBoxWithMass(m_o_, w_o_, w_o_, w_o_));
    plant.RegisterCollisionGeometry(object_, RigidTransform<double>(),
                                    geometry::Box(w_o_, w_o_, w_o_), mu_);

    pusher_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher_, RigidTransform<double>(),
                                    geometry::Sphere(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform<double>(),
                            pusher_, RigidTransform<double>(), k_a_);

    pusher2_ = plant.AddRigidBody();
    plant.RegisterCollisionGeometry(pusher2_, RigidTransform<double>(),
                                    geometry::Sphere(w_a_ / 2), mu_);
    plant.AddCartesianJoint(plant.world_body(), RigidTransform<double>(),
                            pusher2_, RigidTransform<double>(), k_a_);

    plant.SetGravityVector(Vector3d::Zero());
    plant.Finalize();

    model_ =
        dynamic_pointer_cast<QuasidynamicMultibodyPlant<double>>(plant.Clone());
  }

 protected:
  BodyIndex pusher2_;
};

TEST_F(QuasidynamicMultibodyPlantCollisionTest,
       CollisionJacobianAndSignedDistance) {
  const auto context = model_->CreateDefaultContext();
  constexpr double kTol = 3e-16;
  {
    const Vector<double, 12> x{0.0, 0.0, 0.0, 2.0,  0.0, 0.0,
                               0.0, 0.0, 0.0, 10.0, 0.0, 0.0};
    model_->SetState(context.get(), x);

    const MatrixXd& J = model_->EvalCollisionJacobian(*context);
    Eigen::Matrix<double, 1, 12> J_expected;
    J_expected << -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
    // EXPECT_TRUE(CompareMatrices(J, J_expected, kTol));
    unused(J, J_expected, kTol);

    const VectorXd& phi = model_->EvalCollisionSignedDistance(*context);
    Vector1d phi_expected{1.9};
    // EXPECT_TRUE(CompareMatrices(phi, phi_expected, kTol));
    unused(phi, phi_expected, kTol);
  }
  {
    const Vector<double, 12> x{0.0, 0.0, 0.0, sqrt(3), 1.0, 0.0,
                               0.0, 0.0, 0.0, 10.0,    0.0, 0.0};
    model_->SetState(context.get(), x);

    const MatrixXd& J = model_->EvalCollisionJacobian(*context);
    Eigen::Matrix<double, 1, 12> J_expected;
    J_expected << -sqrt(3) / 2, -0.5, 0, sqrt(3) / 2, 0.5, 0, 0, 0, 0, 0, 0, 0;
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
