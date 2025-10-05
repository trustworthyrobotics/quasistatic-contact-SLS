#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"

#include <memory>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace quasidynamic {

template <typename T>
class DummyModel final : public QuasidynamicDifferentiableContactModel<T> {
 public:
  DummyModel()
      : QuasidynamicDifferentiableContactModel<T>(
            systems::SystemTypeTag<DummyModel>{}, /* kappa */ 1e-10) {
    this->DeclareStateAndPorts(
        /* x */ Vector2<T>::Zero(), /* num_inputs */ 1,
        /* num_contacts */ 1, /* contact_dim */ 1, /* dt */ 0.1);
  }

  /* Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  template <typename U>
  explicit DummyModel(const DummyModel<U>&) : DummyModel<T>() {}

  const MatrixX<T>& EvalQuasidynamicPMatrix(
      const systems::Context<T>&) const override {
    DRAKE_UNREACHABLE();
  }

  const VectorX<T>& EvalQuasidynamicBVector(
      const systems::Context<T>&) const override {
    DRAKE_UNREACHABLE();
  }

  const MatrixX<T>& EvalContactJacobian(
      const systems::Context<T>&) const override {
    DRAKE_UNREACHABLE();
  }

  const VectorX<T>& EvalContactSignedDistance(
      const systems::Context<T>&) const override {
    DRAKE_UNREACHABLE();
  }

  const VectorX<double>& EvalContactFrictionCoefficient(
      const systems::Context<T>&) const override {
    DRAKE_UNREACHABLE();
  }
};

}  // namespace quasidynamic
}  // namespace multibody

namespace systems {
namespace scalar_conversion {
template <>
struct Traits<multibody::quasidynamic::DummyModel> : public NonSymbolicTraits {
};
}  // namespace scalar_conversion
}  // namespace systems

namespace multibody {
namespace quasidynamic {
namespace {

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Vector;
using Eigen::VectorXd;
using systems::Context;
using systems::System;

class QuasidynamicDifferentiableContactModelTest : public ::testing::Test {
 private:
  void SetUp() override { model_ = std::make_unique<DummyModel<double>>(); }

 protected:
  std::unique_ptr<QuasidynamicDifferentiableContactModel<double>> model_;
};

TEST_F(QuasidynamicDifferentiableContactModelTest, SmoothingParameter) {
  auto context = model_->CreateDefaultContext();
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), 1e-10);

  model_->set_dynamics_smoothing(context.get(), 1e-3);
  EXPECT_EQ(model_->get_dynamics_smoothing(*context), 1e-3);
}

TEST_F(QuasidynamicDifferentiableContactModelTest, Ports) {
  EXPECT_EQ(&model_->get_actuation_input_port(), &model_->GetInputPort("u"));
  EXPECT_EQ(&model_->get_state_output_port(), &model_->GetOutputPort("x"));
  EXPECT_EQ(&model_->get_contact_force_output_port(),
            &model_->GetOutputPort("lambda"));

  EXPECT_EQ(model_->get_actuation_input_port().size(), 1);
  EXPECT_EQ(model_->get_state_output_port().size(), 2);
  EXPECT_EQ(model_->get_contact_force_output_port().size(), 1);
}

TEST_F(QuasidynamicDifferentiableContactModelTest, State) {
  std::unique_ptr<Context<double>> context = model_->CreateDefaultContext();

  const VectorXd x = Vector<double, 2>{1.0, 2.0};
  model_->SetState(context.get(), x);

  EXPECT_EQ(model_->GetState(*context), x);
  EXPECT_EQ(model_->get_state_output_port().Eval(*context), x);
}

TEST_F(QuasidynamicDifferentiableContactModelTest, PreUpdate) {
  std::unique_ptr<Context<double>> context = model_->CreateDefaultContext();

  const VectorXd x = Vector<double, 2>{1.0, 2.0};
  const VectorXd u = Vector<double, 1>{3.0};
  model_->SetState(context.get(), x);
  model_->get_actuation_input_port().FixValue(context.get(), u);

  EXPECT_EQ(model_->get_state_output_port().Eval(*context), x);
  EXPECT_EQ(model_->get_contact_force_output_port().Eval(*context),
            (Vector<double, 1>{0.0}));
  EXPECT_EQ(model_->GetDisturbanceMatrix(*context),
            (Matrix<double, 2, 1>::Zero()));
}

}  // namespace
}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake
