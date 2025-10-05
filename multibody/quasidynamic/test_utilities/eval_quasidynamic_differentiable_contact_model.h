#pragma once

#include <tuple>

#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"

namespace drake {
namespace multibody {
namespace quasidynamic {

/** Evaluates the next step state and contact forces.
 @param model The quasidynamic differentiable contact model.
 @param context Context to provide model parameters.
 @param x State at the current time step.
 @param u Input at the current time step.
 @return The state `xnext` and contact forces `lambda` at the next time step,
 and the disturbance matrix E.
 @pre `x` and `u` have the correct sizes.
 @tparam_double_only */
template <typename T>
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd> EvalStep(
    const QuasidynamicDifferentiableContactModel<T>& model,
    systems::Context<T>* context, const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u);

/** Evaluates the jacobians.
 @param model The quasidynamic differentiable contact model.
 @param context Context to provide model parameters.
 @param x State at the current time step.
 @param u Input at the current time step.
 @returns `dxnext_dxu`, `dlambda_dxu`, and `dE_dxu`.
 @pre `x` and `u` have the correct sizes.
 @tparam_nonsymbolic_scalar */
template <typename T>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> EvalJacobian(
    const QuasidynamicDifferentiableContactModel<T>& model,
    systems::Context<T>* context, const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u);

/** Computes the jacobians using finite difference.
 @param model The quasidynamic differentiable contact model.
 @param context Context to provide model parameters.
 @param x State at the current time step.
 @param u Input at the current time step.
 @param epsilon The pertubation amount.
 @returns `dxnext_dxu`, `dlambda_dxu`, and `dE_dxu`.
 @pre `x` and `u` have the correct sizes.
 @tparam_double_only */
template <typename T>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
ComputeFiniteDiffJacobian(
    const QuasidynamicDifferentiableContactModel<T>& model,
    systems::Context<T>* context, const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u, double epsilon = 1e-6);

/** Computes the pseudo inverse of a matrix. */
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& A, double rcond = 1e-12);

}  // namespace quasidynamic
}  // namespace multibody
}  // namespace drake
