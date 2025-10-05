#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace solvers {

// Forward declaration.
class Cone;

/** Solves conic program of the form:
      minimimze   ½ xᵀ P x + qᵀ x
      subject to  b - A x ∈ 𝒦
 @param[in] P The P matrix.
 @param[in] q The q vector.
 @param[in] A The A matrix.
 @param[in] b The b vector.
 @param[in] cones Cones for which their cartesian product define 𝒦.
 @param[in] mu_target The target complementarity value μ.
 @param[out] dx_dmu Derivative of primal solution with respect to μ, ∂x/∂μ;
 @param[out] dz_dmu Derivative of dual solution with respect to μ, ∂z/∂μ;
 @returns The primal solution x and dual solution z.
 @throws std::exception if the solver fails.
 @pre The matrices, vectors, and cones have compatible sizes.
 @pre `mu_target >= 0`.
 @tparam_nonsymbolic_scalar */
template <typename T>
std::tuple<VectorX<T>, VectorX<T>> SolveConicProgram(
    const MatrixX<T>& P, const VectorX<T>& q, const MatrixX<T>& A,
    const VectorX<T>& b, const std::vector<Cone>& cones, double mu_target = 0.0,
    VectorX<T>* dx_dmu = nullptr, VectorX<T>* dz_dmu = nullptr);

/** @p Cone represents a cone constraint. */
class Cone {
 public:
  Cone(const Cone& other);
  Cone(Cone&& other);
  Cone& operator=(const Cone& other);
  Cone& operator=(Cone&& other);

  enum class Tag {
    NonnegativeCone,
    SecondOrderCone,
  };

  Tag tag() const;
  int dimension() const;
  int degree() const;

  /** Returns the log-barrier value -log(s), the log(⋅) function is specific to
   the cone type. */
  double CalcLogBarrier(const Eigen::Ref<const VectorX<double>>& s) const;

  /** Returns the gradient of the log-barrier ∇ₛ(-log(s)). */
  VectorX<double> CalcLogBarrierGradient(
      const Eigen::Ref<const VectorX<double>>& s) const;

  /** Returns the hessian of the log-barrier ∇ₛ²(-log(s)). */
  MatrixX<double> CalcLogBarrierHessian(
      const Eigen::Ref<const VectorX<double>>& s) const;

  /** Returns the identity element in Jordan algebra. */
  VectorX<double> JordanIdentity() const;

  /** Computes the Jordan left multiplicative matrix L(s), so that
   L(s) z = s ∘ z. */
  template <typename T>
  MatrixX<T> JordanLeftMul(const Eigen::Ref<const VectorX<T>>& s) const;

  /** Computes the Jordan left multiplicative inverse L(s)⁻¹, so that
   L(s)⁻¹ (s ∘ z) = z. */
  template <typename T>
  MatrixX<T> JordanLeftMulInverse(const Eigen::Ref<const VectorX<T>>& s) const;

 protected:
  class Impl;

  explicit Cone(std::unique_ptr<Impl> impl);

 private:
  struct ImplDeleter {
    void operator()(Impl*);
  };

  std::unique_ptr<Impl, ImplDeleter> impl_;
};

/** A non-negative cone describes the set
    { s ∈ ℝⁿ : sᵢ ≥ 0, ∀i=1,⋯,n } */
class NonnegativeCone final : public Cone {
 public:
  explicit NonnegativeCone(int dimension);

 private:
  class Impl;
};

/** A second-order cone describes the set
    { (s₀, s₁) ∈ ℝⁿ : s₀ ≥ ∣∣s₁∣∣₂ } */
class SecondOrderCone final : public Cone {
 public:
  explicit SecondOrderCone(int dimension);

 private:
  class Impl;
};

}  // namespace solvers
}  // namespace drake
