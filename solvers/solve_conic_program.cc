#include "drake/solvers/solve_conic_program.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include <clarabel.hpp>
#include <fmt/format.h>

#include "drake/common/autodiff.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/gradient_util.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace solvers {
namespace {

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/* Resize gradient matrices so they have the same number of columns. */
template <typename... Matrices>
void ResizeGradients(Matrices... mats) {
  const std::vector<MatrixXd*> mat_list = {mats...};
  for (auto mat : mat_list) DRAKE_THROW_UNLESS(mat != nullptr);

  int num_derivs = 0;
  for (auto mat : mat_list) {
    num_derivs = std::max(num_derivs, static_cast<int>(mat->cols()));
  }

  for (auto mat : mat_list) {
    DRAKE_THROW_UNLESS(mat->cols() == 0 || mat->cols() == num_derivs);
    if (mat->cols() == 0) {
      mat->resize(mat->rows(), num_derivs);
      mat->setZero();
    }
  }
}

/* Computes (dA)x, i.e., d(Ax) - Adx. */
template <int ColsAtCompileTime>
MatrixXd TensorProduct(
    const MatrixXd& dA,
    const Matrix<double, Eigen::Dynamic, ColsAtCompileTime>& x) {
  return math::matGradMult(dA, x);
}

/* Computes (dAᵀ)z, i.e., d(Aᵀz) - Aᵀdz. */
MatrixXd TransposeTensorProduct(const MatrixXd& dA, const VectorXd& z) {
  return math::matGradMult(math::transposeGrad(dA, z.size()), z);
}

/* Solves linear problem P x = q, where P is positive-definite. */
template <int ColsAtCompileTime>
Matrix<double, Eigen::Dynamic, ColsAtCompileTime> LinearSolve(
    const MatrixXd& P,
    const Matrix<double, Eigen::Dynamic, ColsAtCompileTime>& q) {
  Eigen::LDLT<MatrixXd> solver(P);
  DRAKE_DEMAND(solver.info() == Eigen::Success);
  return solver.solve(q);
}

template <int ColsAtCompileTime>
Matrix<AutoDiffXd, Eigen::Dynamic, ColsAtCompileTime> LinearSolve(
    const MatrixX<AutoDiffXd>& P_ad,
    const Matrix<AutoDiffXd, Eigen::Dynamic, ColsAtCompileTime>& q_ad) {
  static_assert(ColsAtCompileTime == 1);
  const MatrixXd P = math::ExtractValue(P_ad);
  Eigen::LDLT<MatrixXd> solver(P);
  DRAKE_DEMAND(solver.info() == Eigen::Success);

  const Matrix<double, Eigen::Dynamic, ColsAtCompileTime> q =
      math::ExtractValue(q_ad);
  const Matrix<double, Eigen::Dynamic, ColsAtCompileTime> x = solver.solve(q);

  MatrixXd dP = math::ExtractGradient(P_ad);
  MatrixXd dq = math::ExtractGradient(q_ad);
  ResizeGradients(&dP, &dq);
  const MatrixXd dx = solver.solve(dq - TensorProduct(dP, x));

  const Matrix<AutoDiffXd, Eigen::Dynamic, ColsAtCompileTime> x_ad =
      math::InitializeAutoDiff(x, dx);
  return x_ad;
}

/* Solves conic program of the form:
      minimimze   ½ xᵀ P x + qᵀ x
      subject to  b - A x ∈ 𝒦. */
class ConicProgramSolver {
 public:
  ConicProgramSolver(const MatrixXd& P, const VectorXd& q, const MatrixXd& A,
                     const VectorXd& b, const std::vector<Cone>& cones)
      : P_(P), q_(q), A_(A), b_(b), cones_(cones) {
    DRAKE_THROW_UNLESS(P.rows() == P.cols());
    DRAKE_THROW_UNLESS(P.rows() == q.size());
    DRAKE_THROW_UNLESS(A.rows() == b.size());
    DRAKE_THROW_UNLESS(A.cols() == P.cols());
    DRAKE_THROW_UNLESS(std::accumulate(cones.cbegin(), cones.cend(), 0,
                                       [](int sum, const Cone& cone) {
                                         return sum + cone.dimension();
                                       }) == A.rows());

    cones_dim_start_.resize(cones.size());
    cones_dim_start_[0] = 0;
    for (int i = 0; i < ssize(cones) - 1; ++i) {
      cones_dim_start_[i + 1] = cones_dim_start_[i] + cones[i].dimension();
    }
  }

  std::tuple<VectorXd, VectorXd> Solve(double mu_target) {
    DRAKE_THROW_UNLESS(mu_target >= 0.0);

    if (mu_target < 1e-8) return ClarabelSolve();

    const auto [x_warm, _] = ClarabelSolve(mu_target);
    return LogbarrierSolve(mu_target, std::move(x_warm));
  }

 private:
  struct ClarabelCallbackData {
    const int degrees;
    const double mu_approx;
  };

  static int ClarabelTerminationCallback(clarabel::DefaultInfo<double>& info,
                                         void* callback_data) {
    const ClarabelCallbackData& data =
        *static_cast<ClarabelCallbackData*>(callback_data);
    const double mu = info.gap_abs / data.degrees;
    if (mu <= data.mu_approx) return true;
    return false;
  }

  std::tuple<VectorXd, VectorXd> ClarabelSolve(double mu_approx = 0.0) {
    DRAKE_THROW_UNLESS(mu_approx >= 0.0);

    Eigen::SparseMatrix<double> P = P_.sparseView();
    P.makeCompressed();
    Eigen::SparseMatrix<double> A = A_.sparseView();
    A.makeCompressed();
    VectorXd q = q_;
    VectorXd b = b_;

    const clarabel::DefaultSettings<double> settings =
        clarabel::DefaultSettingsBuilder<double>::default_settings()
            .verbose(false)
            .chordal_decomposition_enable(false)
            .presolve_enable(false)
            .build();
    clarabel::DefaultSolver<double> solver(P, q, A, b, ClarabelCones(),
                                           settings);

    const int degrees = std::accumulate(cones_.cbegin(), cones_.cend(), 0,
                                        [](int sum, const Cone& cone) {
                                          return sum + cone.degree();
                                        });
    ClarabelCallbackData data{.degrees = degrees, .mu_approx = mu_approx};
    solver.set_termination_callback(&ClarabelTerminationCallback, &data);

    solver.solve();
    const clarabel::DefaultSolution<double> solution = solver.solution();

    const clarabel::SolverStatus status = solution.status;
    if (!(status == clarabel::SolverStatus::Solved ||
          status == clarabel::SolverStatus::AlmostSolved ||
          status == clarabel::SolverStatus::CallbackTerminated)) {
      throw std::runtime_error(
          fmt::format("SolveConicProgram terminated with status {}",
                      ClarabelStatusToString(status)));
    }

    return {solution.x, solution.z};
  }

  std::vector<clarabel::SupportedConeT<double>> ClarabelCones() const {
    std::vector<clarabel::SupportedConeT<double>> result;
    result.reserve(cones_.size());
    for (const Cone& cone : cones_) {
      switch (cone.tag()) {
        case Cone::Tag::NonnegativeCone: {
          result.push_back(clarabel::NonnegativeConeT(cone.dimension()));
          break;
        }
        case Cone::Tag::SecondOrderCone: {
          result.push_back(clarabel::SecondOrderConeT(cone.dimension()));
          break;
        }
        default:
          DRAKE_UNREACHABLE();
      }
    }
    return result;
  }

  static std::string ClarabelStatusToString(clarabel::SolverStatus status) {
    switch (status) {
      case clarabel::SolverStatus::Unsolved:
        return "Unsolved";
      case clarabel::SolverStatus::Solved:
        return "Solved";
      case clarabel::SolverStatus::PrimalInfeasible:
        return "PrimalInfeasible";
      case clarabel::SolverStatus::DualInfeasible:
        return "DualInfeasible";
      case clarabel::SolverStatus::AlmostSolved:
        return "AlmostSolved";
      case clarabel::SolverStatus::AlmostPrimalInfeasible:
        return "AlmostPrimalInfeasible";
      case clarabel::SolverStatus::AlmostDualInfeasible:
        return "AlmostDualInfeasible";
      case clarabel::SolverStatus::MaxIterations:
        return "MaxIterations";
      case clarabel::SolverStatus::MaxTime:
        return "MaxTime";
      case clarabel::SolverStatus::NumericalError:
        return "NumericalError";
      case clarabel::SolverStatus::InsufficientProgress:
        return "InsufficientProgress";
      case clarabel::SolverStatus::CallbackTerminated:
        return "CallbackTerminated";
      default:
        DRAKE_UNREACHABLE();
    }
  }

  std::tuple<VectorXd, VectorXd> LogbarrierSolve(double mu_target,
                                                 VectorXd x_init,
                                                 int max_iters = 200,
                                                 double tol = 1e-10) const {
    DRAKE_THROW_UNLESS(mu_target > 0.0);
    const double mu = mu_target;

    VectorXd x = std::move(x_init);
    if (!std::isfinite(LogbarrierObjective(mu, x))) {
      x = FindFeasibleX(x);
      DRAKE_DEMAND(std::isfinite(LogbarrierObjective(mu, x)));
    }

    for (int iter = 0; iter < max_iters; ++iter) {
      // Compute gradient and hessian.
      VectorXd g = P_ * x + q_;
      MatrixXd H = P_;
      for (int i = 0; i < ssize(cones_); ++i) {
        const auto [A, b] = GetConeBlock(i);
        const VectorXd s = b - A * x;
        g += mu * (-A.transpose() * cones_[i].CalcLogBarrierGradient(s));
        H += mu * (A.transpose() * cones_[i].CalcLogBarrierHessian(s) * A);
      }

      // Check convergence.
      if (g.norm() < tol) break;

      // Compute Newton step: H * dx = -g.
      const VectorXd dx = LinearSolve(H, (-g).eval());

      // Armijo backtracking line search.
      double t = 1.0;
      const double alpha = 0.01;
      const double beta = 0.5;
      double f = LogbarrierObjective(mu, x);
      while (true) {
        const VectorXd x_trial = x + t * dx;
        const double f_trial = LogbarrierObjective(mu, x_trial);
        if (f_trial < f + alpha * t * g.dot(dx)) break;

        t *= beta;
        if (t < 1e-20) break;
      }
      if (t < 1e-20) break;

      // Update x.
      x += t * dx;
    }

    VectorXd z(A_.rows());
    for (int i = 0; i < ssize(cones_); ++i) {
      const int start = cones_dim_start_[i];
      const int size = cones_[i].dimension();
      const auto [A, b] = GetConeBlock(i);
      z.middleRows(start, size) =
          -mu * cones_[i].CalcLogBarrierGradient(b - A * x);
    }

    return {x, z};
  }

  double LogbarrierObjective(double mu, const VectorXd& x) const {
    DRAKE_THROW_UNLESS(mu > 0.0);
    DRAKE_THROW_UNLESS(x.size() == P_.cols());

    double value = 0.5 * x.dot(P_ * x) + q_.dot(x);
    for (int i = 0; i < ssize(cones_); ++i) {
      const auto [A, b] = GetConeBlock(i);
      value += mu * cones_[i].CalcLogBarrier(b - A * x);
    }
    return value;
  }

  VectorXd FindFeasibleX(const VectorXd& x_init, double epsilon = 1e-6) const {
    DRAKE_THROW_UNLESS(epsilon > 0);
    MathematicalProgram prog;
    auto x_var = prog.NewContinuousVariables(x_init.size());

    for (int i = 0; i < ssize(cones_); ++i) {
      const Cone& cone = cones_[i];
      const auto [A, b] = GetConeBlock(i);
      switch (cone.tag()) {
        case Cone::Tag::NonnegativeCone: {
          const auto lb = VectorXd::Constant(
              cone.dimension(), -std::numeric_limits<double>::infinity());
          // s = b - Ax, s - ε⋅1 ≥ 0.
          prog.AddLinearConstraint(A, lb, b.array() - epsilon, x_var);
          break;
        }
        case Cone::Tag::SecondOrderCone: {
          // s = b - A x, s₀ - ε ≥ √(s₁² + s₂² + ⋯).
          VectorXd b_prime = b;
          b_prime[0] -= epsilon;
          prog.AddLorentzConeConstraint(-A, b_prime, x_var);
          break;
        }
        default:
          DRAKE_UNREACHABLE();
      }
    }

    const MathematicalProgramResult result = solvers::Solve(prog, x_init);
    DRAKE_DEMAND(result.is_success());
    return result.get_x_val();
  }

  std::tuple<Eigen::Ref<const MatrixXd>, Eigen::Ref<const VectorXd>>
  GetConeBlock(int i) const {
    DRAKE_THROW_UNLESS(0 <= i && i < ssize(cones_));
    const int start = cones_dim_start_[i];
    const int size = cones_[i].dimension();
    return {A_.middleRows(start, size), b_.middleRows(start, size)};
  }

  const MatrixXd& P_;
  const VectorXd& q_;
  const MatrixXd& A_;
  const VectorXd& b_;
  const std::vector<Cone>& cones_;
  std::vector<int> cones_dim_start_;
};

/* Solves linear system of the form:
      P Δx + Aᵀ Δz = bₓ
      A Δx + Δs = bₛ
      s ∘ Δz + z ∘ Δs = b_z. */
template <typename T>
class ConicProgramKkt {
 public:
  ConicProgramKkt(const MatrixX<T>& P, const MatrixX<T>& A,
                  const std::vector<Cone>& cones, const VectorX<T>& s,
                  const VectorX<T>& z)
      : P_(P), A_(A) {
    DRAKE_THROW_UNLESS(P.rows() == P.cols());
    DRAKE_THROW_UNLESS(A.cols() == P.cols());
    DRAKE_THROW_UNLESS(std::accumulate(cones.cbegin(), cones.cend(), 0,
                                       [](int sum, const Cone& cone) {
                                         return sum + cone.dimension();
                                       }) == A.rows());
    DRAKE_THROW_UNLESS(s.size() == A.rows());
    DRAKE_THROW_UNLESS(z.size() == A.rows());

    Ls_.resize(s.size(), s.size());
    Ls_.setZero();
    invLs_.resize(s.size(), s.size());
    invLs_.setZero();
    Lz_.resize(z.size(), z.size());
    Lz_.setZero();
    invLz_.resize(z.size(), z.size());
    invLz_.setZero();

    int offset = 0;
    for (const Cone& cone : cones) {
      const int dim = cone.dimension();

      const auto s_k = s.segment(offset, dim);
      Ls_.block(offset, offset, dim, dim) = cone.JordanLeftMul<T>(s_k);
      invLs_.block(offset, offset, dim, dim) =
          cone.JordanLeftMulInverse<T>(s_k);

      const auto z_k = z.segment(offset, dim);
      Lz_.block(offset, offset, dim, dim) = cone.JordanLeftMul<T>(z_k);
      invLz_.block(offset, offset, dim, dim) =
          cone.JordanLeftMulInverse<T>(z_k);

      offset += dim;
    }
  }

  template <int ColsAtCompileTime>
  std::tuple<Matrix<T, Eigen::Dynamic, ColsAtCompileTime>,
             Matrix<T, Eigen::Dynamic, ColsAtCompileTime>,
             Matrix<T, Eigen::Dynamic, ColsAtCompileTime>>
  Solve(const Matrix<T, Eigen::Dynamic, ColsAtCompileTime>& bx,
        const Matrix<T, Eigen::Dynamic, ColsAtCompileTime>& bs,
        const Matrix<T, Eigen::Dynamic, ColsAtCompileTime>& bz) {
    using MatrixType = Matrix<T, Eigen::Dynamic, ColsAtCompileTime>;
    const MatrixType bsz = bs - invLz_ * bz;
    const MatrixType dx = LinearSolve<ColsAtCompileTime>(
        (P_ + A_.transpose() * invLs_ * Lz_ * A_).eval(),
        (bx + A_.transpose() * invLs_ * Lz_ * bsz).eval());
    const MatrixType ds = bs - A_ * dx;
    const MatrixType dz = invLs_ * (bz - Lz_ * ds);
    return {dx, ds, dz};
  }

 private:
  const MatrixX<T>& P_;
  const MatrixX<T>& A_;
  MatrixX<T> Ls_;
  MatrixX<T> invLs_;
  MatrixX<T> Lz_;
  MatrixX<T> invLz_;
};

template <typename T>
std::tuple<VectorX<T>, VectorX<T>, VectorX<T>> ComputeDiffWrtMu(
    const MatrixX<T>& P, const MatrixX<T>& A, const std::vector<Cone>& cones,
    const VectorX<T>& s, const VectorX<T>& z) {
  ConicProgramKkt kkt(P, A, cones, s, z);

  VectorX<T> e(A.rows());
  int offset = 0;
  for (const Cone& cone : cones) {
    const int dim = cone.dimension();
    e.segment(offset, dim) = cone.JordanIdentity();
    offset += dim;
  }

  const VectorX<T> bx = VectorX<T>::Zero(P.rows());
  const VectorX<T> bs = VectorX<T>::Zero(A.rows());

  return kkt.Solve(bx, bs, e);
}

}  // namespace

/* Function SolveConicProgram<double>. */
template <>
std::tuple<VectorXd, VectorXd> SolveConicProgram<double>(
    const MatrixXd& P, const VectorXd& q, const MatrixXd& A, const VectorXd& b,
    const std::vector<Cone>& cones, double mu_target, VectorXd* dx_dmu_out,
    VectorXd* dz_dmu_out) {
  ConicProgramSolver solver(P, q, A, b, cones);
  const auto [x, z] = solver.Solve(mu_target);

  if (dx_dmu_out != nullptr || dz_dmu_out != nullptr) {
    const VectorXd s = b - A * x;
    const auto [dx_dmu, ds_dmu, dz_dmu] = ComputeDiffWrtMu(P, A, cones, s, z);
    if (dx_dmu_out != nullptr) *dx_dmu_out = dx_dmu;
    if (dz_dmu_out != nullptr) *dz_dmu_out = dz_dmu;
  }

  return {x, z};
}

/* Function SolveConicProgram<AutoDiffXd>. */
template <>
std::tuple<VectorX<AutoDiffXd>, VectorX<AutoDiffXd>>
SolveConicProgram<AutoDiffXd>(const MatrixX<AutoDiffXd>& P_ad,
                              const VectorX<AutoDiffXd>& q_ad,
                              const MatrixX<AutoDiffXd>& A_ad,
                              const VectorX<AutoDiffXd>& b_ad,
                              const std::vector<Cone>& cones, double mu_target,
                              VectorX<AutoDiffXd>* dx_dmu_out,
                              VectorX<AutoDiffXd>* dz_dmu_out) {
  const MatrixXd P = math::ExtractValue(P_ad);
  const VectorXd q = math::ExtractValue(q_ad);
  const MatrixXd A = math::ExtractValue(A_ad);
  const VectorXd b = math::ExtractValue(b_ad);
  ConicProgramSolver solver(P, q, A, b, cones);
  const auto [x, z] = solver.Solve(mu_target);
  const VectorXd s = b - A * x;

  MatrixXd dP = math::ExtractGradient(P_ad);
  MatrixXd dq = math::ExtractGradient(q_ad);
  MatrixXd dA = math::ExtractGradient(A_ad);
  MatrixXd db = math::ExtractGradient(b_ad);
  ResizeGradients(&dP, &dq, &dA, &db);

  ConicProgramKkt kkt(P, A, cones, s, z);
  const MatrixXd bx =
      -(TensorProduct(dP, x) + dq + TransposeTensorProduct(dA, z));
  const MatrixXd bs = -(TensorProduct(dA, x) - db);
  const MatrixXd bz = MatrixXd::Zero(bs.rows(), bs.cols());
  const auto [dx, ds, dz] = kkt.Solve(bx, bs, bz);

  const VectorX<AutoDiffXd> x_ad = math::InitializeAutoDiff(x, dx);
  const VectorX<AutoDiffXd> z_ad = math::InitializeAutoDiff(z, dz);

  if (dx_dmu_out != nullptr || dz_dmu_out != nullptr) {
    const VectorX<AutoDiffXd> s_ad = math::InitializeAutoDiff(s, ds);
    const auto [dx_dmu, ds_dmu, dz_dmu] =
        ComputeDiffWrtMu(P_ad, A_ad, cones, s_ad, z_ad);
    if (dx_dmu_out != nullptr) *dx_dmu_out = dx_dmu;
    if (dz_dmu_out != nullptr) *dz_dmu_out = dz_dmu;
  }

  return {x_ad, z_ad};
}

/* Class Cone. */
class Cone::Impl {
 public:
  virtual ~Impl() = default;

  virtual Cone::Tag tag() const = 0;
  virtual int dimension() const = 0;
  virtual int degree() const = 0;

  virtual std::unique_ptr<Impl> Clone() const = 0;

  virtual double CalcLogBarrier(const Eigen::Ref<const VectorXd>& s) const = 0;

  virtual VectorXd CalcLogBarrierGradient(
      const Eigen::Ref<const VectorXd>& s) const = 0;

  virtual MatrixXd CalcLogBarrierHessian(
      const Eigen::Ref<const VectorXd>& s) const = 0;

  virtual VectorXd JordanIdentity() const = 0;

  virtual MatrixXd JordanLeftMul(const Eigen::Ref<const VectorXd>& s) const = 0;

  virtual MatrixX<AutoDiffXd> JordanLeftMul(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const = 0;

  virtual MatrixXd JordanLeftMulInverse(
      const Eigen::Ref<const VectorXd>& s) const = 0;

  virtual MatrixX<AutoDiffXd> JordanLeftMulInverse(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const = 0;
};

void Cone::ImplDeleter::operator()(Impl* impl) {
  delete impl;
}

Cone::Cone(std::unique_ptr<Impl> impl)
    : impl_(std::unique_ptr<Impl, ImplDeleter>(impl.release())) {
  DRAKE_THROW_UNLESS(impl_ != nullptr);
}

Cone::Cone(const Cone& other) : Cone(other.impl_->Clone()) {}

Cone::Cone(Cone&& other) : Cone(other) {}

Cone& Cone::operator=(const Cone& other) {
  this->impl_ =
      std::unique_ptr<Impl, ImplDeleter>(other.impl_->Clone().release());
  return *this;
}

Cone& Cone::operator=(Cone&& other) {
  *this = other;
  return *this;
}

Cone::Tag Cone::tag() const {
  return impl_->tag();
}

int Cone::dimension() const {
  return impl_->dimension();
}

int Cone::degree() const {
  return impl_->degree();
}

double Cone::CalcLogBarrier(const Eigen::Ref<const VectorXd>& s) const {
  DRAKE_THROW_UNLESS(s.size() == dimension());
  return impl_->CalcLogBarrier(s);
}

VectorXd Cone::CalcLogBarrierGradient(
    const Eigen::Ref<const VectorXd>& s) const {
  DRAKE_THROW_UNLESS(s.size() == dimension());
  return impl_->CalcLogBarrierGradient(s);
}

MatrixXd Cone::CalcLogBarrierHessian(
    const Eigen::Ref<const VectorXd>& s) const {
  DRAKE_THROW_UNLESS(s.size() == dimension());
  return impl_->CalcLogBarrierHessian(s);
}

VectorXd Cone::JordanIdentity() const {
  return impl_->JordanIdentity();
}

template <typename T>
MatrixX<T> Cone::JordanLeftMul(const Eigen::Ref<const VectorX<T>>& s) const {
  DRAKE_THROW_UNLESS(s.size() == dimension());
  return impl_->JordanLeftMul(s);
}

template <typename T>
MatrixX<T> Cone::JordanLeftMulInverse(
    const Eigen::Ref<const VectorX<T>>& s) const {
  DRAKE_THROW_UNLESS(s.size() == dimension());
  return impl_->JordanLeftMulInverse(s);
}

template MatrixXd Cone::JordanLeftMul<double>(
    const Eigen::Ref<const VectorXd>&) const;
template MatrixX<AutoDiffXd> Cone::JordanLeftMul<AutoDiffXd>(
    const Eigen::Ref<const VectorX<AutoDiffXd>>&) const;
template MatrixXd Cone::JordanLeftMulInverse<double>(
    const Eigen::Ref<const VectorXd>&) const;
template MatrixX<AutoDiffXd> Cone::JordanLeftMulInverse<AutoDiffXd>(
    const Eigen::Ref<const VectorX<AutoDiffXd>>&) const;

/* Class NonnegativeCone. */
class NonnegativeCone::Impl final : public Cone::Impl {
 public:
  explicit Impl(int dimension) : dimension_(dimension) {
    DRAKE_THROW_UNLESS(dimension >= 1);
  }

  Cone::Tag tag() const override { return Cone::Tag::NonnegativeCone; }

  int dimension() const override { return dimension_; }

  int degree() const override { return dimension_; }

  std::unique_ptr<Cone::Impl> Clone() const override {
    return std::make_unique<NonnegativeCone::Impl>(dimension_);
  }

  double CalcLogBarrier(const Eigen::Ref<const VectorXd>& s) const override {
    double result = 0;
    for (int i = 0; i < dimension(); ++i) {
      result += (s[i] > 0.0) ? -std::log(s[i])
                             : std::numeric_limits<double>::infinity();
    }
    return result;
  }

  VectorXd CalcLogBarrierGradient(
      const Eigen::Ref<const VectorXd>& s) const override {
    // Clamp s to avoid division by zero.
    return -s.cwiseMax(1e-8).cwiseInverse();
  }

  MatrixXd CalcLogBarrierHessian(
      const Eigen::Ref<const VectorXd>& s) const override {
    // Clamp s to avoid division by zero.
    return s.cwiseMax(1e-8).cwiseAbs2().cwiseInverse().asDiagonal();
  }

  VectorXd JordanIdentity() const override {
    return VectorXd::Ones(dimension());
  }

  template <typename T>
  static MatrixX<T> JordanLeftMulImpl(const Eigen::Ref<const VectorX<T>>& s) {
    return s.asDiagonal();
  }

  MatrixXd JordanLeftMul(const Eigen::Ref<const VectorXd>& s) const override {
    return JordanLeftMulImpl(s);
  }

  MatrixX<AutoDiffXd> JordanLeftMul(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const override {
    return JordanLeftMulImpl(s);
  }

  template <typename T>
  static MatrixX<T> JordanLeftMulInverseImpl(
      const Eigen::Ref<const VectorX<T>>& s) {
    return s.cwiseInverse().asDiagonal();
  }

  MatrixXd JordanLeftMulInverse(
      const Eigen::Ref<const VectorXd>& s) const override {
    return JordanLeftMulInverseImpl(s);
  }

  MatrixX<AutoDiffXd> JordanLeftMulInverse(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const override {
    return JordanLeftMulInverseImpl(s);
  }

 private:
  int dimension_;
};

NonnegativeCone::NonnegativeCone(int dimension)
    : Cone(std::make_unique<NonnegativeCone::Impl>(dimension)) {}

/* Class SecondOrderCone. */
class SecondOrderCone::Impl final : public Cone::Impl {
 public:
  explicit Impl(int dimension) : dimension_(dimension) {
    DRAKE_THROW_UNLESS(dimension >= 2);
  }

  Cone::Tag tag() const override { return Cone::Tag::SecondOrderCone; }

  int dimension() const override { return dimension_; }

  int degree() const override { return 1; }

  std::unique_ptr<Cone::Impl> Clone() const override {
    return std::make_unique<SecondOrderCone::Impl>(dimension_);
  }

  double CalcLogBarrier(const Eigen::Ref<const VectorXd>& s) const override {
    const auto u = s.tail(dimension() - 1);
    const double q = s[0] * s[0] - u.dot(u);
    const double result = (q > 0.0 && s[0] > 0)
                              ? -0.5 * std::log(q)
                              : std::numeric_limits<double>::infinity();
    return result;
  }

  VectorXd CalcLogBarrierGradient(
      const Eigen::Ref<const VectorXd>& s) const override {
    const auto u = s.tail(dimension() - 1);
    double q = s[0] * s[0] - u.dot(u);
    q = std::max(q, 1e-8);  // Clamp q to avoid division by zero.
    VectorXd s_prime = s;
    s_prime.tail(dimension() - 1) *= -1;
    return -1 / q * s_prime;
  }

  MatrixXd CalcLogBarrierHessian(
      const Eigen::Ref<const VectorXd>& s) const override {
    const auto u = s.tail(dimension() - 1);
    double q = s[0] * s[0] - u.dot(u);
    q = std::max(q, 1e-8);  // Clamp q to avoid division by zero.
    VectorXd s_prime = s;
    s_prime.tail(dimension() - 1) *= -1;
    VectorXd d = VectorXd::Ones(dimension());
    d.tail(dimension() - 1) *= -1;
    return 2 / (q * q) * s_prime * s_prime.transpose() -
           1 / q * d.asDiagonal().toDenseMatrix();
  }

  VectorXd JordanIdentity() const override {
    VectorXd identity = VectorXd::Zero(dimension());
    identity[0] = 1;
    return identity;
  }

  template <typename T>
  static MatrixX<T> JordanLeftMulImpl(const Eigen::Ref<const VectorX<T>>& s) {
    const int d1 = s.size() - 1;
    const T& s0 = s[0];
    const auto s1 = s.tail(d1);

    MatrixX<T> L(s.size(), s.size());
    L(0, 0) = s0;
    L.topRightCorner(1, d1) = s1.transpose();
    L.bottomLeftCorner(d1, 1) = s1;
    L.bottomRightCorner(d1, d1) = s0 * MatrixX<T>::Identity(d1, d1);

    return L;
  }

  MatrixXd JordanLeftMul(const Eigen::Ref<const VectorXd>& s) const override {
    return JordanLeftMulImpl(s);
  }

  MatrixX<AutoDiffXd> JordanLeftMul(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const override {
    return JordanLeftMulImpl(s);
  }

  template <typename T>
  static MatrixX<T> JordanLeftMulInverseImpl(
      const Eigen::Ref<const VectorX<T>>& s) {
    const int d1 = s.size() - 1;
    const T& s0 = s[0];
    const auto s1 = s.tail(d1);

    MatrixX<T> invL(s.size(), s.size());
    invL(0, 0) = s0;
    invL.topRightCorner(1, d1) = -s1.transpose();
    invL.bottomLeftCorner(d1, 1) = -s1;
    invL.bottomRightCorner(d1, d1) =
        ((s0 * s0 - s1.dot(s1)) * MatrixX<T>::Identity(d1, d1) +
         s1 * s1.transpose()) /
        s0;
    invL /= (s0 * s0 - s1.dot(s1));

    return invL;
  }

  MatrixXd JordanLeftMulInverse(
      const Eigen::Ref<const VectorXd>& s) const override {
    return JordanLeftMulInverseImpl(s);
  }

  MatrixX<AutoDiffXd> JordanLeftMulInverse(
      const Eigen::Ref<const VectorX<AutoDiffXd>>& s) const override {
    return JordanLeftMulInverseImpl(s);
  }

 private:
  int dimension_;
};

SecondOrderCone::SecondOrderCone(int dimension)
    : Cone(std::make_unique<SecondOrderCone::Impl>(dimension)) {}

}  // namespace solvers
}  // namespace drake
