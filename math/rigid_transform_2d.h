#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/fmt_ostream.h"
#include "drake/common/hash.h"
#include "drake/common/never_destroyed.h"

namespace drake {
namespace math {

/**
 This class represents a proper rigid transform between two frames which can
 be regarded in two ways.  A rigid transform describes the "pose" between two
 frames A and B (i.e., the relative orientation and position of A to B).
 Alternately, it can be regarded as a distance-preserving operator that can
 rotate and/or translate a rigid body without changing its shape or size
 (rigid) and without mirroring/reflecting the body (proper), e.g., it can add
 one position vector to another and express the result in a particular basis
 as `p_AoQ_A = X_AB * p_BoQ_B` (Q is any point).  In many ways, this rigid
 transform class is conceptually similar to using a homogeneous matrix as a
 linear operator.  See operator* documentation for an exception.

 The class stores a rotation angle following the right-handed rule with the
 z-axis. The class also stores a position vector from Ao (the origin of frame
 A) to Bo (the origin of frame B).  The position vector is expressed in frame
 A. The monogram notation for the transform relating frame A to B is `X_AB`.
 The monogram notation for the rotation matrix relating A to B is `R_AB`.
 The monogram notation for the position vector from Ao to Bo is `p_AoBo_A`.
 See @ref multibody_quantities for monogram notation for dynamics.

 @note This class does not store the frames associated with the transform and
 cannot enforce correct usage of this class.  For example, it makes sense to
 multiply %RigidTransform2ds as `X_AB * X_BC`, but not `X_AB * X_CB`.

 @note This class is not a 3x3 transformation matrix -- even though its
 operator*() methods act mostly like 3x3 matrix multiplication.  Instead,
 this class contains a scalr rotation angle and a 2x1 position vector.
 To convert this to a 2x3 matrix, use GetAsMatrix23().
 To convert this to a 3x3 matrix, use GetAsMatrix3().
 To convert this to an Eigen::Isometry, use GetAsIsometry().

 @tparam_default_scalar
 */
template <typename T>
class RigidTransform2d {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(RigidTransform2d);

  /** Constructs the %RigidTransform2d that corresponds to aligning the two
   frames so unit vectors Ax = Bx, Ay = By and point Ao is coincident
   with Bo.  Hence, the constructed %RigidTransform2d contains an identity
   rotation and a zero position vector. */
  RigidTransform2d() {}

  /** Constructs a %RigidTransform2d from a rotation angle and a position
   vector.
   @param[in] theta rotation matrix relating frames A and B (e.g., `theta_AB`).
   @param[in] p position vector from frame A's origin to frame B's origin,
   expressed in frame A.  In monogram notation p is denoted `p_AoBo_A`. */
  RigidTransform2d(const T& theta, const Vector2<T>& p) { set(theta, p); }

  /** Constructs a %RigidTransform2d with a given rotation angle and a zero
   position vector.
   @param[in] theta rotation matrix relating frames A and B (e.g., `theta_AB`).
   */
  explicit RigidTransform2d(const T& theta) { set_angle(theta); }

  /** Constructs a %RigidTransform2d that contains an identity rotation and a
   given position vector `p`.
   @param[in] p position vector from frame A's origin to frame B's origin,
   expressed in frame A.  In monogram notation p is denoted `p_AoBo_A`. */
  explicit RigidTransform2d(const Vector2<T>& p) { set_translation(p); }

  /** Constructs a %RigidTransform2d that contains an identity RotationMatrix
   and the position vector underlying the given `translation`.
   @param[in] translation translation-only transform that stores p_AoQ_A, the
   position vector from frame A's origin to a point Q, expressed in frame A.
   @note The constructed %RigidTransform2d `X_AAq` relates frame A to a frame
   Aq whose basis unit vectors are aligned with Ax, Ay, Az and whose origin
   position is located at point Q.
   @note This constructor provides an implicit conversion from Translation to
   %RigidTransform2d. */
  RigidTransform2d(  // NOLINT(runtime/explicit)
      const Eigen::Translation<T, 2>& translation) {
    set_translation(translation.translation());
  }

  /** Sets `this` %RigidTransform2d from a rotation angle and a position vector.
   @param[in] theta rotation matrix relating frames A and B (e.g., `theta_AB`).
   @param[in] p position vector from frame A's origin to frame B's origin,
   expressed in frame A.  In monogram notation p is denoted `p_AoBo_A`. */
  void set(const T& theta, const Vector2<T>& p) {
    set_angle(theta);
    set_translation(p);
  }

  /** Returns theta_AB, the rotation angle portion of `this` %RigidTransform2d.
   @return theta_AB the rotation angle portion of `this` %RigidTransform2d. */
  const T& angle() const { return theta_AB_; }

  /** Sets the rotation angle part of `this` %RigidTransform2d.
   @param[in] theta an angle (in radians). */
  void set_angle(const T& theta) { theta_AB_ = theta; }

  /** Returns `p_AoBo_A`, the position vector portion of `this`
   %RigidTransform2d, i.e., position vector from Ao (frame A's origin) to Bo
   (frame B's origin). */
  const Vector2<T>& translation() const { return p_AoBo_A_; }

  /** Sets the position vector portion of `this` %RigidTransform2d.
   @param[in] p position vector from Ao (frame A's origin) to Bo (frame B's
   origin) expressed in frame A.  In monogram notation p is denoted p_AoBo_A. */
  void set_translation(const Vector2<T>& p) { p_AoBo_A_ = p; }

  /** Creates a %RigidTransform2d templatized on a scalar type U from a
   %RigidTransform2d templatized on scalar type T.  For example,
   ```
   RigidTransform2d<double> source = RigidTransform2d<double>::Identity();
   RigidTransform2d<AutoDiffXd> foo = source.cast<AutoDiffXd>();
   ```
   @tparam U Scalar type on which the returned %RigidTransform2d is
   templated. */
  template <typename U>
  RigidTransform2d<U> cast() const
    requires is_default_scalar<U>
  {  // NOLINT(whitespace/braces)
    const U theta = static_cast<U>(theta_AB_);
    const Vector2<U> p = p_AoBo_A_.template cast<U>();
    return RigidTransform2d<U>(theta, p);
  }

  /** Returns the identity %RigidTransform2d (corresponds to coincident frames).
   @return the %RigidTransform2d that corresponds to aligning the two frames so
   unit vectors Ax = Bx, Ay = By and point Ao is coincident with Bo. */
  static const RigidTransform2d<T>& Identity() {
    // @internal This method's name was chosen to mimic Eigen's Identity().
    static const never_destroyed<RigidTransform2d<T>> kIdentity;
    return kIdentity.access();
  }

  /** Returns R_AB, the rotation matrix portion of `this` %RigidTransform2d.
   @return R_AB the rotation matrix portion of `this` %RigidTransform2d. */
  Eigen::Matrix2<T> rotation_matrix() const {
    Eigen::Rotation2D<T> rotation(theta_AB_);
    return rotation.toRotationMatrix();
  }

  /** Returns the 3x3 matrix associated with this %RigidTransform2d, i.e., X_AB.
   <pre>
    ┌                ┐
    │ R_AB  p_AoBo_A │
    │                │
    │   0      1     │
    └                ┘
   </pre> */
  Matrix3<T> GetAsMatrix3() const {
    Matrix3<T> pose;
    pose.template topLeftCorner<2, 2>() = rotation_matrix();
    pose.template topRightCorner<2, 1>() = translation();
    pose.row(2) = Vector3<T>{0, 0, 1};
    return pose;
  }

  /** Returns the 3x4 matrix associated with this %RigidTransform2d, i.e., X_AB.
   <pre>
    ┌                ┐
    │ R_AB  p_AoBo_A │
    └                ┘
   </pre> */
  Eigen::Matrix<T, 2, 3> GetAsMatrix23() const {
    Eigen::Matrix<T, 2, 3> pose;
    pose.template topLeftCorner<2, 2>() = rotation_matrix();
    pose.template topRightCorner<2, 1>() = translation();
    return pose;
  }

  /** Returns the isometry in ℜ² that is equivalent to a %RigidTransform2d. */
  Eigen::Transform<T, 2, Eigen::Isometry> GetAsIsometry2() const {
    Eigen::Transform<T, 2, Eigen::Isometry> pose;
    pose.linear() = rotation_matrix();
    pose.translation() = translation();
    return pose;
  }

  /** Sets `this` %RigidTransform2d so it corresponds to aligning the two frames
   so unit vectors Ax = Bx, Ay = By and point Ao is coincident with Bo. */
  const RigidTransform2d<T>& SetIdentity() {
    set(0.0, Vector2<T>::Zero());
    return *this;
  }

  /** Returns X_BA = X_AB⁻¹, the inverse of `this` %RigidTransform2d.
   @note The inverse of %RigidTransform2d X_AB is X_BA, which contains the
   rotation matrix R_BA = R_AB⁻¹ = R_ABᵀ and the position vector `p_BoAo_B_`
   (position from B's origin Bo to A's origin Ao, expressed in frame B).
   @note: The square-root of a %RigidTransform2d's condition number is
   roughly the magnitude of the position vector.  The accuracy of the
   calculation for the inverse of a %RigidTransform2d drops off with the sqrt
   condition number. */
  RigidTransform2d<T> inverse() const {
    // @internal This method's name was chosen to mimic Eigen's inverse().
    RigidTransform2d<T> X_BA;
    X_BA.set_angle(-angle());
    X_BA.set_translation(X_BA.rotation_matrix() * (-p_AoBo_A_));
    return X_BA;
  }

  /** Calculates the product of `this` inverted and another %RigidTransform2d.
   If you consider `this` to be the transform X_AB, and `other` to be
   X_AC, then this method returns X_BC = X_AB⁻¹ * X_AC. For T==double, this
   method can be _much_ faster than inverting first and then performing the
   composition, because it can take advantage of the special structure of
   a rigid transform to avoid unnecessary memory and floating point
   operations. On some platforms it can use SIMD instructions for further
   speedups.
   @param[in] other %RigidTransform2d that post-multiplies `this` inverted.
   @retval X_BC where X_BC = this⁻¹ * other.
   @note It is possible (albeit improbable) to create an invalid rigid
   transform by accumulating round-off error with a large number of
   multiplies. */
  RigidTransform2d<T> InvertAndCompose(const RigidTransform2d<T>& other) const {
    const RigidTransform2d<T>& X_AC = other;  // Nicer name.
    RigidTransform2d<T> X_BC;
    const RigidTransform2d<T> X_BA = inverse();
    X_BC = X_BA * X_AC;
    return X_BC;
  }

  /** In-place multiply of `this` %RigidTransform2d `X_AB` by `other`
   %RigidTransform2d `X_BC`.
   @param[in] other %RigidTransform2d that post-multiplies `this`.
   @returns `this` %RigidTransform2d which has been multiplied by `other`.
   On return, `this = X_AC`, where `X_AC = X_AB * X_BC`. */
  RigidTransform2d<T>& operator*=(const RigidTransform2d<T>& other) {
    set(this->angle() + other.angle(), *this * other.translation());
    return *this;
  }

  /** Multiplies `this` %RigidTransform2d `X_AB` by the `other`
   %RigidTransform2d `X_BC` and returns the %RigidTransform2d `X_AC = X_AB *
   X_BC`. */
  RigidTransform2d<T> operator*(const RigidTransform2d<T>& other) const {
    RigidTransform2d<T> X_AC;
    X_AC.set(this->angle() + other.angle(), *this * other.translation());
    return X_AC;
  }

  /** Multiplies `this` %RigidTransform2d `X_AB` by the position vector
   `p_BoQ_B` which is from Bo (B's origin) to an arbitrary point Q.
   @param[in] p_BoQ_B position vector from Bo to Q, expressed in frame B.
   @retval p_AoQ_A position vector from Ao to Q, expressed in frame A. */
  Vector2<T> operator*(const Vector2<T>& p_BoQ_B) const {
    return translation() + rotation_matrix() * p_BoQ_B;
  }

  /** Multiplies `this` %RigidTransform2d `X_AB` by the n position vectors
   `p_BoQ1_B` ... `p_BoQn_B`, where `p_BoQi_B` is the iᵗʰ position vector
   from Bo (frame B's origin) to an arbitrary point Qi, expressed in frame B.
   @param[in] p_BoQ_B `2 x n` matrix with n position vectors `p_BoQi_B` or
   an expression that resolves to a `2 x n` matrix of position vectors.
   @retval p_AoQ_A `2 x n` matrix with n position vectors `p_AoQi_A`, i.e., n
   position vectors from Ao (frame A's origin) to Qi, expressed in frame A.
   Specifically, this operator* is defined so that `X_AB * p_BoQ_B` returns
   `p_AoQ_A = p_AoBo_A + R_AB * p_BoQ_B`, where
   `p_AoBo_A` is the position vector from Ao to Bo expressed in A and
   `R_AB` is the rotation matrix relating the orientation of frames A and B.
   @note As needed, use parentheses.  This operator* is not associative.
   To see this, let `p = p_AoBo_A`, `q = p_BoQ_B` and note
   (X_AB * q) * 7 = (p + R_AB * q) * 7 ≠ X_AB * (q * 7) = p + R_AB * (q * 7).
   @code{.cc}
   const RigidTransform2d<double> X_AB(0.1, Vector3d(1, 2, 3));
   Eigen::Matrix<double, 2, 3> p_BoQ_B;
   p_BoQ_B.col(0) = Vector2d(4, 5);
   p_BoQ_B.col(1) = Vector2d(6, 7);
   p_BoQ_B.col(2) = Vector2d(8, 9);
   const Eigen::Matrix<double, 2, 3> p_AoQ_A = X_AB * p_BoQ_B;
   @endcode */
  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, Derived::ColsAtCompileTime>
  operator*(const Eigen::MatrixBase<Derived>& p_BoQ_B) const {
    if (p_BoQ_B.rows() != 2) {
      throw std::logic_error(
          "Error: Inner dimension for matrix multiplication is not 2.");
    }
    // Express position vectors in terms of frame A as p_BoQ_A = R_AB * p_BoQ_B.
    const Matrix2<typename Derived::Scalar> R_AB = rotation_matrix();
    const Eigen::Matrix<typename Derived::Scalar, 2, Derived::ColsAtCompileTime>
        p_BoQ_A = R_AB * p_BoQ_B;

    // Reserve space (on stack or heap) to store the result.
    const int number_of_position_vectors = p_BoQ_B.cols();
    Eigen::Matrix<typename Derived::Scalar, 2, Derived::ColsAtCompileTime>
        p_AoQ_A(2, number_of_position_vectors);

    // Create each returned position vector as p_AoQi_A = p_AoBo_A + p_BoQi_A.
    for (int i = 0; i < number_of_position_vectors; ++i)
      p_AoQ_A.col(i) = translation() + p_BoQ_A.col(i);

    return p_AoQ_A;
  }

  /** Implements the @ref hash_append concept.
   @pre T implements the hash_append concept. */
  template <class HashAlgorithm>
  // NOLINTNEXTLINE(runtime/references) Per hash_append convention.
  friend void hash_append(HashAlgorithm& hasher,
                          const RigidTransform2d& X) noexcept {
    using drake::hash_append;
    hash_append(hasher, X.theta_AB_);
    const T* begin = X.p_AoBo_A_.data();
    const T* end = X.p_AoBo_A_.data() + X.p_AoBo_A_.size();
    using drake::hash_append_range;
    hash_append_range(hasher, begin, end);
  }

 private:
  // Rotation angle relating two frames, e.g. frame A and frame B.
  T theta_AB_{0.0};

  // Position vector from A's origin Ao to B's origin Bo, expressed in A.
  Vector2<T> p_AoBo_A_{Vector2<T>::Zero()};
};

/** Stream insertion operator to write an instance of RigidTransform2d into a
 `std::ostream`. Especially useful for debugging.
 @relates RigidTransform2d. */
template <typename T>
std::ostream& operator<<(std::ostream& out, const RigidTransform2d<T>& X);

}  // namespace math
}  // namespace drake

// Format RigidTransform2d using its operator<<.
// TODO(jwnimmer-tri) Add a real formatter and deprecate the operator<<.
namespace fmt {
template <typename T>
struct formatter<drake::math::RigidTransform2d<T>> : drake::ostream_formatter {
};
}  // namespace fmt

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::math::RigidTransform2d);
