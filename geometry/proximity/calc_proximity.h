#pragma once

#include "drake/geometry/query_results/signed_distance_pair.h"
#include "drake/geometry/shape_2d_specification.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rigid_transform_2d.h"

namespace drake {
namespace geometry {

/** Computes the proximity of a pair of 2D shapes. Only valid when the
 geometries are not in collision.
 @param shape_A Shape A.
 @param X_WA The pose of frame A relative to the world frame.
 @param shape_B Shape B.
 @param X_WA The pose of frame B relative to the world frame.
 @param kappa Smoothing parameter. (Zero means no smoothing and larger values
              mean higher smoothing.)
 @pre `kappa >= 0`
 @tparam_nonsymbolic_scalar */
template <typename T>
SignedDistancePair2d<T> CalcProximity2d(const Shape2d& shape_A,
                                        const math::RigidTransform2d<T>& X_WA,
                                        const Shape2d& shape_B,
                                        const math::RigidTransform2d<T>& X_WB,
                                        double kappa = 0.0);

/** Computes the proximity of a pair of 3D shapes. Only valid when the
 geometries are not in collision.
 @param shape_A Shape A.
 @param X_WA The pose of frame A relative to the world frame.
 @param shape_B Shape B.
 @param X_WA The pose of frame B relative to the world frame.
 @param kappa Smoothing parameter. (Zero means no smoothing and larger values
              mean higher smoothing.)
 @pre `kappa >= 0`
 @tparam_nonsymbolic_scalar */
template <typename T>
SignedDistancePair<T> CalcProximity(const Shape& shape_A,
                                    const math::RigidTransform<T>& X_WA,
                                    const Shape& shape_B,
                                    const math::RigidTransform<T>& X_WB,
                                    double kappa = 0.0);

}  // namespace geometry
}  // namespace drake
