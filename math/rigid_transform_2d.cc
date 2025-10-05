#include "drake/math/rigid_transform_2d.h"

namespace drake {
namespace math {

template <typename T>
std::ostream& operator<<(std::ostream& out, const RigidTransform2d<T>& X) {
  const T& theta = X.angle();
  const Vector2<T>& p = X.translation();
  out << fmt::format("theta = {} xy = {} {}", theta, p.x(), p.y());
  return out;
}

// clang-format off
DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS((
    static_cast<std::ostream&(*)(std::ostream&, const RigidTransform2d<T>&)>(
        &operator<< )
));
// clang-format on

}  // namespace math
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::math::RigidTransform2d);
