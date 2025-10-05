#include "drake/geometry/shape_2d_specification.h"

#include <cmath>
#include <utility>

#include <fmt/format.h>

#include "drake/common/drake_throw.h"
#include "drake/common/overloaded.h"

namespace drake {
namespace geometry {

/* Shape2d. */
Shape2d::Shape2d() = default;

Shape2d::~Shape2d() = default;

std::unique_ptr<Shape2d> Shape2d::Clone() const {
  return DoClone();
}

/* Circle. */
Circle::Circle(double radius) : radius_(radius) {
  DRAKE_THROW_UNLESS(std::isfinite(radius) && radius >= 0, radius);
}

std::string Circle::do_to_string() const {
  return fmt::format("Circle(radius={})", radius());
}

/* Obround. */
Obround::Obround(double radius, double length)
    : radius_(radius), length_(length) {
  DRAKE_THROW_UNLESS(std::isfinite(radius) && radius > 0, radius);
  DRAKE_THROW_UNLESS(std::isfinite(length) && length > 0, length);
}

std::string Obround::do_to_string() const {
  return fmt::format("Obround(radius={}, length={})", radius(), length());
}

/* Rectangle. */
Rectangle::Rectangle(double width, double height)
    : width_(width), height_(height) {
  DRAKE_THROW_UNLESS(std::isfinite(width) && width > 0, width);
  DRAKE_THROW_UNLESS(std::isfinite(height) && height > 0, height);
}

std::string Rectangle::do_to_string() const {
  return fmt::format("Rectangle(width={}, height={})", width(), height());
}

/* The NVI function definitions are enough boilerplate to merit a macro to
 implement them, and we might as well toss in the dtor for good measure. */
#define DRAKE_DEFINE_SHAPE2D_SUBCLASS_BOILERPLATE(ShapeType)            \
  ShapeType::~ShapeType() = default;                                    \
  std::unique_ptr<Shape2d> ShapeType::DoClone() const {                 \
    return std::unique_ptr<ShapeType>(new ShapeType(*this));            \
  }                                                                     \
  std::string_view ShapeType::do_type_name() const {                    \
    return #ShapeType;                                                  \
  }                                                                     \
  Shape2d::VariantShape2dConstPtr ShapeType::get_variant_this() const { \
    return this;                                                        \
  }

DRAKE_DEFINE_SHAPE2D_SUBCLASS_BOILERPLATE(Circle)
DRAKE_DEFINE_SHAPE2D_SUBCLASS_BOILERPLATE(Obround)
DRAKE_DEFINE_SHAPE2D_SUBCLASS_BOILERPLATE(Rectangle)

#undef DRAKE_DEFINE_SHAPE2D_SUBCLASS_BOILERPLATE

/* CalcArea. */
double CalcArea(const Shape2d& shape) {
  return shape.Visit(overloaded{
      [](const Circle& circle) {
        return M_PI * std::pow(circle.radius(), 2);
      },
      [](const Obround& obround) {
        return M_PI * std::pow(obround.radius(), 2) +
               obround.length() * obround.radius() * 2;
      },
      [](const Rectangle& rectangle) {
        return rectangle.width() * rectangle.height();
      },
  });
}

}  // namespace geometry
}  // namespace drake
