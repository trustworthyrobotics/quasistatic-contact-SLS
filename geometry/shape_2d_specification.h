#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/fmt_ostream.h"

namespace drake {
namespace geometry {

#ifndef DRAKE_DOXYGEN_CXX
class Circle;
class Obround;
class Rectangle;
#endif

/** The abstract base class for all 2D shape specifications. Concrete subclasses
  exist for specific shapes (e.g., Circle, Obround, etc.).

  The Shape class has two key properties:

   - it is cloneable, and
   - it can be dispatched (see Visit).

 Note that the Shape2d class hierarchy is closed to third-party extensions. All
 Shape classes must be defined within Drake directly (and in this h/cc file
 pair in particular). */
class Shape2d {
 public:
  virtual ~Shape2d();

  /** Creates a copy of this shape. */
  std::unique_ptr<Shape2d> Clone() const;

  /** Returns the (unqualified) type name of this Shape, e.g., "Circle". */
  std::string_view type_name() const { return do_type_name(); }

  /** Returns a string representation of this shape. */
  std::string to_string() const { return do_to_string(); }

  /** Calls the given `visitor` function with `*this` as the sole argument, but
   with `*this` downcast to be the shape's concrete subclass. For example, if
   this shape is a %Circle then calls `visitor(static_cast<const Box&>(*this))`.
   @tparam ReturnType The return type to coerce return values into. When not
   `void`, anything returned by the visitor must be implicitly convertible to
   this type. When `void`, the return type will be whatever the Vistor's call
   operator returns by default.

   To see examples of how this is used, you can check the Drake source code,
   e.g., check the implementation of CalcArea() for one example. */
  template <typename ReturnType = void, typename Visitor>
  decltype(auto) Visit(Visitor&& visitor) const {
    if constexpr (std::is_same_v<ReturnType, void>) {
      return std::visit(
          [&visitor](auto* shape) {
            return visitor(*shape);
          },
          get_variant_this());
    } else {
      return std::visit(
          [&visitor](auto* shape) -> ReturnType {
            return visitor(*shape);
          },
          get_variant_this());
    }
  }

 protected:
  /** (Internal use only) Constructor for use by derived classes.
  All subclasses of Shape2d must be marked `final`. */
  Shape2d();

  // This is DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN, marked "internal use only".
  /** (Internal use only) For derived classes. */
  Shape2d(const Shape2d&) = default;
  /** (Internal use only) For derived classes. */
  Shape2d& operator=(const Shape2d&) = default;
  /** (Internal use only) For derived classes. */
  Shape2d(Shape2d&&) = default;
  /** (Internal use only) For derived classes. */
  Shape2d& operator=(Shape2d&&) = default;

  /** (Internal use only) NVI for Clone(). */
  virtual std::unique_ptr<Shape2d> DoClone() const = 0;

  /** (Internal use only) NVI for type_name(). */
  virtual std::string_view do_type_name() const = 0;

  /** (Internal use only) NVI for to_string(). */
  virtual std::string do_to_string() const = 0;

  /** (Internal use only) All concrete subclasses, as const pointers. */
  using VariantShape2dConstPtr = std::variant<  //
      const Circle*,                            //
      const Obround*,                           //
      const Rectangle*>;

  /** (Internal use only) NVI-like helper function for Visit(). */
  virtual VariantShape2dConstPtr get_variant_this() const = 0;
};

/** Definition of a circle. It is centered in its canonical frame with the given
 radius. */
class Circle final : public Shape2d {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Circle);

  /** Constructs a sphere with the given `radius`.
   @throws std::exception if `radius` is not finite *non-negative*. Note that a
                              zero radius is considered valid. */
  explicit Circle(double radius);

  ~Circle() final;

  double radius() const { return radius_; }

 private:
  std::unique_ptr<Shape2d> DoClone() const final;
  std::string_view do_type_name() const final;
  std::string do_to_string() const final;
  VariantShape2dConstPtr get_variant_this() const final;

  double radius_{};
};

/** Definition of an obround. The obround can be thought of as a rectangle with
 circular caps attached. The obround's length refers to the length of the
 rectangular region, and the radius applies to the circular caps. The capsule is
 defined in its canonical frame C, centered on the frame origin and with the
 length of the obround parallel to the frame's x-axis. */
class Obround final : public Shape2d {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Obround);

  /** Constructs an obround with the given `radius` and `length`.
   @throws std::exception if `radius` or `length` is not finite positive. */
  explicit Obround(double radius, double length);

  ~Obround() final;

  double radius() const { return radius_; }

  double length() const { return length_; }

 private:
  std::unique_ptr<Shape2d> DoClone() const final;
  std::string_view do_type_name() const final;
  std::string do_to_string() const final;
  VariantShape2dConstPtr get_variant_this() const final;

  double radius_{};
  double length_{};
};

/** Definition of a rectangle. The rectangle is defined in its canonical frame
 C, centered on the frame origin and with the length of the rectangle parallel
 to the frame's x-axis. */
class Rectangle final : public Shape2d {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Rectangle);

  /** Constructs an rectangle with the given `width` and `height`.
   @throws std::exception if `width` or `height` is not finite positive. */
  Rectangle(double width, double height);

  ~Rectangle() final;

  double width() const { return width_; }
  double height() const { return height_; }

 private:
  std::unique_ptr<Shape2d> DoClone() const final;
  std::string_view do_type_name() const final;
  std::string do_to_string() const final;
  VariantShape2dConstPtr get_variant_this() const final;

  double width_{};
  double height_{};
};

/** Calculates the area of the Shape2d. */
double CalcArea(const Shape2d& shape);

}  // namespace geometry
}  // namespace drake

DRAKE_FORMATTER_AS(, drake::geometry, Circle, x, x.to_string())
