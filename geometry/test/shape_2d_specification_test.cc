#include "drake/geometry/shape_2d_specification.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/overloaded.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace geometry {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

GTEST_TEST(CircleTest, Test) {
  EXPECT_ANY_THROW(Circle(-1.0));
  EXPECT_ANY_THROW((Circle(kInf)));
  EXPECT_NO_THROW(Circle(0.0));

  Circle circle(2.0);
  EXPECT_EQ(circle.type_name(), "Circle");
  EXPECT_EQ(circle.to_string(), "Circle(radius=2)");
  EXPECT_DOUBLE_EQ(CalcArea(circle), M_PI * 4);
}

GTEST_TEST(ObroundTest, Test) {
  EXPECT_ANY_THROW(Obround(-1.0, 1.0));
  EXPECT_ANY_THROW(Obround(kInf, 1.0));
  EXPECT_ANY_THROW(Obround(1.0, -1.0));
  EXPECT_ANY_THROW(Obround(1.0, kInf));

  Obround obround(2.0, 3.0);
  EXPECT_EQ(obround.type_name(), "Obround");
  EXPECT_EQ(obround.to_string(), "Obround(radius=2, length=3)");
  EXPECT_DOUBLE_EQ(CalcArea(obround), M_PI * 4 + 12);
}

GTEST_TEST(RectangleTest, Test) {
  EXPECT_ANY_THROW(Rectangle(-1.0, 1.0));
  EXPECT_ANY_THROW(Rectangle(kInf, 1.0));
  EXPECT_ANY_THROW(Rectangle(1.0, -1.0));
  EXPECT_ANY_THROW(Rectangle(1.0, kInf));

  Rectangle rectangle(2.0, 3.0);
  EXPECT_EQ(rectangle.type_name(), "Rectangle");
  EXPECT_EQ(rectangle.to_string(), "Rectangle(width=2, height=3)");
  EXPECT_DOUBLE_EQ(CalcArea(rectangle), 6.0);
}

}  // namespace
}  // namespace geometry
}  // namespace drake
