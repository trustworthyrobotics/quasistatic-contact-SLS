#include <vector>

#include "drake/bindings/generated_docstrings/solvers.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/solvers/solvers_py.h"
#include "drake/solvers/solve_conic_program.h"

namespace drake {
namespace pydrake {
namespace internal {

template <typename T>
void DefineSolveConicProgram(py::module m, T) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc_solvers.drake.solvers;
  py::tuple param = GetPyParam<T>();

  AddTemplateFunction(
      m, "SolveConicProgram",
      [](const MatrixX<T>& P, const VectorX<T>& q, const MatrixX<T>& A,
          const VectorX<T>& b, const std::vector<Cone>& cones, double mu_target,
          bool diff_wrt_mu) {
        if (!diff_wrt_mu) {
          const auto [x, z] = SolveConicProgram(P, q, A, b, cones, mu_target);
          return py::make_tuple(x, z);
        } else {
          VectorX<T> dx_dmu, dz_dmu;
          const auto [x, z] =
              SolveConicProgram(P, q, A, b, cones, mu_target, &dx_dmu, &dz_dmu);
          return py::make_tuple(x, z, dx_dmu, dz_dmu);
        }
      },
      param, py::arg("P"), py::arg("q"), py::arg("A"), py::arg("b"),
      py::arg("cones"), py::arg("mu_target") = 0.0,
      py::arg("diff_wrt_mu") = false, doc.SolveConicProgram.doc);
}

void DefineSolversConic(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::solvers;
  constexpr auto& doc = pydrake_doc_solvers.drake.solvers;

  {
    using Class = Cone;
    constexpr auto& cls_doc = doc.Cone;
    auto cls = py::class_<Class>(m, "Cone", cls_doc.doc);
    cls  // BR
        .def("dimension", &Class::dimension, cls_doc.dimension.doc)
        .def("degree", &Class::degree, cls_doc.degree.doc);
    DefCopyAndDeepCopy(&cls);
  }
  {
    using Class = NonnegativeCone;
    constexpr auto& cls_doc = doc.NonnegativeCone;
    auto cls = py::class_<Class, Cone>(m, "NonnegativeCone", cls_doc.doc);
    cls  // BR
        .def(py::init<int>(), py::arg("dimension"), cls_doc.ctor.doc);
    DefCopyAndDeepCopy(&cls);
  }
  {
    using Class = SecondOrderCone;
    constexpr auto& cls_doc = doc.SecondOrderCone;
    auto cls = py::class_<Class, Cone>(m, "SecondOrderCone", cls_doc.doc);
    cls  // BR
        .def(py::init<int>(), py::arg("dimension"), cls_doc.ctor.doc);
    DefCopyAndDeepCopy(&cls);
  }

  type_visit([m](auto dummy) { DefineSolveConicProgram(m, dummy); },
      NonSymbolicScalarPack{});
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
