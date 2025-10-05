#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "drake/bindings/generated_docstrings/multibody_quasidynamic.h"
#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/type_pack.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"
#include "drake/multibody/quasidynamic/quasidynamic_linear_pusher.h"
#include "drake/multibody/quasidynamic/quasidynamic_multibody_plant.h"
#include "drake/multibody/quasidynamic/quasidynamic_planar_plant.h"

namespace drake {
namespace pydrake {
namespace {

template <typename T>
void DoScalarDependentDefinitions(py::module m, T) {
  py::tuple param = GetPyParam<T>();

  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody::quasidynamic;
  constexpr auto& doc =
      pydrake_doc_multibody_quasidynamic.drake.multibody.quasidynamic;

  {
    using Class = QuasidynamicDifferentiableContactModel<T>;
    constexpr auto& cls_doc = doc.QuasidynamicDifferentiableContactModel;
    auto cls = DefineTemplateClassWithDefault<Class, systems::LeafSystem<T>>(
        m, "QuasidynamicDifferentiableContactModel", param, cls_doc.doc);
    cls  // BR
        .def("num_contacts", &Class::num_contacts, cls_doc.num_contacts.doc)
        .def("contact_dim", &Class::contact_dim, cls_doc.contact_dim.doc)
        .def("get_dynamics_smoothing", &Class::get_dynamics_smoothing,
            py::arg("context"), cls_doc.get_dynamics_smoothing.doc)
        .def("set_dynamics_smoothing", &Class::set_dynamics_smoothing,
            py::arg("context"), py::arg("dynamics_smoothing"),
            cls_doc.set_dynamics_smoothing.doc)
        .def("get_actuation_input_port", &Class::get_actuation_input_port,
            py_rvp::reference_internal, cls_doc.get_actuation_input_port.doc)
        .def("get_state_output_port", &Class::get_state_output_port,
            py_rvp::reference_internal, cls_doc.get_state_output_port.doc)
        .def("get_contact_force_output_port",
            &Class::get_contact_force_output_port, py_rvp::reference_internal,
            cls_doc.get_contact_force_output_port.doc)
        .def("GetState", &Class::GetState, py::arg("context"),
            cls_doc.GetState.doc)
        .def("SetState", &Class::SetState, py::arg("context"), py::arg("x"),
            cls_doc.SetState.doc)
        .def("GetDisturbanceMatrix", &Class::GetDisturbanceMatrix,
            py::arg("context"), cls_doc.GetDisturbanceMatrix.doc)
        .def("EvalQuasidynamicPMatrix", &Class::EvalQuasidynamicPMatrix,
            py::arg("context"), cls_doc.EvalQuasidynamicPMatrix.doc)
        .def("EvalQuasidynamicBVector", &Class::EvalQuasidynamicBVector,
            py::arg("context"), cls_doc.EvalQuasidynamicBVector.doc)
        .def("EvalContactJacobian", &Class::EvalContactJacobian,
            py::arg("context"), cls_doc.EvalContactJacobian.doc)
        .def("EvalContactSignedDistance", &Class::EvalContactSignedDistance,
            py::arg("context"), cls_doc.EvalContactSignedDistance.doc)
        .def("EvalContactFrictionCoefficient",
            &Class::EvalContactFrictionCoefficient, py::arg("context"),
            cls_doc.EvalContactFrictionCoefficient.doc);
    DefClone(&cls);
  }

  {
    using Class = QuasidynamicLinearPusher<T>;
    constexpr auto& cls_doc = doc.QuasidynamicLinearPusher;
    auto cls = DefineTemplateClassWithDefault<Class,
        QuasidynamicDifferentiableContactModel<T>>(
        m, "QuasidynamicLinearPusher", param, cls_doc.doc);
    cls  // BR
        .def(py::init<double, double, double, double, double>(), py::arg("k_a"),
            py::arg("m_o"), py::arg("w"), py::arg("dt"),
            py::arg("dynamics_smoothing") = 0, cls_doc.ctor.doc)
        .def("SetMeshcat", &Class::SetMeshcat, py::arg("meshcat"),
            cls_doc.SetMeshcat.doc);
    DefClone(&cls);
  }

  {
    using Class = QuasidynamicMultibodyPlant<T>;
    constexpr auto& cls_doc = doc.QuasidynamicMultibodyPlant;
    auto cls = DefineTemplateClassWithDefault<Class,
        QuasidynamicDifferentiableContactModel<T>>(
        m, "QuasidynamicMultibodyPlant", param, cls_doc.doc);
    cls  // BR
        .def(py::init<double, double, double>(), py::arg("dt"),
            py::arg("dynamics_smoothing") = 0,
            py::arg("geometry_smoothing") = 0, cls_doc.ctor.doc)
        .def("AddRigidBody", &Class::AddRigidBody,
            py::arg("M_BBo_B") = multibody::SpatialInertia<double>::Zero(),
            cls_doc.AddRigidBody.doc)
        .def("world_body", &Class::world_body, cls_doc.world_body.doc)
        .def("AddRevoluteJoint", &Class::AddRevoluteJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("axis"), py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddRevoluteJoint.doc)
        .def("AddPrismaticJoint", &Class::AddPrismaticJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("axis"), py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddPrismaticJoint.doc)
        .def("AddCartesianJoint", &Class::AddCartesianJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddCartesianJoint.doc)
        .def("AddRpyFloatingJoint", &Class::AddRpyFloatingJoint,
            py::arg("body_P"), py::arg("X_PJp"), py::arg("body_C"),
            py::arg("X_CJc"), py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddRpyFloatingJoint.doc)
        .def("RegisterCollisionGeometry", &Class::RegisterCollisionGeometry,
            py::arg("body"), py::arg("X_BG"), py::arg("shape"),
            py::arg("friction_coefficient"),
            cls_doc.RegisterCollisionGeometry.doc)
        .def("RegisterVisualGeometry", &Class::RegisterVisualGeometry,
            py::arg("body"), py::arg("X_BG"), py::arg("shape"), py::arg("rgba"),
            py::arg("name") = std::nullopt, cls_doc.RegisterVisualGeometry.doc)
        .def("SetGravityVector", &Class::SetGravityVector, py::arg("g_W"),
            cls_doc.SetGravityVector.doc)
        .def("Finalize", &Class::Finalize, cls_doc.Finalize.doc)
        .def("get_geometry_smoothing", &Class::get_geometry_smoothing,
            py::arg("context"), cls_doc.get_geometry_smoothing.doc)
        .def("set_geometry_smoothing", &Class::set_geometry_smoothing,
            py::arg("context"), py::arg("geometry_smoothing"),
            cls_doc.set_geometry_smoothing.doc)
        .def("EvalPoseInWorld", &Class::EvalPoseInWorld, py::arg("context"),
            py::arg("body"), cls_doc.EvalPoseInWorld.doc)
        .def("EvalCollisionJacobian", &Class::EvalCollisionJacobian,
            py::arg("context"), cls_doc.EvalCollisionJacobian.doc)
        .def("EvalCollisionSignedDistance", &Class::EvalCollisionSignedDistance,
            py::arg("context"), cls_doc.EvalCollisionSignedDistance.doc)
        .def("SetMeshcat", &Class::SetMeshcat, py::arg("meshcat"),
            py::arg("name_prefix") = "quasidynamic_multibody_plant",
            cls_doc.SetMeshcat.doc);
    DefClone(&cls);
  }

  {
    using Class = QuasidynamicPlanarPlant<T>;
    constexpr auto& cls_doc = doc.QuasidynamicPlanarPlant;
    auto cls = DefineTemplateClassWithDefault<Class,
        QuasidynamicDifferentiableContactModel<T>>(
        m, "QuasidynamicPlanarPlant", param, cls_doc.doc);
    cls  // BR
        .def(py::init<double, double, double>(), py::arg("dt"),
            py::arg("dynamics_smoothing") = 0,
            py::arg("geometry_smoothing") = 0, cls_doc.ctor.doc)
        .def("AddRigidBody",
            py::overload_cast<double, const Vector2<double>&, double>(
                &Class::AddRigidBody),
            py::arg("mass"), py::arg("p_BBcm_B"), py::arg("I_BBcm"),
            cls_doc.AddRigidBody.doc_3args)
        .def("AddRigidBody", py::overload_cast<>(&Class::AddRigidBody),
            cls_doc.AddRigidBody.doc_0args)
        .def("world_body", &Class::world_body, cls_doc.world_body.doc)
        .def("AddRevoluteJoint", &Class::AddRevoluteJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddRevoluteJoint.doc)
        .def("AddPrismaticJoint", &Class::AddPrismaticJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddPrismaticJoint.doc)
        .def("AddCartesianJoint", &Class::AddCartesianJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddCartesianJoint.doc)
        .def("AddPlanarJoint", &Class::AddPlanarJoint, py::arg("body_P"),
            py::arg("X_PJp"), py::arg("body_C"), py::arg("X_CJc"),
            py::arg("actuation_stiffness") = std::nullopt,
            cls_doc.AddPlanarJoint.doc)
        .def("RegisterCollisionGeometry", &Class::RegisterCollisionGeometry,
            py::arg("body"), py::arg("X_BG"), py::arg("shape"),
            py::arg("friction_coefficient"),
            cls_doc.RegisterCollisionGeometry.doc)
        .def("RegisterVisualGeometry",
            py::overload_cast<multibody::BodyIndex,
                const math::RigidTransform2d<double>&, const geometry::Shape2d&,
                const geometry::Rgba&, std::optional<std::string_view>>(
                &Class::RegisterVisualGeometry),
            py::arg("body"), py::arg("X_BG"), py::arg("shape"), py::arg("rgba"),
            py::arg("name") = std::nullopt,
            cls_doc.RegisterVisualGeometry
                .doc_was_unable_to_choose_unambiguous_names)
        .def("RegisterVisualGeometry",
            py::overload_cast<multibody::BodyIndex,
                const math::RigidTransform<double>&, const geometry::Shape&,
                const geometry::Rgba&, std::optional<std::string_view>>(
                &Class::RegisterVisualGeometry),
            py::arg("body"), py::arg("X_BG"), py::arg("shape"), py::arg("rgba"),
            py::arg("name") = std::nullopt,
            cls_doc.RegisterVisualGeometry
                .doc_was_unable_to_choose_unambiguous_names)
        .def("SetGravityVector", &Class::SetGravityVector, py::arg("g_W"),
            cls_doc.SetGravityVector.doc)
        .def("Finalize", &Class::Finalize, cls_doc.Finalize.doc)
        .def("get_geometry_smoothing", &Class::get_geometry_smoothing,
            py::arg("context"), cls_doc.get_geometry_smoothing.doc)
        .def("set_geometry_smoothing", &Class::set_geometry_smoothing,
            py::arg("context"), py::arg("geometry_smoothing"),
            cls_doc.set_geometry_smoothing.doc)
        .def("EvalPoseInWorld", &Class::EvalPoseInWorld, py::arg("context"),
            py::arg("body"), cls_doc.EvalPoseInWorld.doc)
        .def("EvalCollisionJacobian", &Class::EvalCollisionJacobian,
            py::arg("context"), cls_doc.EvalCollisionJacobian.doc)
        .def("EvalCollisionSignedDistance", &Class::EvalCollisionSignedDistance,
            py::arg("context"), cls_doc.EvalCollisionSignedDistance.doc)
        .def("SetMeshcat", &Class::SetMeshcat, py::arg("meshcat"),
            py::arg("name_prefix") = "quasidynamic_planar_plant",
            py::arg("z_height") = 0.05, cls_doc.SetMeshcat.doc);
    DefClone(&cls);
  }
}

}  // namespace

PYBIND11_MODULE(quasidynamic, m) {
  PYDRAKE_PREVENT_PYTHON3_MODULE_REIMPORT(m);
  m.doc() = "Bindings for multibody quasidynamic.";

  py::module::import("pydrake.geometry");
  py::module::import("pydrake.multibody.tree");
  py::module::import("pydrake.systems.framework");

  type_visit([m](auto dummy) { DoScalarDependentDefinitions(m, dummy); },
      NonSymbolicScalarPack{});
}

}  // namespace pydrake
}  // namespace drake
