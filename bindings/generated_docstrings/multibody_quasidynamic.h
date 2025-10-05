#pragma once

// GENERATED FILE DO NOT EDIT
// This file contains docstrings for the Python bindings that were
// automatically extracted by mkdoc.py.

#include <array>
#include <utility>

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// #include "drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h"
// #include "drake/multibody/quasidynamic/quasidynamic_linear_pusher.h"
// #include "drake/multibody/quasidynamic/quasidynamic_multibody_plant.h"
// #include "drake/multibody/quasidynamic/quasidynamic_planar_plant.h"

// Symbol: pydrake_doc_multibody_quasidynamic
constexpr struct /* pydrake_doc_multibody_quasidynamic */ {
  // Symbol: drake
  struct /* drake */ {
    // Symbol: drake::multibody
    struct /* multibody */ {
      // Symbol: drake::multibody::quasidynamic
      struct /* quasidynamic */ {
        // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel
        struct /* QuasidynamicDifferentiableContactModel */ {
          // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
          const char* doc =
R"""(``QuasidynamicDifferentiableContactModel`` describes the dynamics of
multibody systems as quasidynamic. The model steps at a fixed time
step. The contact dynamics can be smoothed for differentiability.)""";
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::DeclareStateAndPorts
          struct /* DeclareStateAndPorts */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Declares the states and input output ports.

Parameter ``x``:
    The default state.

Parameter ``num_inputs``:
    Number of inputs.

Parameter ``num_contacts``:
    Number of contacts.

Parameter ``contact_dim``:
    Dimension of contact, can be 1, 2, or 3.

Parameter ``dt``:
    Time step size.

Precondition:
    ``x.size() > 0``.

Precondition:
    ``num_inputs > 0``.

Precondition:
    ``num_contacts > 0``.

Precondition:
    ``contact_dim == 1 || contact_dim == 2 || contact_dim == 3``.

Precondition:
    ``dt > 0``.)""";
          } DeclareStateAndPorts;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::EvalContactFrictionCoefficient
          struct /* EvalContactFrictionCoefficient */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Evaluates the friction coefficient for each contact.)""";
          } EvalContactFrictionCoefficient;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::EvalContactJacobian
          struct /* EvalContactJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Evaluates the contact jacobian and signed distance.)""";
          } EvalContactJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::EvalContactSignedDistance
          struct /* EvalContactSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Evaluates the contact signed distance.)""";
          } EvalContactSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::EvalQuasidynamicBVector
          struct /* EvalQuasidynamicBVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Evaluates the b(x,u) vector governing the quasidynamic model.)""";
          } EvalQuasidynamicBVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::EvalQuasidynamicPMatrix
          struct /* EvalQuasidynamicPMatrix */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Evaluates the P(x) matrix governing the quasidynamic model.)""";
          } EvalQuasidynamicPMatrix;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::GetDisturbanceMatrix
          struct /* GetDisturbanceMatrix */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Returns a reference to the disturbance matrix in a given context.

Precondition:
    ``context`` is compatible with this system.)""";
          } GetDisturbanceMatrix;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::GetState
          struct /* GetState */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Returns a reference to the vector of state in a given context.

Precondition:
    ``context`` is compatible with this system.)""";
          } GetState;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::QuasidynamicDifferentiableContactModel<T>
          struct /* ctor */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Constructur invoked by derived classes.

Parameter ``converter``:
    Set to systems::SystemTypeTag<DerivedClass>{}.

Parameter ``kappa``:
    The dynamics smoothing parameter.

Precondition:
    ``dynamics_smoothing >= 0``.)""";
          } ctor;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::SetState
          struct /* SetState */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Sets the state in a given Context from a given vector.

Precondition:
    ``context`` is compatible with this system.

Precondition:
    ``x`` has the correct size.)""";
          } SetState;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::contact_dim
          struct /* contact_dim */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Returns the dimension of the contact, may be 1, 2, or 3.)""";
          } contact_dim;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::get_actuation_input_port
          struct /* get_actuation_input_port */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Returns the actuation input port)""";
          } get_actuation_input_port;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::get_contact_force_output_port
          struct /* get_contact_force_output_port */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Returns the contact force output port.)""";
          } get_contact_force_output_port;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::get_dynamics_smoothing
          struct /* get_dynamics_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Gets the dynamics smoothing parameter.)""";
          } get_dynamics_smoothing;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::get_state_output_port
          struct /* get_state_output_port */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Returns the state output port.)""";
          } get_state_output_port;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::num_contacts
          struct /* num_contacts */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc = R"""(Returns the number of contact pairs.)""";
          } num_contacts;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicDifferentiableContactModel::set_dynamics_smoothing
          struct /* set_dynamics_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_differentiable_contact_model.h
            const char* doc =
R"""(Sets the dynamics smoothing parameter. (Zero means no smoothing and
larger values mean higher smoothing.)

Precondition:
    ``dynamics_smoothing >= 0``.)""";
          } set_dynamics_smoothing;
        } QuasidynamicDifferentiableContactModel;
        // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher
        struct /* QuasidynamicLinearPusher */ {
          // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
          const char* doc =
R"""(``QuasidynamicLinearPusher`` represents a 1D pusher controlled by a
stiffness controller and pushes an object.)""";
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::EvalContactFrictionCoefficient
          struct /* EvalContactFrictionCoefficient */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc =
R"""(Evaluates the friction coefficient for the contact.)""";
          } EvalContactFrictionCoefficient;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::EvalContactJacobian
          struct /* EvalContactJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc =
R"""(Evaluates the contact jacobian and signed distance.)""";
          } EvalContactJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::EvalContactSignedDistance
          struct /* EvalContactSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc = R"""(Evaluates the contact signed distance.)""";
          } EvalContactSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::EvalQuasidynamicBVector
          struct /* EvalQuasidynamicBVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc =
R"""(Evaluates the b(x,u) vector governing the quasidynamic model.)""";
          } EvalQuasidynamicBVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::EvalQuasidynamicPMatrix
          struct /* EvalQuasidynamicPMatrix */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc =
R"""(Evaluates the P(x) matrix governing the quasidynamic model.)""";
          } EvalQuasidynamicPMatrix;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::QuasidynamicLinearPusher<T>
          struct /* ctor */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc =
R"""(Constructs a linear pusher.

Parameter ``k_a``:
    Stiffness of the pusher controller.

Parameter ``m_o``:
    Mass of the pushed object.

Parameter ``w``:
    Width of both the pusher and the object.

Parameter ``dt``:
    Time step size.

Parameter ``dynamics_smoothing``:
    The dynamics smoothing parameter.

Precondition:
    ``k_a > 0``.

Precondition:
    ``m_o > 0``.

Precondition:
    ``w > 0``.

Precondition:
    ``dt > 0``.

Precondition:
    ``dynamics_smoothing >= 0``.)""";
          } ctor;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicLinearPusher::SetMeshcat
          struct /* SetMeshcat */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_linear_pusher.h
            const char* doc = R"""(Sets the meshcat for which to draw to.)""";
          } SetMeshcat;
        } QuasidynamicLinearPusher;
        // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant
        struct /* QuasidynamicMultibodyPlant */ {
          // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
          const char* doc =
R"""(``QuasidynamicMultibodyPlant`` represents a spatial quasidynamic
model. The dynamics are governed by P(x) x⁺ + b(x,u) − J(x)ᵀ λ = 0.)""";
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::AddCartesianJoint
          struct /* AddCartesianJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Adds a Cartesian joint.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddCartesianJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::AddPrismaticJoint
          struct /* AddPrismaticJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Adds a prismatic joint.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``axis``:
    A vector specifying the translation axis for this joint.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddPrismaticJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::AddRevoluteJoint
          struct /* AddRevoluteJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Adds a revolute joint.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``axis``:
    A vector specifying the axis of revolution for this joint.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddRevoluteJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::AddRigidBody
          struct /* AddRigidBody */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Adds a rigid body with the provided spatial inertia.

Parameter ``M_BBo_B``:
    Spatial inertia of the rigid body, computed about the body frame
    origin Bo and expressed in the body frame B.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddRigidBody;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::AddRpyFloatingJoint
          struct /* AddRpyFloatingJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Adds a roll-pitch-yaw floating joint.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddRpyFloatingJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalCollisionJacobian
          struct /* EvalCollisionJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the collision jacobian.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalCollisionJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalCollisionSignedDistance
          struct /* EvalCollisionSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the collision signed distance.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalCollisionSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalContactFrictionCoefficient
          struct /* EvalContactFrictionCoefficient */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the friction coefficient for the contact.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactFrictionCoefficient;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalContactJacobian
          struct /* EvalContactJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the contact jacobian.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalContactSignedDistance
          struct /* EvalContactSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the contact signed distance.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalPoseInWorld
          struct /* EvalPoseInWorld */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the pose of a body in the world frame.

Raises:
    RuntimeError if called pre-finalize.

Precondition:
    ``body`` is a valid index.)""";
          } EvalPoseInWorld;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalQuasidynamicBVector
          struct /* EvalQuasidynamicBVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the b(x,u) vector governing the quasidynamic model.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalQuasidynamicBVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::EvalQuasidynamicPMatrix
          struct /* EvalQuasidynamicPMatrix */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Evaluates the P(x) matrix governing the quasidynamic model.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalQuasidynamicPMatrix;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::Finalize
          struct /* Finalize */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Finalizes the QuasidynamicMultibodyPlant. This method must be called
after all elements in the model (bodies, joints, geometries) are added
and before any computations are performed.)""";
          } Finalize;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::QuasidynamicMultibodyPlant<T>
          struct /* ctor */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Constructs a quasidynamic planar plant.

Parameter ``dt``:
    Time step size.

Parameter ``dynamics_smoothing``:
    The dynamics smoothing parameter.

Parameter ``geometry_smoothing``:
    The geometry smoothing parameter.

Precondition:
    ``dt > 0``.

Precondition:
    ``dynamics_smoothing >= 0``.

Precondition:
    ``geometry_smoothing >= 0``.)""";
          } ctor;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::RegisterCollisionGeometry
          struct /* RegisterCollisionGeometry */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Registers a geometry to be used for collision for a given body.

Parameter ``body``:
    The body for which geometry is being registered.

Parameter ``X_BG``:
    The fixed pose of the geometry frame G in the body frame B.

Parameter ``shape``:
    The geometry used for collision.

Parameter ``friction_coefficient``:
    The friction coefficient μ of the object. The effective
    coefficient of friction at the interface of two objects is
    computed as 2μ₁μ₂ / (μ₁+μ₂).

Precondition:
    ``body`` is a valid index.

Precondition:
    ``friction_coefficient >= 0``.

Raises:
    RuntimeError if called post-finalize.)""";
          } RegisterCollisionGeometry;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::RegisterVisualGeometry
          struct /* RegisterVisualGeometry */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Registers a geometry to be used for visualization for a given body.

Parameter ``body``:
    The body for which geometry is being registered.

Parameter ``X_BG``:
    The fixed pose of the geometry frame G in the body frame B.

Parameter ``shape``:
    The geometry used for visualization.

Parameter ``rgba``:
    The color for the geometry.

Parameter ``name``:
    The name for the geometry.

Precondition:
    ``body`` is a valid index.

Raises:
    RuntimeError if called post-finalize.)""";
          } RegisterVisualGeometry;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::SetGravityVector
          struct /* SetGravityVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Sets the acceleration of gravity vector, expressed in the world frame
W. The default gravity vector is set to [0, 0, -9.80665].

Raises:
    RuntimeError if called post-finalize.)""";
          } SetGravityVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::SetMeshcat
          struct /* SetMeshcat */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc = R"""(Sets the meshcat for which to draw to.)""";
          } SetMeshcat;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::get_geometry_smoothing
          struct /* get_geometry_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc = R"""(Gets the geometry smoothing parameter.)""";
          } get_geometry_smoothing;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::set_geometry_smoothing
          struct /* set_geometry_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Sets the geometry smoothing parameter. (Zero means no smoothing and
larger values mean higher smoothing.)

Precondition:
    ``geometry_smoothing >= 0``.)""";
          } set_geometry_smoothing;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicMultibodyPlant::world_body
          struct /* world_body */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_multibody_plant.h
            const char* doc =
R"""(Returns the index corresponding to the world body.)""";
          } world_body;
        } QuasidynamicMultibodyPlant;
        // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant
        struct /* QuasidynamicPlanarPlant */ {
          // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
          const char* doc =
R"""(``QuasidynamicPlanarPlant`` represents a planar quasidynamic model.
The dynamics are governed by P(x) x⁺ + b(x,u) − J(x)ᵀ λ = 0.)""";
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::AddCartesianJoint
          struct /* AddCartesianJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Adds a Cartesian joint with translation about frame J's x-axis and
y-axis.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddCartesianJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::AddPlanarJoint
          struct /* AddPlanarJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Adds a planar joint with translation about frame J's x-axis and y-axis
and rotation about z-axis.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddPlanarJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::AddPrismaticJoint
          struct /* AddPrismaticJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Adds a prismatic joint with translation about frame J's x-axis.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddPrismaticJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::AddRevoluteJoint
          struct /* AddRevoluteJoint */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Adds a revolute joint with rotation about frame J's z-axis.

Parameter ``body_P``:
    Index of the parent body.

Parameter ``X_PJp``:
    The fixed pose of frame Jp attached to the parent frame P.

Parameter ``body_C``:
    Index of the child body.

Parameter ``X_CJc``:
    The fixed pose of frame Jc attached to the child frame C.

Parameter ``actuation_stiffness``:
    If not null, make this joint actuated with the specified
    stiffness.

Precondition:
    ``body_P`` and ``body_C`` are valid indexes.

Precondition:
    ``actuation_stiffness`` is positive is not null.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddRevoluteJoint;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::AddRigidBody
          struct /* AddRigidBody */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc_3args =
R"""(Adds a rigid body B for a given mass, center of mass position, and
central rotational inertia.

Parameter ``mass``:
    Mass of the rigid body B.

Parameter ``p_BBcm_B``:
    The position vector from body frame origin to body center of mass
    Bcm, expressed in the frame body frame B.

Parameter ``I_BBcm``:
    Rigid body B's rotation inertia about Bcm.

Precondition:
    ``mass > 0``.

Precondition:
    ``I_BBcm > 0``.

Raises:
    RuntimeError if called post-finalize.)""";
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc_0args =
R"""(Adds a rigid body with inertia set to zero.

Raises:
    RuntimeError if called post-finalize.)""";
          } AddRigidBody;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalCollisionJacobian
          struct /* EvalCollisionJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the collision jacobian.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalCollisionJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalCollisionSignedDistance
          struct /* EvalCollisionSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the collision signed distance.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalCollisionSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalContactFrictionCoefficient
          struct /* EvalContactFrictionCoefficient */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the friction coefficient for the contact.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactFrictionCoefficient;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalContactJacobian
          struct /* EvalContactJacobian */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the contact jacobian.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactJacobian;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalContactSignedDistance
          struct /* EvalContactSignedDistance */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the contact signed distance.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalContactSignedDistance;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalPoseInWorld
          struct /* EvalPoseInWorld */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the pose of a body in the world frame.

Raises:
    RuntimeError if called pre-finalize.

Precondition:
    ``body`` is a valid index.)""";
          } EvalPoseInWorld;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalQuasidynamicBVector
          struct /* EvalQuasidynamicBVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the b(x,u) vector governing the quasidynamic model.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalQuasidynamicBVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::EvalQuasidynamicPMatrix
          struct /* EvalQuasidynamicPMatrix */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Evaluates the P(x) matrix governing the quasidynamic model.

Raises:
    RuntimeError if called pre-finalize.)""";
          } EvalQuasidynamicPMatrix;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::Finalize
          struct /* Finalize */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Finalizes the QuasidynamicPlanarPlant. This method must be called
after all elements in the model (bodies, joints, geometries) are added
and before any computations are performed.)""";
          } Finalize;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::QuasidynamicPlanarPlant<T>
          struct /* ctor */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Constructs a quasidynamic planar plant.

Parameter ``dt``:
    Time step size.

Parameter ``dynamics_smoothing``:
    The dynamics smoothing parameter.

Parameter ``geometry_smoothing``:
    The geometry smoothing parameter.

Precondition:
    ``dt > 0``.

Precondition:
    ``dynamics_smoothing >= 0``.

Precondition:
    ``geometry_smoothing >= 0``.)""";
          } ctor;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::RegisterCollisionGeometry
          struct /* RegisterCollisionGeometry */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Registers a geometry to be used for collision for a given body.

Parameter ``body``:
    The body for which geometry is being registered.

Parameter ``X_BG``:
    The fixed pose of the geometry frame G in the body frame B.

Parameter ``shape``:
    The geometry used for collision.

Parameter ``friction_coefficient``:
    The friction coefficient μ of the object. The effective
    coefficient of friction at the interface of two objects is
    computed as 2μ₁μ₂ / (μ₁+μ₂).

Precondition:
    ``body`` is a valid index.

Precondition:
    ``friction_coefficient >= 0``.

Raises:
    RuntimeError if called post-finalize.)""";
          } RegisterCollisionGeometry;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::RegisterVisualGeometry
          struct /* RegisterVisualGeometry */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc_was_unable_to_choose_unambiguous_names =
R"""(Registers a geometry to be used for visualization for a given body.

Parameter ``body``:
    The body for which geometry is being registered.

Parameter ``X_BG``:
    The fixed pose of the geometry frame G in the body frame B.

Parameter ``shape``:
    The geometry used for visualization.

Parameter ``rgba``:
    The color for the geometry.

Parameter ``name``:
    The name for the geometry.

Precondition:
    ``body`` is a valid index.

Raises:
    RuntimeError if called post-finalize.)""";
          } RegisterVisualGeometry;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::SetGravityVector
          struct /* SetGravityVector */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Sets the acceleration of gravity vector, expressed in the world frame
W. The default gravity vector is set to zero.

Raises:
    RuntimeError if called post-finalize.)""";
          } SetGravityVector;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::SetMeshcat
          struct /* SetMeshcat */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc = R"""(Sets the meshcat for which to draw to.)""";
          } SetMeshcat;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::get_geometry_smoothing
          struct /* get_geometry_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc = R"""(Gets the geometry smoothing parameter.)""";
          } get_geometry_smoothing;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::set_geometry_smoothing
          struct /* set_geometry_smoothing */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Sets the geometry smoothing parameter. (Zero means no smoothing and
larger values mean higher smoothing.)

Precondition:
    ``geometry_smoothing >= 0``.)""";
          } set_geometry_smoothing;
          // Symbol: drake::multibody::quasidynamic::QuasidynamicPlanarPlant::world_body
          struct /* world_body */ {
            // Source: drake/multibody/quasidynamic/quasidynamic_planar_plant.h
            const char* doc =
R"""(Returns the index corresponding to the world body.)""";
          } world_body;
        } QuasidynamicPlanarPlant;
      } quasidynamic;
    } multibody;
  } drake;
} pydrake_doc_multibody_quasidynamic;

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
