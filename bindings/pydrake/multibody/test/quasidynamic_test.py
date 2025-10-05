import copy
import unittest

import numpy as np

from pydrake.common.test_utilities import numpy_compare
from pydrake.geometry import (
    Box,
    Circle,
    Rectangle,
    Rgba,
    Sphere,
)
from pydrake.math import RigidTransform, RigidTransform2d
from pydrake.multibody.quasidynamic import (
    QuasidynamicLinearPusher_,
    QuasidynamicMultibodyPlant_,
    QuasidynamicPlanarPlant_,
)
from pydrake.multibody.tree import SpatialInertia


class TestMultibodyDer(unittest.TestCase):
    @numpy_compare.check_nonsymbolic_types
    def test_quasidynamic_linear_pusher(self, T):
        dut = QuasidynamicLinearPusher_[T](
            k_a=1.0, m_o=0.01, w=0.1, dt=0.01, dynamics_smoothing=1e-8
        )

        self.assertEqual(dut.num_contacts(), 1)
        self.assertEqual(dut.contact_dim(), 1)

        self.assertEqual(dut.get_actuation_input_port().get_name(), "u")
        self.assertEqual(dut.get_state_output_port().get_name(), "x")
        self.assertEqual(
            dut.get_contact_force_output_port().get_name(), "lambda"
        )

        context = dut.CreateDefaultContext()

        kappa = 1e-2
        dut.set_dynamics_smoothing(context=context, dynamics_smoothing=kappa)
        self.assertEqual(dut.get_dynamics_smoothing(context=context), kappa)

        dut.SetState(context=context, x=[0.0, 0.2])
        numpy_compare.assert_float_equal(
            dut.GetState(context=context), [0.0, 0.2]
        )

        dut.get_actuation_input_port().FixValue(context, [0.1])

        context.SetDiscreteState(dut.EvalUniquePeriodicDiscreteUpdate(context))

        xnext = dut.get_state_output_port().Eval(context)
        xnext_expected = np.array([0.0004963, 0.2009950])
        numpy_compare.assert_float_allclose(xnext, xnext_expected, atol=1e-7)

        lambda_ = dut.get_contact_force_output_port().Eval(context)
        numpy_compare.assert_float_allclose(
            lambda_,
            kappa / (np.array([-1, 1]) @ xnext_expected - 0.1),
            atol=1e-7,
        )

        E = dut.GetDisturbanceMatrix(context=context)
        self.assertEqual(E.shape, (2, 1))

        dut.SetMeshcat(meshcat=None)

        if T is float:
            dut = dut.ToAutoDiffXd()
        else:
            dut = dut.ToScalarType[float]()

        dut.Clone()
        copy.copy(dut)
        copy.deepcopy(dut)

    @numpy_compare.check_nonsymbolic_types
    def test_quasidynamic_multibody_plant(self, T):
        dut = QuasidynamicMultibodyPlant_[T](
            dt=0.01,
            dynamics_smoothing=1e-8,
            geometry_smoothing=0.0,
        )

        object = dut.AddRigidBody(
            M_BBo_B=SpatialInertia.SolidBoxWithMass(
                mass=0.01, lx=0.4, ly=0.4, lz=0.4
            )
        )
        dut.RegisterCollisionGeometry(
            body=object,
            X_BG=RigidTransform(),
            shape=Box(0.4, 0.4, 0.4),
            friction_coefficient=0.05,
        )
        dut.RegisterVisualGeometry(
            body=object,
            X_BG=RigidTransform(),
            shape=Box(0.4, 0.4, 0.4),
            rgba=Rgba(0, 0, 1, 1),
            name="object",
        )

        pusher = dut.AddRigidBody()
        dut.RegisterCollisionGeometry(
            body=pusher,
            X_BG=RigidTransform(),
            shape=Sphere(0.05),
            friction_coefficient=0.05,
        )
        dut.RegisterVisualGeometry(
            body=pusher,
            X_BG=RigidTransform(),
            shape=Sphere(0.05),
            rgba=Rgba(1, 0, 0, 1),
            name="pusher",
        )
        dut.AddCartesianJoint(
            body_P=dut.world_body(),
            X_PJp=RigidTransform(),
            body_C=pusher,
            X_CJc=RigidTransform(),
            actuation_stiffness=[1.0, 0.9, 0.8],
        )

        dut.SetGravityVector([0.0, 0.0, 0.0])
        dut.Finalize()
        dut = dut.Clone()

        self.assertEqual(dut.num_contacts(), 1)
        self.assertEqual(dut.contact_dim(), 3)

        self.assertEqual(dut.get_actuation_input_port().get_name(), "u")
        self.assertEqual(dut.get_state_output_port().get_name(), "x")
        self.assertEqual(
            dut.get_contact_force_output_port().get_name(), "lambda"
        )

        context = dut.CreateDefaultContext()

        kappa = 1e-2
        dut.set_dynamics_smoothing(context=context, dynamics_smoothing=kappa)
        self.assertEqual(dut.get_dynamics_smoothing(context=context), kappa)

        x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0]
        dut.SetState(context=context, x=x)
        numpy_compare.assert_float_equal(dut.GetState(context=context), x)

        X_WB = dut.EvalPoseInWorld(context=context, body=object)
        numpy_compare.assert_float_equal(X_WB.translation(), x[-3:])
        self.assertTrue(X_WB.rotation().IsNearlyIdentity())

        dut.get_actuation_input_port().FixValue(context, [0.1, 0.0, 0.0])

        context.SetDiscreteState(dut.EvalUniquePeriodicDiscreteUpdate(context))

        xnext = dut.get_state_output_port().Eval(context)
        numpy_compare.assert_float_allclose(
            xnext,
            [0.0004963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3509950, 0.0, 0.0],
            atol=1e-7,
        )

        lambda_ = dut.get_contact_force_output_port().Eval(context)
        self.assertEqual(lambda_.shape, (3,))

        E = dut.GetDisturbanceMatrix(context=context)
        self.assertEqual(E.shape, (9, 1))

        self.assertEqual(dut.EvalCollisionJacobian(context).shape, (0, 9))
        self.assertEqual(dut.EvalCollisionSignedDistance(context).shape, (0,))

        dut.SetMeshcat(meshcat=None)

        if T is float:
            dut = dut.ToAutoDiffXd()
        else:
            dut = dut.ToScalarType[float]()

        dut.Clone()
        copy.copy(dut)
        copy.deepcopy(dut)

    @numpy_compare.check_nonsymbolic_types
    def test_quasidynamic_planar_plant(self, T):
        dut = QuasidynamicPlanarPlant_[T](
            dt=0.01,
            dynamics_smoothing=1e-8,
            geometry_smoothing=0.0,
        )

        object = dut.AddRigidBody(mass=0.01, p_BBcm_B=[0, 0], I_BBcm=0.0003)
        dut.RegisterCollisionGeometry(
            body=object,
            X_BG=RigidTransform2d(),
            shape=Rectangle(0.4, 0.4),
            friction_coefficient=0.05,
        )
        dut.RegisterVisualGeometry(
            body=object,
            X_BG=RigidTransform2d(),
            shape=Rectangle(0.4, 0.4),
            rgba=Rgba(0, 0, 1, 1),
            name="object",
        )
        dut.RegisterVisualGeometry(
            body=object,
            X_BG=RigidTransform(),
            shape=Box(0.4, 0.4, 0.1),
            rgba=Rgba(0, 0, 1, 1),
            name="object_2",
        )

        pusher = dut.AddRigidBody()
        dut.RegisterCollisionGeometry(
            body=pusher,
            X_BG=RigidTransform2d(),
            shape=Circle(0.05),
            friction_coefficient=0.05,
        )
        dut.RegisterVisualGeometry(
            body=pusher,
            X_BG=RigidTransform2d(),
            shape=Circle(0.05),
            rgba=Rgba(1, 0, 0, 1),
            name="pusher",
        )
        dut.AddCartesianJoint(
            body_P=dut.world_body(),
            X_PJp=RigidTransform2d(),
            body_C=pusher,
            X_CJc=RigidTransform2d(),
            actuation_stiffness=[1.0, 0.9],
        )

        dut.SetGravityVector([0.0, 0.0])
        dut.Finalize()
        dut = dut.Clone()

        self.assertEqual(dut.num_contacts(), 1)
        self.assertEqual(dut.contact_dim(), 2)

        self.assertEqual(dut.get_actuation_input_port().get_name(), "u")
        self.assertEqual(dut.get_state_output_port().get_name(), "x")
        self.assertEqual(
            dut.get_contact_force_output_port().get_name(), "lambda"
        )

        context = dut.CreateDefaultContext()

        kappa = 1e-2
        dut.set_dynamics_smoothing(context=context, dynamics_smoothing=kappa)
        self.assertEqual(dut.get_dynamics_smoothing(context=context), kappa)

        x = [0.0, 0.0, 0.35, 0.0, 0.0]
        dut.SetState(context=context, x=x)
        numpy_compare.assert_float_equal(dut.GetState(context=context), x)

        X_WB = dut.EvalPoseInWorld(context=context, body=object)
        numpy_compare.assert_float_equal(X_WB.translation(), x[2:4])
        numpy_compare.assert_float_equal(X_WB.angle(), x[4])

        dut.get_actuation_input_port().FixValue(context, [0.1, 0.0])

        context.SetDiscreteState(dut.EvalUniquePeriodicDiscreteUpdate(context))

        xnext = dut.get_state_output_port().Eval(context)
        numpy_compare.assert_float_allclose(
            xnext, [0.0004963, 0.0, 0.3509950, 0.0, 0.0], atol=1e-7
        )

        lambda_ = dut.get_contact_force_output_port().Eval(context)
        self.assertEqual(lambda_.shape, (2,))

        E = dut.GetDisturbanceMatrix(context=context)
        self.assertEqual(E.shape, (5, 1))

        self.assertEqual(dut.EvalCollisionJacobian(context).shape, (0, 5))
        self.assertEqual(dut.EvalCollisionSignedDistance(context).shape, (0,))

        dut.SetMeshcat(meshcat=None)

        if T is float:
            dut = dut.ToAutoDiffXd()
        else:
            dut = dut.ToScalarType[float]()

        dut.Clone()
        copy.copy(dut)
        copy.deepcopy(dut)
