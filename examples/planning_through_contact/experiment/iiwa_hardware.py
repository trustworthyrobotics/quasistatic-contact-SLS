import numpy as np
from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
)
from pydrake.all import (
    Gain,
    OutputPort,
    Adder,
    Diagram,
    DiagramBuilder,
    IiwaCommandSender,
    IiwaControlMode,
    IiwaStatusReceiver,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    position_enabled,
    torque_enabled,
    Multiplexer,
    ConstantVectorSource,
    MatrixGain,
    DrakeLcmInterface,
    DrakeLcm,
    LcmInterfaceSystem,
    LeafSystem,
    Simulator,
)


LCM_CHANNEL_SUFFIXS = ["", "_2"]


def _MakeIiwaRobot(
    lcm: DrakeLcmInterface,
    control_mode=IiwaControlMode.kPositionAndTorque,
    lcm_channel_suffix=""
) -> Diagram:
    assert isinstance(control_mode, IiwaControlMode)

    builder = DiagramBuilder()

    # Publish IIWA command.
    # IIWA driver won't respond faster than 1000Hz in torque_only mode and
    # 200Hz in other modes
    publish_period = 0.005
    if control_mode == IiwaControlMode.kTorqueOnly:
        publish_period = 0.001

    iiwa_command_sender = builder.AddSystem(
        IiwaCommandSender(control_mode=control_mode)
    )
    iiwa_command_publisher = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel="IIWA_COMMAND" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_command,
            lcm=lcm,
            publish_period=publish_period,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(
        iiwa_command_sender.get_output_port(),
        iiwa_command_publisher.get_input_port(),
    )
    if position_enabled(control_mode):
        builder.ExportInput(
            iiwa_command_sender.get_position_input_port(),
            "position",
        )
    if torque_enabled(control_mode):
        builder.ExportInput(
            iiwa_command_sender.get_torque_input_port(),
            "torque",
        )
    # Receive IIWA status and populate the output ports.
    iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
    iiwa_status_subscriber = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_status,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )
    builder.Connect(
        iiwa_status_subscriber.get_output_port(),
        iiwa_status_receiver.get_input_port(),
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_position_commanded_output_port(),
        "position_commanded",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_position_measured_output_port(),
        "position_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_velocity_estimated_output_port(),
        "velocity_estimated",
    )

    # These are negated as outlined in drake/manipulation/README.
    def NegatedPort(
        builder: DiagramBuilder, output_port: OutputPort
    ) -> OutputPort:
        negater = builder.AddNamedSystem(
            f"signflip_{output_port.get_name()}", Gain(-1,
                                                       size=output_port.size())
        )
        builder.Connect(output_port, negater.get_input_port())
        return negater.get_output_port()

    builder.ExportOutput(
        NegatedPort(
            builder=builder,
            output_port=iiwa_status_receiver.get_torque_commanded_output_port(),
        ),
        "torque_commanded",
    )
    builder.ExportOutput(
        NegatedPort(
            builder=builder,
            output_port=iiwa_status_receiver.get_torque_measured_output_port(),
        ),
        "torque_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_torque_external_output_port(),
        "torque_external",
    )

    mux = builder.AddSystem(Multiplexer(input_sizes=[7, 7]))
    builder.Connect(
        iiwa_status_receiver.get_position_measured_output_port(),
        mux.get_input_port(0),
    )
    builder.Connect(
        iiwa_status_receiver.get_velocity_estimated_output_port(),
        mux.get_input_port(1),
    )
    builder.ExportOutput(
        mux.get_output_port(),
        "state_estimated",
    )

    return builder.Build()


def _MakeLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm


def HomeIiwaRobot(robot_index: int, home_joint_angles: list[float]):
    assert 0 <= robot_index and robot_index < len(LCM_CHANNEL_SUFFIXS)
    assert len(home_joint_angles) == 7

    class HomingController(LeafSystem):
        def __init__(self):
            super().__init__()
            self.x_target = np.array(home_joint_angles)

            self.DeclareVectorInputPort("position_estimated", 7)

            state_index = self.DeclareDiscreteState(7)
            self.DeclareStateOutputPort("position_command", state_index)
            self.DeclarePeriodicDiscreteUpdateEvent(
                period_sec=0.005,
                offset_sec=0.0,
                update=self.DiscreteUpdate
            )

        def DiscreteUpdate(self, context, discrete_state):
            x_target = self.x_target
            x = self.get_input_port().Eval(context)
            eps = 0.01
            delta_x = np.clip(x_target - x, -eps, eps)
            x_next = x + delta_x
            discrete_state.set_value(x_next)

    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)
    iiwa = builder.AddSystem(
        _MakeIiwaRobot(
            lcm_channel_suffix=LCM_CHANNEL_SUFFIXS[robot_index],
            lcm=lcm,
        )
    )
    controller = builder.AddSystem(HomingController())
    builder.Connect(
        iiwa.GetOutputPort("position_measured"),
        controller.get_input_port()
    )
    builder.Connect(
        controller.get_output_port(),
        iiwa.GetInputPort("position")
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    iiwa_context = diagram.GetSubsystemContext(iiwa, context)

    while context.get_time() < 10.0:
        simulator.AdvanceTo(context.get_time() + 0.1)
        joint_angles = iiwa.GetOutputPort(
            "position_measured").Eval(iiwa_context)
        if np.all(np.abs(joint_angles - home_joint_angles) < np.deg2rad(0.5)):
            break


def MakePlanarBimanualStation() -> Diagram:
    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)

    for i, lcm_channel_suffix in enumerate(LCM_CHANNEL_SUFFIXS):
        iiwa = builder.AddSystem(
            _MakeIiwaRobot(
                lcm=lcm,
                control_mode=IiwaControlMode.kPositionAndTorque,
                lcm_channel_suffix=lcm_channel_suffix,
            )
        )

        nominal_position_source = builder.AddSystem(
            ConstantVectorSource([0, np.pi/2, -np.pi/2, 0, 0, 0, 0])
        )

        B = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0],
        ]).T
        gain = builder.AddSystem(
            MatrixGain(B)
        )

        adder = builder.AddSystem(Adder(num_inputs=2, size=7))

        builder.Connect(
            gain.get_output_port(),
            adder.get_input_port(0),
        )
        builder.Connect(
            nominal_position_source.get_output_port(),
            adder.get_input_port(1),
        )
        builder.Connect(
            adder.get_output_port(),
            iiwa.GetInputPort("position"),
        )
        builder.ExportInput(
            gain.get_input_port(),
            f"input_{i}"
        )

        gain = builder.AddSystem(
            MatrixGain(B.T)
        )
        builder.Connect(
            iiwa.GetOutputPort("position_measured"),
            gain.get_input_port(),
        )
        builder.ExportOutput(
            gain.get_output_port(),
            f"position_{i}"
        )

    return builder.Build()


def HomePlanarBimanualStation(robot0_pos, robot1_pos):
    assert len(robot0_pos) == 3
    assert len(robot1_pos) == 3

    def get_joint_angles(robot_pos):
        return [
            robot_pos[0],
            np.pi/2,
            -np.pi/2,
            robot_pos[1],
            0,
            -robot_pos[2],
            0,
        ]

    HomeIiwaRobot(
        robot_index=0,
        home_joint_angles=get_joint_angles(robot0_pos),
    )
    HomeIiwaRobot(
        robot_index=1,
        home_joint_angles=get_joint_angles(robot1_pos),
    )
