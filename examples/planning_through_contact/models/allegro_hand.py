import os
from pydrake.all import (
    QuasidynamicMultibodyPlant,
    Box,
    RigidTransform,
    Mesh,
    Rgba,
    BodyIndex,
    Shape,
    RollPitchYaw,
    Capsule,
    Sphere,
)


def AddAllegroHand(
    plant: QuasidynamicMultibodyPlant,
    name: str = "allegro_hand",
    friction_coefficient: float = 0.2,
    fingertip_friction_coefficient: float = None,
    actuation_stiffness: float = 1.0,
    color: Rgba = Rgba(1, 0, 0, 0.5),
) -> tuple[list[float], list[float]]:
    if fingertip_friction_coefficient is None:
        fingertip_friction_coefficient = friction_coefficient

    kwargs = {
        "color": color,
    }

    # hand_root
    hand_root = plant.world_body()
    X_W_handroot = RigidTransform()
    box = Box(0.0408, 0.113 / 2, 0.095 / 2)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/hand_root",
        body=hand_root,
        X_BG_prox=[
            RigidTransform([-0.0093, +0.113 / 4, 0.0475 + 0.095 / 4]),
            RigidTransform([-0.0093, -0.113 / 4, 0.0475 + 0.095 / 4]),
            RigidTransform([-0.0093, -0.113 / 4, 0.0475 - 0.095 / 4]),
            RigidTransform([-0.0093, +0.113 / 4, 0.0475 - 0.095 / 4]),
        ],
        friction_coefficient=friction_coefficient,
        shape_prox=[box] * 4,
        X_BG_vis=RigidTransform([0, 0, 0.095]),
        shape_vis=FindMesh("base_link.gltf"),
        **kwargs,
    )

    # link_0
    link_0 = plant.AddRigidBody()
    X_W_link0 = XyzRpy(0, 0.0435, 0.093458, -0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_0",
        body=link_0,
        X_BG_prox=None,
        shape_prox=None,
        friction_coefficient=None,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_0.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=hand_root,
        X_PJp=X_W_handroot.inverse() @ X_W_link0,
        body_C=link_0,
        X_CJc=RigidTransform(),
        axis=[0, 0, 1],
        actuation_stiffness=actuation_stiffness,
    )

    # link_1
    link_1 = plant.AddRigidBody()
    X_W_link1 = XyzRpy(0, 0.044929, 0.109796, -0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_1",
        body=link_1,
        X_BG_prox=RigidTransform([0, 0, 0.027]),
        shape_prox=Capsule(0.012, 0.054),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_1.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_0,
        X_PJp=X_W_link0.inverse() @ X_W_link1,
        body_C=link_1,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_2
    link_2 = plant.AddRigidBody()
    X_W_link2 = XyzRpy(0, 0.049636, 0.16359, -0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_2",
        body=link_2,
        X_BG_prox=RigidTransform([0, 0, 0.0312]),
        shape_prox=Capsule(0.012, 0.012),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_2.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_1,
        X_PJp=X_W_link1.inverse() @ X_W_link2,
        body_C=link_2,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_3
    link_3 = plant.AddRigidBody()
    X_W_link3 = XyzRpy(0, 0.052983, 0.201844, -0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_3",
        body=link_3,
        X_BG_prox=RigidTransform([0, 0, 0.0267]),
        shape_prox=Sphere(0.012),
        friction_coefficient=fingertip_friction_coefficient,
        X_BG_vis=[RigidTransform(), RigidTransform([0, 0, 0.0267])],
        shape_vis=[FindMesh("link_3.0.gltf"), FindMesh("link_3.0_tip.gltf")],
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_2,
        X_PJp=X_W_link2.inverse() @ X_W_link3,
        body_C=link_3,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_4
    link_4 = plant.AddRigidBody()
    X_W_link4 = XyzRpy(0, 0, 0.0957, 0, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_4",
        body=link_4,
        X_BG_prox=None,
        shape_prox=None,
        friction_coefficient=None,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_0.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=hand_root,
        X_PJp=X_W_handroot.inverse() @ X_W_link4,
        body_C=link_4,
        X_CJc=RigidTransform(),
        axis=[0, 0, 1],
        actuation_stiffness=actuation_stiffness,
    )

    # link_5
    link_5 = plant.AddRigidBody()
    X_W_link5 = XyzRpy(0, 0, 0.1121, 0, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_5",
        body=link_5,
        X_BG_prox=RigidTransform([0, 0, 0.027]),
        shape_prox=Capsule(0.012, 0.054),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_1.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_4,
        X_PJp=X_W_link4.inverse() @ X_W_link5,
        body_C=link_5,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_6
    link_6 = plant.AddRigidBody()
    X_W_link6 = XyzRpy(0, 0, 0.1661, 0, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_6",
        body=link_6,
        X_BG_prox=RigidTransform([0, 0, 0.0312]),
        shape_prox=Capsule(0.012, 0.012),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_2.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_5,
        X_PJp=X_W_link5.inverse() @ X_W_link6,
        body_C=link_6,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_7
    link_7 = plant.AddRigidBody()
    X_W_link7 = XyzRpy(0, 0, 0.2045, 0, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_7",
        body=link_7,
        X_BG_prox=RigidTransform([0, 0, 0.0267]),
        shape_prox=Sphere(0.012),
        friction_coefficient=fingertip_friction_coefficient,
        X_BG_vis=[RigidTransform(), RigidTransform([0, 0, 0.0267])],
        shape_vis=[FindMesh("link_3.0.gltf"), FindMesh("link_3.0_tip.gltf")],
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_6,
        X_PJp=X_W_link6.inverse() @ X_W_link7,
        body_C=link_7,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_8
    link_8 = plant.AddRigidBody()
    X_W_link8 = XyzRpy(0, -0.0435, 0.093458, 0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_8",
        body=link_8,
        X_BG_prox=None,
        shape_prox=None,
        friction_coefficient=None,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_0.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=hand_root,
        X_PJp=X_W_handroot.inverse() @ X_W_link8,
        body_C=link_8,
        X_CJc=RigidTransform(),
        axis=[0, 0, 1],
        actuation_stiffness=actuation_stiffness,
    )

    # link_9
    link_9 = plant.AddRigidBody()
    X_W_link9 = XyzRpy(0, -0.044929, 0.109796, 0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_9",
        body=link_9,
        X_BG_prox=RigidTransform([0, 0, 0.027]),
        shape_prox=Capsule(0.012, 0.054),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_1.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_8,
        X_PJp=X_W_link8.inverse() @ X_W_link9,
        body_C=link_9,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_10
    link_10 = plant.AddRigidBody()
    X_W_link10 = XyzRpy(0, -0.049636, 0.16359, 0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_10",
        body=link_10,
        X_BG_prox=RigidTransform([0, 0, 0.0312]),
        shape_prox=Capsule(0.012, 0.012),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_2.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_9,
        X_PJp=X_W_link9.inverse() @ X_W_link10,
        body_C=link_10,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_11
    link_11 = plant.AddRigidBody()
    X_W_link11 = XyzRpy(0, -0.052983, 0.201844, 0.087267, 0, 0)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_11",
        body=link_11,
        X_BG_prox=RigidTransform([0, 0, 0.0267]),
        shape_prox=Sphere(0.012),
        friction_coefficient=fingertip_friction_coefficient,
        X_BG_vis=[RigidTransform(), RigidTransform([0, 0, 0.0267])],
        shape_vis=[FindMesh("link_3.0.gltf"), FindMesh("link_3.0_tip.gltf")],
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_10,
        X_PJp=X_W_link10.inverse() @ X_W_link11,
        body_C=link_11,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_12
    link_12 = plant.AddRigidBody()
    X_W_link12 = XyzRpy(-0.0182, 0.019333, 0.049013, 3.14159, -1.48353, 1.5708)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_12",
        body=link_12,
        X_BG_prox=None,
        shape_prox=None,
        friction_coefficient=None,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_12.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=hand_root,
        X_PJp=X_W_handroot.inverse() @ X_W_link12,
        body_C=link_12,
        X_CJc=RigidTransform(),
        axis=[-1, 0, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_13
    link_13 = plant.AddRigidBody()
    X_W_link13 = XyzRpy(-0.0132, 0.056728, 0.018638, 3.14159, -1.48353, 1.5708)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_13",
        body=link_13,
        X_BG_prox=None,
        shape_prox=None,
        friction_coefficient=None,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_13.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_12,
        X_PJp=X_W_link12.inverse() @ X_W_link13,
        body_C=link_13,
        X_CJc=RigidTransform(),
        axis=[0, 0, 1],
        actuation_stiffness=actuation_stiffness,
    )

    # link_14
    link_14 = plant.AddRigidBody()
    X_W_link14 = XyzRpy(-0.0132, 0.074361, 0.017096, 3.14159, -1.48353, 1.5708)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_14",
        body=link_14,
        X_BG_prox=RigidTransform([0, 0, 0.0244]),
        shape_prox=Capsule(0.012, 0.030),
        friction_coefficient=friction_coefficient,
        X_BG_vis=RigidTransform(),
        shape_vis=FindMesh("link_14.0.gltf"),
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_13,
        X_PJp=X_W_link13.inverse() @ X_W_link14,
        body_C=link_14,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # link_15
    link_15 = plant.AddRigidBody()
    X_W_link15 = XyzRpy(-0.0132, 0.125565, 0.012616, 3.14159, -1.48353, 1.5708)
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        name=name + "/link_15",
        body=link_15,
        X_BG_prox=RigidTransform([0, 0, 0.0273]),
        shape_prox=Capsule(0.012, 0.030),
        friction_coefficient=fingertip_friction_coefficient,
        X_BG_vis=[RigidTransform(), RigidTransform([0, 0, 0.0423])],
        shape_vis=[FindMesh("link_15.0.gltf"), FindMesh("link_15.0_tip.gltf")],
        **kwargs,
    )
    plant.AddRevoluteJoint(
        body_P=link_14,
        X_PJp=X_W_link14.inverse() @ X_W_link15,
        body_C=link_15,
        X_CJc=RigidTransform(),
        axis=[0, 1, 0],
        actuation_stiffness=actuation_stiffness,
    )

    # Joint limits
    lb = [-0.57, -0.296, -0.274, -0.327] * 3 + [0.36, -0.204, -0.289, -0.26]
    ub = [0.57, 1.71, 1.809, 1.718] * 3 + [1.496, 1.263, 1.744, 1.819]
    return lb, ub


def RegisterCollisionAndVisualGeometry(
    plant: QuasidynamicMultibodyPlant,
    body: BodyIndex,
    X_BG_prox: RigidTransform | list[RigidTransform] | None,
    shape_prox: Shape | list[Shape] | None,
    friction_coefficient: float | None,
    X_BG_vis: RigidTransform | list[RigidTransform] | None,
    shape_vis: Shape | list[Shape] | None,
    name: str,
    color: Rgba,
) -> None:
    # Register collision geometry.
    if shape_prox is not None:
        if not isinstance(shape_prox, (list, tuple)):
            X_BG_prox = [X_BG_prox]
            shape_prox = [shape_prox]
        for i in range(len(shape_prox)):
            plant.RegisterCollisionGeometry(
                body, X_BG_prox[i], shape_prox[i], friction_coefficient
            )
    # Register collision geometry as visual geometry.
    if shape_prox is not None:
        pos = name.find("/")
        name_prox = name[:pos] + "_proximity" + name[pos:]
        for i in range(len(shape_prox)):
            plant.RegisterVisualGeometry(
                body,
                X_BG_prox[i],
                shape_prox[i],
                color,
                name_prox if len(shape_prox) == 1 else name_prox + "/" + str(i),
            )
    # Register mesh geometry as visual geometry.
    if shape_vis is not None:
        if not isinstance(shape_vis, (list, tuple)):
            X_BG_vis = [X_BG_vis]
            shape_vis = [shape_vis]
        for i in range(len(shape_vis)):
            plant.RegisterVisualGeometry(
                body,
                X_BG_vis[i],
                shape_vis[i],
                color,
                name if len(shape_vis) == 1 else name + "/" + str(i),
            )


def FindMesh(filename: str) -> Mesh:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(curr_dir, "allegro_hand_meshes")
    file_path = os.path.join(meshes_dir, filename)
    return Mesh(file_path)


def XyzRpy(x, y, z, roll, pitch, yaw) -> RigidTransform:
    rpy = RollPitchYaw(roll, pitch, yaw)
    xyz = [x, y, z]
    return RigidTransform(rpy, xyz)
