import os
import numpy as np
from pydrake.all import (
    QuasidynamicPlanarPlant,
    Box,
    RigidTransform,
    RigidTransform2d,
    RotationMatrix,
    Mesh,
    Rgba,
    BodyIndex,
    Shape,
    RollPitchYaw,
    Cylinder,
    Sphere,
    Obround,
)


def AddPlanarIiwa(
    plant: QuasidynamicPlanarPlant,
    name: str = "planar_iiwa",
    X_WR: RigidTransform2d = RigidTransform2d(),
    friction_coefficient: float = 0.2,
    actuation_stiffness: list[float] = [1.0, 1.0, 1.0],
    color: Rgba = Rgba(1, 0, 0, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    assert len(actuation_stiffness) == 3
    pi = np.pi

    kwargs = {
        "friction_coefficient": friction_coefficient,
        "color": color,
    }

    # base
    base = plant.world_body()
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        body=base,
        name=name + "/base",
        X_BG_prox=None,
        shape_prox=None,
        X_BG_vis=RigidTransform(
            RotationMatrix.MakeZRotation(X_WR.angle()),
            np.concatenate((X_WR.translation(), [-0.34])),
        ),
        shape_vis=FindMesh("link_0.gltf"),
        **kwargs,
    )

    # segment_1
    segment_1 = plant.AddRigidBody()
    plant.AddRevoluteJoint(
        body_P=base,
        X_PJp=X_WR,
        body_C=segment_1,
        X_CJc=RigidTransform2d(),
        actuation_stiffness=actuation_stiffness[0],
    )
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        body=segment_1,
        name=name + "/segment_1",
        X_BG_prox=RigidTransform2d([0.1325, 0]),
        shape_prox=Obround(0.0675, 0.265),
        X_BG_vis=[
            XyzRpy(0, 0, -0.1825, 0, 0, 0),
            XyzRpy(0, 0, 0.0005, pi / 2, -pi / 2, -pi),
            XyzRpy(0.184, 0, 0.0005, -pi / 2, 0, -pi / 2),
        ],
        shape_vis=[
            FindMesh("link_1.gltf"),
            FindMesh("link_2.gltf"),
            FindMesh("link_3.gltf"),
        ],
        **kwargs,
    )

    # segment_2
    segment_2 = plant.AddRigidBody()
    plant.AddRevoluteJoint(
        body_P=segment_1,
        X_PJp=RigidTransform2d([0.4, 0]),
        body_C=segment_2,
        X_CJc=RigidTransform2d(),
        actuation_stiffness=actuation_stiffness[1],
    )
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        body=segment_2,
        name=name + "/segment_2",
        X_BG_prox=RigidTransform2d([0.1225, 0]),
        shape_prox=Obround(0.0675, 0.245),
        X_BG_vis=[
            XyzRpy(0, 0, 0.0005, 0, 0, -pi / 2),
            XyzRpy(0.184, 0, 0.0005, pi / 2, 0, pi / 2),
        ],
        shape_vis=[
            FindMesh("link_4.gltf"),
            FindMesh("link_5.gltf"),
        ],
        **kwargs,
    )

    # segment_3
    segment_3 = plant.AddRigidBody()
    plant.AddRevoluteJoint(
        body_P=segment_2,
        X_PJp=RigidTransform2d([0.4, 0]),
        body_C=segment_3,
        X_CJc=RigidTransform2d(),
        actuation_stiffness=actuation_stiffness[2],
    )
    RegisterCollisionAndVisualGeometry(
        plant=plant,
        body=segment_3,
        name=name + "/segment_3",
        X_BG_prox=RigidTransform2d([0.04925, 0]),
        shape_prox=Obround(0.0675, 0.1385),
        X_BG_vis=[
            XyzRpy(0, 0, 0.0605, 0, pi, -pi / 2),
            XyzRpy(0.0805, 0, 0.0005, 0, pi / 2, 0),
            XyzRpy(0.1185, 0, 0.0005, 0, 0, 0),
            XyzRpy(0.0860, 0, 0.0005, 0, pi / 2, 0),
        ],
        shape_vis=[
            FindMesh("link_6.gltf"),
            FindMesh("link_7.gltf"),
            Sphere(0.0675),
            Cylinder(0.0675, 0.065),
        ],
        **kwargs,
    )

    # Joint limits
    ub = np.deg2rad([170, 120, 120])
    lb = -ub
    return lb, ub


def RegisterCollisionAndVisualGeometry(
    plant: QuasidynamicPlanarPlant,
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
                Rgba(1, 1, 1, 1),
                name if len(shape_vis) == 1 else name + "/" + str(i),
            )


def FindMesh(filename: str) -> Mesh:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(curr_dir, "iiwa_meshes")
    file_path = os.path.join(meshes_dir, filename)
    return Mesh(file_path)


def XyzRpy(x, y, z, roll, pitch, yaw) -> RigidTransform:
    rpy = RollPitchYaw(roll, pitch, yaw)
    xyz = [x, y, z]
    return RigidTransform(rpy, xyz)
