import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from collections import deque
from pupil_apriltags import Detector
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    RotationMatrix,
)


class PlanarPositionDetector(LeafSystem):
    def __init__(self, max_stored_images=999):
        super().__init__()

        # RealSense Camera
        w, h = 1280, 720
        fps = 30

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        pipeline.start(config)

        profile = pipeline.get_active_profile()
        intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        fx, fy = intr.fx, intr.fy
        cx, cy = intr.ppx, intr.ppy
        camera_params = (fx, fy, cx, cy)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array(intr.coeffs)

        self._pipeline = pipeline
        self._camera_params = camera_params
        self._camera_matrix = camera_matrix
        self._distortion_coeffs = distortion_coeffs

        # Map of tag_id to tag_size (meters)
        self._tag_sizes = {
            0: 0.06667,
            1: 0.06667,
            10: 0.099,
        }
        # AprilTag detector
        self._detector = Detector(families="tagStandard41h12")

        # Pool camera frame
        color_image = self._GetImage(blocking=True)
        planar_pos = self._DetectPlanarPosition(color_image)
        if planar_pos is None:
            raise RuntimeError("Missing tags and thus cannot detect position")

        # Declare state
        state_index = self.DeclareDiscreteState(planar_pos)
        self.DeclareStateOutputPort("planar_position", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/fps,
            offset_sec=0.0,
            update=self.DiscreteUpdate,
        )

        # Image storage queue
        self._images = deque(maxlen=max_stored_images)

    def Close(self):
        self._pipeline.stop()

    def DiscreteUpdate(self, context, discrete_values):
        color_image = self._GetImage()
        if color_image is None:
            return
        self._images.append(color_image)

        planar_pos = self._DetectPlanarPosition(color_image)
        if planar_pos is not None:
            discrete_values.set_value(planar_pos)

    def SaveImages(self, filename):
        dir_path = Path(filename).parent
        if str(dir_path) != ".":
            dir_path.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(self._images):
            cv2.imwrite(filename + f"_{i:03d}.png", image)

    def _GetImage(self, blocking=False):
        if blocking:
            newest_frame = self._pipeline.wait_for_frames()
        else:
            newest_frame = None
            while True:
                frame = self._pipeline.poll_for_frames()
                if not frame:
                    break
                newest_frame = frame

        if newest_frame is None:
            return None

        color_frame = newest_frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.undistort(
            color_image,
            self._camera_matrix,
            self._distortion_coeffs,
        )
        return color_image

    def _DetectPlanarPosition(self, color_image) -> np.ndarray | None:
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        detections = self._DetectTags(gray_image)
        planar_pos = self._CalcPlanarPosition(detections)
        return planar_pos

    def _DetectTags(self, gray_image):
        all_detections = []
        for tag_size in set(self._tag_sizes.values()):
            detections = self._detector.detect(
                gray_image,
                estimate_tag_pose=True,
                camera_params=self._camera_params,
                tag_size=tag_size
            )
            current_ids = [tag_id for tag_id,
                           size in self._tag_sizes.items() if size == tag_size]
            detections = [d for d in detections if d.tag_id in current_ids]
            all_detections.extend(detections)
        return all_detections

    def _CalcPlanarPosition(self, detections):
        X_CA, X_CB, X_CT = None, None, None
        for det in detections:
            pose = RigidTransform(RotationMatrix(det.pose_R), det.pose_t)
            if det.tag_id == 0:
                X_CA = pose
            elif det.tag_id == 1:
                X_CB = pose
            elif det.tag_id == 10:
                X_CT = pose
        if any([X_CA is None, X_CB is None, X_CT is None]):
            return None

        p_TA = X_CT.inverse() @ X_CA.translation()
        p_TB = X_CT.inverse() @ X_CB.translation()
        p_TO = (p_TA + p_TB) / 2

        x_TO = p_TB - p_TA
        x_TO = x_TO / np.linalg.norm(x_TO)
        z_TO = np.array([0, 0, -1])
        y_TO = np.cross(z_TO, x_TO)
        R_TO = RotationMatrix(np.vstack((x_TO, y_TO, z_TO)).T)

        X_TO = RigidTransform(R_TO, p_TO)
        X_OT = X_TO.inverse()
        R_OT = X_OT.rotation().matrix()
        p_OT = X_OT.translation()

        pos = p_OT[:2] - np.array([0.0, 0.04])
        angle = np.atan2(R_OT[1, 0], R_OT[0, 0])
        return np.concatenate((pos, [angle]))
