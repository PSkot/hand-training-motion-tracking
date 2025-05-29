import cv2
import numpy as np
import torch
from run_inference import crop_resize_new
import argparse
import src.landmarks as lm
import json
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import sys
import src.model_setup as model_setup
import src.helpers as helpers
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import train_kmeans as train_kmeans
from scipy.spatial.transform import Rotation as Rscipy

CLUSTER_MAPPING = helpers.read_user_feedback()
IMG_OUTPUT_FOLDER = "../images"


def reapply_rotation(aligned_landmarks, R_original, axes_to_restore=[]):
    """
    Reapply selected axis rotations from the original rotation matrix R to aligned landmarks.

    Args:
        aligned_landmarks (np.ndarray): (N, 3) aligned landmarks
        R_original (np.ndarray): (3, 3) original rotation matrix from alignment
        axes_to_restore (list of str): Axes to restore, e.g., ["X", "Z"]

    Returns:
        rotated (np.ndarray): (N, 3) landmarks with partial rotation reapplied
    """

    if not axes_to_restore:
        return aligned_landmarks

    axes_to_restore = set(axis.upper() for axis in axes_to_restore)
    rot = Rscipy.from_matrix(R_original)
    euler = rot.as_euler("XYZ", degrees=False)

    # Zero out unwanted axes
    if "X" not in axes_to_restore:
        euler[0] = 0
    if "Y" not in axes_to_restore:
        euler[1] = 0
    if "Z" not in axes_to_restore:
        euler[2] = 0

    # Build filtered rotation matrix
    R_filtered = Rscipy.from_euler("XYZ", euler).as_matrix()

    # Reapply to aligned landmarks
    rotated = (R_filtered @ aligned_landmarks.T).T
    return rotated


class HandExerciseApp(QWidget):
    def __init__(
        self,
        resnet,
        kmeans,
        t_landmarks,
        init_exercise="0000",
        max_line_dist=0.6,
        display_teacher=True,
        display_dist_lines=True,
    ):
        super().__init__()
        self.setWindowTitle("HandyApp")
        self.resnet = resnet
        self.kmeans = kmeans
        self.t_landmarks_dict = t_landmarks
        self.landmarks = None
        self.exercise = "fstretch"
        self.exercise_state = init_exercise
        self.t_landmarks = t_landmarks[self.exercise_state]
        self.t_landmarks = lm.align_landmarks_stable(self.t_landmarks.reshape(21, 3))[0]
        self.max_line_dist = max_line_dist
        self.display_teacher = display_teacher
        self.display_dist_lines = display_dist_lines

        # Layout
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Webcam display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # OpenGL Widget
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=0.6, elevation=10)
        self.view.setMinimumSize(600, 600)
        layout.addWidget(self.view)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((21, 3)), size=10, color=(1, 1, 1, 1)
        )
        # Reference hand scatter
        if self.display_teacher:
            self.ref_scatter = gl.GLScatterPlotItem(
                pos=self.t_landmarks, size=10, color=(1, 0, 0, 1)  # red
            )

        self.view.addItem(self.scatter)

        if self.display_teacher:
            self.view.addItem(self.ref_scatter)

        self.lines = []
        self.bone_pairs = helpers.read_hand_keypoints()
        for _ in self.bone_pairs:
            line = gl.GLLinePlotItem()
            self.view.addItem(line)
            self.lines.append(line)

        if self.display_teacher:
            self.ref_lines = []
            for a, b in self.bone_pairs:
                pts = np.array([self.t_landmarks[a], self.t_landmarks[b]])
                line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 1), width=2)
                self.view.addItem(line)
                self.ref_lines.append(line)

        if self.display_teacher and self.display_dist_lines:
            self.dotted_lines = []
            for _ in range(21):
                dotted = gl.GLLinePlotItem(
                    pos=np.zeros((2, 3)), color=(0, 1, 0, 1), width=1
                )
                self.view.addItem(dotted)
                self.dotted_lines.append(dotted)

        # Set camera position and center
        self.view.setCameraPosition(
            pos=pg.Vector(1, 2, 0), distance=2.5, elevation=-90, azimuth=0
        )

        # Set view center (i.e., origin point of axes)
        self.view.opts["center"] = pg.Vector(1, 0, 0)

        # Optional: Add grid planes aligned with axes for orientation
        grid_x = gl.GLGridItem()
        grid_x.rotate(90, 0, 1, 0)  # X-Z plane
        grid_x.translate(1, 0.5, 0.25)
        self.view.addItem(grid_x)

        grid_y = gl.GLGridItem()
        grid_y.rotate(90, 1, 0, 0)  # Y-Z plane
        grid_y.translate(1, 0.5, 0.25)
        self.view.addItem(grid_y)

        grid_z = gl.GLGridItem()  # X-Y plane
        grid_z.translate(1, 0.5, 0.25)
        self.view.addItem(grid_z)

        # Start webcam
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Update dummy keypoints (replace this with your model inference)
        # Preprocess the frame as needed for your model
        input_frame = process_frame(frame)
        self.input_frame = frame

        self.landmarks = self.resnet.predict(input_frame)

        # Predict the cluster
        cluster = kmeans_model.predict(
            # train_kmeans.wrist_relative_distances(landmarks.reshape(21, 3)).reshape(
            #     1, -1
            lm.align_landmarks_stable(self.landmarks.reshape(21, 3))[0].reshape(-1, 63)
            # )
        )[0]

        # Update scatter
        aligned, R = lm.align_landmarks_stable(self.landmarks.reshape(21, 3))
        self.hand_keypoints = reapply_rotation(aligned, R)  # [0]

        distances = np.linalg.norm(self.hand_keypoints - self.t_landmarks, axis=1)
        max_dist = self.max_line_dist
        norm_dists = np.clip(distances / max_dist, 0, 1)

        colors = np.array([self.gradient_color(t) for t in norm_dists])

        if self.display_teacher:
            self.ref_scatter.setData(pos=self.t_landmarks, color=colors)

        cluster_item = (
            CLUSTER_MAPPING.get(self.exercise).get(self.exercise_state).get(cluster)
        )

        # Draw "border"
        cv2.putText(
            frame,
            f"{cluster_item.get("message")}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            5,
        )

        # Draw text
        cv2.putText(
            frame,
            f"{cluster_item.get("message")}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            cluster_item.get("color"),
            2,
        )

        # Show webcam frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        # Update OpenGL plot
        self.scatter.setData(pos=self.hand_keypoints)

        for i, (a, b) in enumerate(self.bone_pairs):
            pts = np.array([self.hand_keypoints[a], self.hand_keypoints[b]])
            self.lines[i].setData(pos=pts, color=(1, 1, 0, 1), width=2)

            if self.display_teacher and self.display_dist_lines:
                # Update dotted lines between current and reference keypoints
                for i in range(21):
                    a = self.hand_keypoints[i]
                    b = self.t_landmarks[i]
                    self.dotted_lines[i].setData(
                        pos=np.array([a, b]), color=(0, 1, 0, 1), width=1
                    )

    def closeEvent(self, event):
        self.cap.release()

    def keyPressEvent(self, event):
        if event.text().lower() == "q":
            fig = lm.get_landmark_plot(self.landmarks.reshape(21, 3), "Image frame")
            fig.show()

    @staticmethod
    def gradient_color(t):
        """Maps 0.0 (green) to 0.5 (yellow) to 1.0 (red)."""
        if t <= 0.5:
            # Green to Yellow
            r = 2 * t
            g = 1
            b = 0
        else:
            # Yellow to Red
            r = 1
            g = 1 - 2 * (t - 0.5)
            b = 0
        return (r, g, b, 1)


def process_frame(frame):
    # Convert to float32 first (to avoid weird behavior)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to tensor and permute to (C, H, W)
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)

    # return frame_tensor
    return crop_resize_new(img, 224).unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file."
    )
    args = parser.parse_args()

    hand_model = model_setup.get_hand_model(args.config)
    kmeans_model = model_setup.get_kmeans(
        "./saved_models/kmeans_fstretch_0000_aligned_landmarks.pkl"
    )

    t_path = f"./inference/landmark_predictions/teacher/fstretch/0000.json"
    init_exercise = "0000"

    with open(t_path, "r") as f:
        t_landmarks = {init_exercise: np.array(json.load(f))}

    app = QApplication(sys.argv)
    window = HandExerciseApp(
        hand_model, kmeans_model, t_landmarks, init_exercise, display_teacher=True
    )
    window.show()
    sys.exit(app.exec_())
