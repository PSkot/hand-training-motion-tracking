import numpy as np
import glob
import os
import json
import plotly.graph_objects as go
import src.helpers as helpers
from matplotlib import cm
import plotly.graph_objects as go
import numpy as np


def read_landmark_files(root):
    image_paths = glob.glob(os.path.join(root, "**/*.json"), recursive=True)
    image_paths = [path for path in image_paths if "combined.json" not in path]

    landmarks = []

    for path in image_paths:
        with open(path, "r") as f:
            data = json.load(f)
            landmarks.append(np.array(data))

    return np.stack(landmarks)


def save_landmarks(landmarks: np.array, path: str):
    with open(path, "w") as f:
        json.dump(landmarks.tolist(), f)


def combine_student_landmarks(root):
    exercises = os.listdir(root)
    for exercise in exercises:
        exercise_path = os.path.join(root, exercise)
        stages = os.listdir(exercise_path)

        for stage in stages:
            stage_path = os.path.join(exercise_path, stage)
            landmarks = read_landmark_files(stage_path)
            save_landmarks(landmarks, os.path.join(stage_path, "combined.json"))


def align_landmarks_stable(source):
    """
    Align 21x3 hand landmarks to a stable, local hand-centric coordinate frame.

    The frame:
    - Origin: wrist (joint 0)
    - X-axis: wrist → index base (joint 5)
    - Z-axis: normal of palm plane (wrist, index base, pinky base)
    - Y-axis: completes right-handed frame

    Args:
        source (np.ndarray): (21, 3) array of 3D hand joints in camera space

    Returns:
        aligned (np.ndarray): (21, 3) landmarks in stable local frame
        R (np.ndarray): (3, 3) rotation matrix from world to local frame
    """
    source = np.asarray(source, dtype=np.float32)

    # Define key joints
    wrist = source[0]
    index_base = source[5]
    pinky_base = source[17]

    # Define X-axis: wrist → index base
    x_axis = index_base - wrist
    x_axis /= np.linalg.norm(x_axis)

    # Define Z-axis: normal to palm (via wrist → index, wrist → pinky)
    v1 = index_base - wrist
    v2 = pinky_base - wrist
    z_axis = np.cross(v1, v2)
    z_axis /= np.linalg.norm(z_axis)

    # Define Y-axis: right-handed system
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Construct rotation matrix (world → local)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3, 3)

    # Translate to wrist, then rotate into local frame
    centered = source - wrist
    aligned = (R.T @ centered.T).T  # rotate each point

    return aligned, R


def get_landmark_plot(hand_keypoints, title, reference_hand_keypoints=None):
    connections = helpers.read_hand_keypoints()

    # Scatter plot for the primary hand
    scatter_main = go.Scatter3d(
        x=hand_keypoints[:, 0],
        y=hand_keypoints[:, 1],
        z=hand_keypoints[:, 2],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="Patient Hand",
    )

    # Lines (bones) for the primary hand
    lines_main = [
        go.Scatter3d(
            x=[hand_keypoints[i, 0], hand_keypoints[j, 0]],
            y=[hand_keypoints[i, 1], hand_keypoints[j, 1]],
            z=[hand_keypoints[i, 2], hand_keypoints[j, 2]],
            mode="lines",
            line=dict(color="navy", width=2),
            showlegend=False,
        )
        for i, j in connections
    ]

    data = [scatter_main] + lines_main

    if reference_hand_keypoints is not None:
        # Reference hand plot
        scatter_ref = go.Scatter3d(
            x=reference_hand_keypoints[:, 0],
            y=reference_hand_keypoints[:, 1],
            z=reference_hand_keypoints[:, 2],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Teacher Hand",
        )

        lines_ref = [
            go.Scatter3d(
                x=[reference_hand_keypoints[i, 0], reference_hand_keypoints[j, 0]],
                y=[reference_hand_keypoints[i, 1], reference_hand_keypoints[j, 1]],
                z=[reference_hand_keypoints[i, 2], reference_hand_keypoints[j, 2]],
                mode="lines",
                line=dict(color="#d62728", width=2),
                showlegend=False,
            )
            for i, j in connections
        ]

        # Add dotted lines between corresponding keypoints
        dotted_lines = [
            go.Scatter3d(
                x=[hand_keypoints[i, 0], reference_hand_keypoints[i, 0]],
                y=[hand_keypoints[i, 1], reference_hand_keypoints[i, 1]],
                z=[hand_keypoints[i, 2], reference_hand_keypoints[i, 2]],
                mode="lines",
                line=dict(color="green", width=2, dash="dot"),
                showlegend=(i == 0),  # Show legend only for the first line
                name="Keypoint Displacements" if i == 0 else None,
            )
            for i in range(min(len(hand_keypoints), len(reference_hand_keypoints)))
        ]

        data += [scatter_ref] + lines_ref + dotted_lines

    fig = go.Figure(data=data)

    fig.update_layout(
        title=f"3D Hand Skeleton, {title}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            # xaxis=dict(range=[2, 0]),
            # yaxis=dict(range=[1, -1]),
            # zaxis=dict(range=[0.8, -0.1]),
        ),
        scene_camera=dict(eye=dict(x=0, y=0, z=-2)),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig
