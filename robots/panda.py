"""
Panda robot metadata for LIBERO.

Defines obs key names, state assembly, and action space.
State: eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2) = 8-dim.
"""

import numpy as np

# LIBERO obs keys for Panda
EEF_POS_KEY = "robot0_eef_pos"             # shape (3,)
EEF_QUAT_KEY = "robot0_eef_quat"           # shape (4,) in xyzw order
GRIPPER_QPOS_KEY = "robot0_gripper_qpos"   # shape (2,)

# Camera obs keys in LIBERO → pi0.5 image names
CAMERA_KEYS = {
    "agentview_image": "image",             # 3rd-person view
    "robot0_eye_in_hand_image": "wrist_image",   # wrist camera
}

ACTION_DIM = 7   # 6-DoF delta EEF + gripper
STATE_DIM = 8    # eef pos(3) + eef axis-angle(3) + gripper qpos(2)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Match LeRobot's LIBERO quaternion conversion exactly."""
    quat = np.asarray(quat, dtype=np.float32)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return axis * angle


def assemble_state(obs: dict) -> np.ndarray:
    """Build 8-dim state vector from LIBERO obs dict."""
    eef_pos = obs[EEF_POS_KEY]                         # (3,)
    eef_quat = obs[EEF_QUAT_KEY]                       # (4,)
    eef_axis_angle = quat_to_axis_angle(eef_quat)       # (3,)
    gripper = obs[GRIPPER_QPOS_KEY]                    # (2,)
    return np.concatenate([eef_pos, eef_axis_angle, gripper])  # (8,)
