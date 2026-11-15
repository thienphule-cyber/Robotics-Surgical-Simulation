"""Control package for the UR3e vision-guided surgical simulation."""

from .transformations import (
    make_pose,
    to_se3,
    pose_error,
    adjoint,
    rotate_twist,
    transform_points,
    project_points,
    skew,
)
from .kinematics import (
    build_ur3e,
    forward_kinematics,
    inverse_kinematics,
    within_joint_limits,
    clip_to_joint_limits,
    UR3E_HOME_Q,
    UR3E_QLIM,
)
from .jacobian import (
    jacobian_base,
    jacobian_tool,
    jacobian_camera,
    manipulability,
    min_singular_value,
    is_near_singularity,
    damped_pinv,
    cartesian_to_joint_velocity,
    limit_joint_velocity,
)
from .trajectory import (
    JointTrajectory,
    CartesianTrajectory,
    joint_trajectory,
    cartesian_trajectory,
    cartesian_to_joint_trajectory,
    scalar_profile,
)
from .safety import (
    SafetyMonitor,
    SafetyConfig,
    SafetyReport,
    SafetyState,
    Box,
    Sphere,
)
from .motion_controller import MotionController
from .camera import CameraIntrinsics, PinholeCamera, look_at_pose
from .feature_detection import (
    Detection,
    ColorMarkerDetector,
    BlobDetector,
    draw_overlay,
    marker_bgr,
)
from .feature_tracking import FeatureTracker, Track
from .ibvs import (
    IBVSController,
    IBVSResult,
    interaction_matrix,
    desired_from_pose,
    run_ibvs,
)

__all__ = [
    "make_pose", "to_se3", "pose_error", "adjoint", "rotate_twist",
    "transform_points", "project_points", "skew",
    "build_ur3e", "forward_kinematics", "inverse_kinematics",
    "within_joint_limits", "clip_to_joint_limits", "UR3E_HOME_Q", "UR3E_QLIM",
    "jacobian_base", "jacobian_tool", "jacobian_camera", "manipulability",
    "min_singular_value", "is_near_singularity", "damped_pinv",
    "cartesian_to_joint_velocity", "limit_joint_velocity",
]