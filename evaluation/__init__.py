"""Evaluation package: metrics, reports and aggregation for recorded sessions."""

from .accuracy import (
    CONVERGENCE_MARGIN,
    Q_COLS,
    TCP_COLS,
    ibvs_metrics,
    kinematic_accuracy,
    path_metrics,
    pose_errors,
    tracking_availability,
    vision_accuracy,
)
from .performance import (
    motion_metrics,
    recovery_metrics,
    safety_metrics,
    timing_metrics,
)
from .reports import (
    THRESHOLDS,
    aggregate,
    clean,
    evaluate_session,
    flatten,
    plot_session,
    render_markdown,
    write_aggregate_report,
    write_session_report,
)

__all__ = [
    "CONVERGENCE_MARGIN", "Q_COLS", "TCP_COLS",
    "ibvs_metrics", "kinematic_accuracy", "path_metrics", "pose_errors",
    "tracking_availability", "vision_accuracy",
    "motion_metrics", "recovery_metrics", "safety_metrics", "timing_metrics",
    "THRESHOLDS", "aggregate", "clean", "evaluate_session", "flatten",
    "plot_session", "render_markdown", "write_aggregate_report",
    "write_session_report",
]