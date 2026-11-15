"""workcell.py - 3D surgical workcell for the UR3e simulation (Swift).

World frame = robot base frame. Units: metres. z is up, +x points from the
robot towards the surgical table, the patient lies along the y axis
(head towards +y).

Contents
--------
* UR3e (drawn as link cylinders, or with Swift's own geometry)
* Surgical table, robot pedestal, floor
* Patient mannequin made of separate body parts, with surgical regions
* Surgical instrument attached to the end-effector
* Eye-in-hand camera and external camera
* Safety barriers, warning sign and restricted zones
* Colour-coded markers on the surgical site (features for IBVS)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np
import spatialgeometry as sg
import swift
from spatialmath import SE3

from control import (
    Box,
    CameraIntrinsics,
    PinholeCamera,
    SafetyConfig,
    build_ur3e,
    inverse_kinematics,
    look_at_pose,
    marker_bgr,
    project_points,
    skew,
)

# Marker layout (units of marker_half_size) around the surgical site.
_MARKER_OFFSETS = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0), (0, 1.5)]


@dataclass
class WorkcellConfig:
    # --- floor and robot pedestal ---
    floor_z: float = -0.80
    pedestal_size: tuple = (0.30, 0.30, 0.80)

    # --- surgical table (top surface at table_top_z) ---
    table_top_z: float = -0.05
    table_x_range: tuple = (0.18, 0.78)
    table_length: float = 1.90
    table_thickness: float = 0.06

    # --- mannequin (centred on the table width) ---
    torso_size: tuple = (0.38, 0.55, 0.20)
    head_radius: float = 0.10
    leg_size: tuple = (0.15, 0.70, 0.12)
    arm_size: tuple = (0.09, 0.55, 0.08)

    # --- surgical regions: name -> (x, y) on the torso surface ---
    regions: dict = field(default_factory=lambda: {
        "abdomen": (0.40, -0.02),
        "chest": (0.40, 0.12),
    })
    default_region: str = "abdomen"

    # --- instrument and approach ---
    instrument_length: float = 0.12
    instrument_radius: float = 0.006
    tip_standoff: float = 0.04       # tip height above the surface at the goal
    tool_yaw_deg: float = 0.0        # yaw of the tool about the vertical axis

    # --- cameras ---
    image_size: tuple = (640, 480)
    eye_camera_fov_deg: float = 80.0
    eye_camera_offset: tuple = (0.09, 0.0, -0.02)   # was (0.08, 0.0, 0.04)
    eye_camera_tilt_deg: float = -27.0              # was -33.0
    external_camera_fov_deg: float = 70.0
    external_camera_eye: tuple = (1.10, -0.90, 0.70)
    external_camera_target: tuple = (0.30, 0.0, 0.0)

    # --- IBVS markers ---
    n_markers: int = 4
    marker_half_size: float = 0.03

    # --- safety layout ---
    workspace_x: tuple = (-0.45, 0.75)
    workspace_y: tuple = (-0.55, 0.55)
    workspace_z_max: float = 0.80
    head_zone_margin: float = 0.08
    barrier_height: float = 1.20


class Workcell:
    def __init__(self, config: WorkcellConfig | None = None, robot=None,
                headless: bool = False, realtime: bool = True, browser=None,
                mesh_robot: bool = True):
        """
        headless : do not start Swift (useful for tests; the external camera
            rendering still works).
        """
        self.cfg = config or WorkcellConfig()
        if self.cfg.n_markers > len(_MARKER_OFFSETS):
            raise ValueError(f"n_markers must be <= {len(_MARKER_OFFSETS)}")

        self.robot = robot if robot is not None else build_ur3e()
        self.env = None
        self._wire_boxes = []       # (center, size, bgr) for external rendering
        self._wire_spheres = []     # (center, radius, bgr)
        self._link_shapes = []
        self.body_parts = {}
        self.restricted_zones = []
        self.region = self.cfg.default_region

        if not headless:
            self.env = swift.Swift()
            self.env.launch(realtime=realtime, browser=browser)
            self._set_viewer_camera()

        self._build_environment()
        self._build_cameras()
        self._build_markers()
        self._build_robot_visual(mesh_robot)
        self.update()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rgba(bgr, alpha=1.0):
        b, g, r = bgr
        return (r / 255.0, g / 255.0, b / 255.0, alpha)

    def _add(self, shape):
        if self.env is not None:
            self.env.add(shape)
        return shape

    def _add_box(self, name, center, size, bgr, alpha=1.0, wire=True):
        shape = sg.Cuboid(scale=list(size), color=self._rgba(bgr, alpha))
        shape.T = SE3.Trans(*center).A
        self._add(shape)
        if wire:
            self._wire_boxes.append((np.asarray(center, float), np.asarray(size, float), bgr))
        self.body_parts[name] = {"center": np.asarray(center, float),
                                "size": np.asarray(size, float)}
        return shape

    def _set_viewer_camera(self):
        try:
            self.env.set_camera_pose([1.4, -1.2, 0.8], [0.30, 0.0, -0.1])
        except Exception:
            pass    # older Swift versions: keep the default viewpoint

    # ------------------------------------------------------------------
    # Environment: floor, pedestal, table, mannequin, barriers, zones
    # ------------------------------------------------------------------
    def _build_environment(self):
        c = self.cfg
        grey, wood = (90, 90, 90), (120, 130, 140)
        skin, cloth = (150, 180, 220), (170, 200, 150)

        # Floor
        self._add_box("floor", (0.3, 0.0, c.floor_z - 0.005), (3.0, 3.0, 0.01),
                    (60, 60, 60), wire=False)

        # Robot pedestal (top at z = 0, the robot base plane)
        ps = c.pedestal_size
        self._add_box("pedestal", (0.0, 0.0, -ps[2] / 2), ps, grey)

        # Surgical table
        x0, x1 = c.table_x_range
        tx, tw = (x0 + x1) / 2, x1 - x0
        top_c = c.table_top_z - c.table_thickness / 2
        self._add_box("table_top", (tx, 0.0, top_c), (tw, c.table_length, c.table_thickness), wood)
        col_top = c.table_top_z - c.table_thickness
        col_h = col_top - c.floor_z
        self._add_box("table_column", (tx, 0.0, c.floor_z + col_h / 2), (0.25, 0.25, col_h), grey)

        # Mannequin parts
        tz = c.table_top_z
        tw_, tl, th = c.torso_size
        self.torso_top_z = tz + th
        self._add_box("torso", (tx, 0.0, tz + th / 2), c.torso_size, cloth)

        head_c = np.array([tx, tl / 2 + 0.03 + c.head_radius, tz + c.head_radius])
        head = sg.Sphere(radius=c.head_radius, color=self._rgba(skin))
        head.T = SE3.Trans(*head_c).A
        self._add(head)
        self._wire_spheres.append((head_c, c.head_radius, skin))
        self.body_parts["head"] = {"center": head_c, "size": np.full(3, 2 * c.head_radius)}

        for side, sign in (("left", -1), ("right", 1)):
            ax = tx + sign * (tw_ / 2 + 0.02 + c.arm_size[0] / 2)
            self._add_box(f"{side}_arm", (ax, 0.0, tz + c.arm_size[2] / 2), c.arm_size, skin)
            lx = tx + sign * (c.leg_size[0] / 2 + 0.01)
            ly = -tl / 2 - 0.02 - c.leg_size[1] / 2
            self._add_box(f"{side}_leg", (lx, ly, tz + c.leg_size[2] / 2), c.leg_size, cloth)

        # Safety barriers (translucent) and warning sign
        h = c.barrier_height
        zc = c.floor_z + h / 2
        yellow = (0, 215, 255)
        self._add_box("barrier_back", (-0.50, 0.0, zc), (0.02, 1.40, h), yellow, 0.25)
        self._add_box("barrier_left", (0.25, -0.70, zc), (1.50, 0.02, h), yellow, 0.25)
        self._add_box("barrier_right", (0.25, 0.70, zc), (1.50, 0.02, h), yellow, 0.25)
        self._add_box("warning_sign", (-0.485, 0.0, 0.0), (0.01, 0.30, 0.20), (0, 0, 220), wire=False)

        # Restricted zone around the head (the robot must not enter it)
        m = c.head_zone_margin
        r = c.head_radius + m
        z_min = c.table_top_z
        z_max = head_c[2] + r
        zone = Box("head", head_c - np.array([r, r, 0.0]) + np.array([0, 0, z_min - head_c[2]]),
                head_c + np.array([r, r, 0.0]) + np.array([0, 0, z_max - head_c[2]]))
        self.restricted_zones.append(zone)
        center = (zone.min_corner + zone.max_corner) / 2
        size = zone.max_corner - zone.min_corner
        self._add_box("zone_head", center, size, (0, 0, 255), 0.25, wire=False)
        self._wire_boxes.append((center, size, (0, 0, 255)))

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    def _build_cameras(self):
        c = self.cfg
        w, h = c.image_size

        # Eye-in-hand camera: fixed hand-eye transform in the flange frame.
        T_mount = SE3.Trans(*c.eye_camera_offset) * SE3.Ry(np.deg2rad(c.eye_camera_tilt_deg))
        self.eye_camera = PinholeCamera(
            CameraIntrinsics.from_fov(w, h, c.eye_camera_fov_deg),
            T_mount_cam=T_mount, name="eye-in-hand", near=0.03, far=1.5)

        # External camera: fixed in the world.
        self.external_camera = PinholeCamera(
            CameraIntrinsics.from_fov(w, h, c.external_camera_fov_deg),
            T_mount_cam=look_at_pose(c.external_camera_eye, c.external_camera_target),
            name="external", near=0.05, far=6.0)

        # Visual bodies of the cameras
        self._eye_body = self._add(sg.Cuboid(scale=[0.03, 0.02, 0.02],
                                            color=self._rgba((40, 40, 40))))
        # Bracket: one bar from the wrist rim (not from the instrument) to the camera
        ox, oy, oz = c.eye_camera_offset
        reach = float(np.hypot(ox, oy))
        self._eye_bracket = []          # (shape, start, end) in the flange frame
        if reach > 0.03:
            u = np.array([ox, oy]) / reach
            p0 = np.array([u[0] * 0.025, u[1] * 0.025, oz])
            p1 = np.array([ox, oy, oz])
            shape = self._add(sg.Cylinder(radius=0.004, length=float(np.linalg.norm(p1 - p0)),
                                        color=self._rgba((60, 60, 60))))
            self._eye_bracket.append((shape, p0, p1))
        ext_body = self._add(sg.Cuboid(scale=[0.08, 0.05, 0.05],
                                    color=self._rgba((40, 40, 40))))
        ext_body.T = self.external_camera.world_pose().A
        # Stand for the external camera: base plate, pole and mounting head
        ex, ey, ez = c.external_camera_eye
        pole_h = (ez - 0.04) - c.floor_z
        self._add_box("camera_stand_base", (ex, ey, c.floor_z + 0.01),
                    (0.34, 0.34, 0.02), (70, 70, 70), wire=False)
        self._add_box("camera_stand_pole", (ex, ey, c.floor_z + pole_h / 2),
                    (0.04, 0.04, pole_h), (110, 110, 110), wire=False)
        self._add_box("camera_stand_head", (ex, ey, ez - 0.03),
                    (0.07, 0.07, 0.02), (70, 70, 70), wire=False)

    # ------------------------------------------------------------------
    # Markers on the surgical site (IBVS features)
    # ------------------------------------------------------------------
    def _build_markers(self):
        self._marker_shapes = []
        for i in range(self.cfg.n_markers):
            s = sg.Sphere(radius=self.eye_camera.marker_radius,
                        color=self._rgba(marker_bgr(i)))
            self._marker_shapes.append(self._add(s))
        self.select_region(self.region)

    @property
    def site(self) -> np.ndarray:
        """Centre of the selected surgical region on the mannequin surface."""
        x, y = self.cfg.regions[self.region]
        return np.array([x, y, self.torso_top_z])

    @property
    def marker_points(self) -> np.ndarray:
        """(n_markers, 3) world positions of the markers around the site."""
        half = self.cfg.marker_half_size
        pts = [self.site + np.array([dx * half, dy * half, 0.001])
            for dx, dy in _MARKER_OFFSETS[:self.cfg.n_markers]]
        return np.array(pts)

    def select_region(self, name: str):
        """Choose the surgical region; markers move to that region."""
        if name not in self.cfg.regions:
            raise KeyError(f"Unknown region '{name}'. Available: {list(self.cfg.regions)}")
        self.region = name
        for shape, p in zip(self._marker_shapes, self.marker_points):
            shape.T = SE3.Trans(*p).A

    # ------------------------------------------------------------------
    # Robot, instrument, poses
    # ------------------------------------------------------------------
    def _build_robot_visual(self, mesh_robot: bool):
        c = self.cfg
        self._visual_robot = None

        if mesh_robot and self.env is not None:
            try:
                from ir_support.robots import UR3e      # UR3e with 3D meshes
                self._visual_robot = UR3e()
                self._visual_robot.add_to_env(self.env)
            except Exception as exc:
                print(f"[workcell] UR3e meshes not available ({exc}); using link cylinders")
                self._visual_robot = None

        if self._visual_robot is None:                  # fallback: simple cylinders
            pts = self._joint_points(self.robot.q)
            for a, b in zip(pts[:-1], pts[1:]):
                shape = sg.Cylinder(radius=0.03, length=float(np.linalg.norm(b - a)),
                                    color=self._rgba((200, 120, 40)))
                self._link_shapes.append(self._add(shape))

        self._instrument = self._add(sg.Cylinder(
            radius=c.instrument_radius, length=c.instrument_length,
            color=self._rgba((220, 220, 220))))
        self._tip = self._add(sg.Sphere(radius=c.instrument_radius * 1.4,
                                        color=self._rgba((0, 0, 255))))

    def _joint_points(self, q) -> np.ndarray:
        """Base point followed by the origin of each joint frame."""
        pts = [np.asarray(self.robot.base.t, dtype=float)]
        for T in self.robot.fkine_all(np.asarray(q, dtype=float)):
            p = np.asarray(T.t, dtype=float)
            if np.linalg.norm(p - pts[-1]) > 1e-9:
                pts.append(p)
        return np.array(pts)

    @staticmethod
    def _segment_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """4x4 pose of a z-axis cylinder spanning the segment a -> b."""
        d = b - a
        length = np.linalg.norm(d)
        T = np.eye(4)
        T[:3, 3] = (a + b) / 2.0
        if length < 1e-9:
            return T
        z = d / length
        ref = np.array([0.0, 0.0, 1.0])
        v = np.cross(ref, z)
        cos = float(ref @ z)
        if cos < -1.0 + 1e-6:                  # (almost) straight down: flip about x
            R = np.diag([1.0, -1.0, -1.0])
        elif np.linalg.norm(v) < 1e-9:         # straight up
            R = np.eye(3)
        else:
            vx = skew(v)
            R = np.eye(3) + vx + vx @ vx / (1.0 + cos)
        T[:3, :3] = R
        return T

    def desired_flange_pose(self) -> SE3:
        """Flange pose that puts the instrument tip just above the site,
        pointing straight down."""
        c = self.cfg
        R = SE3.Rz(np.deg2rad(c.tool_yaw_deg)).R @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.site + np.array([0.0, 0.0, c.tip_standoff + c.instrument_length])
        return SE3(T, check=False)

    def start_flange_pose(self) -> SE3:
        """Starting pose: a bit higher, closer to the robot and yawed relative to the goal."""
        goal = self.desired_flange_pose()
        T_start = np.eye(4)
        T_start[:3, :3] = SE3.Rz(np.deg2rad(10.0)).R @ goal.R
        T_start[:3, 3] = goal.t + np.array([-0.06, 0.03, 0.06])   # was [-0.06, 0.05, 0.12]
        return SE3(T_start, check=False)

    def go_to_start(self) -> bool:
        """Teleport the robot to the start pose (IK). Returns False if unreachable.

        Call this before enabling safety: the monitor stops the robot if its
        current pose is already outside the allowed workspace.
        """
        q, ok = inverse_kinematics(self.robot, self.start_flange_pose(), q0=self.robot.q)
        if ok:
            self.robot.q = q
            self.step(0.01)
        return ok

    # ------------------------------------------------------------------
    # Safety configuration matching this workcell
    # ------------------------------------------------------------------
    def make_safety_config(self, **overrides) -> SafetyConfig:
        c = self.cfg
        workspace = Box(
            "workcell",
            [c.workspace_x[0], c.workspace_y[0], self.torso_top_z - 0.01],
            [c.workspace_x[1], c.workspace_y[1], c.workspace_z_max])
        kwargs = dict(
            workspace=workspace,
            restricted_zones=list(self.restricted_zones),
            floor_z=c.table_top_z,
            tool_length=c.instrument_length,
        )
        kwargs.update(overrides)
        return SafetyConfig(**kwargs)

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------
    def update(self, q=None):
        """Move all dynamic shapes to match the robot configuration."""
        q = np.asarray(self.robot.q if q is None else q, dtype=float)
        c = self.cfg
        if self._visual_robot is not None:
            self._visual_robot.q = q

        if self._link_shapes:
            pts = self._joint_points(q)
            for shape, a, b in zip(self._link_shapes, pts[:-1], pts[1:]):
                shape.T = self._segment_pose(a, b)

        T_flange = self.robot.fkine(q)
        self._instrument.T = (T_flange * SE3.Tz(c.instrument_length / 2)).A
        self._tip.T = (T_flange * SE3.Tz(c.instrument_length)).A
        self._eye_body.T = (T_flange * self.eye_camera.T_mount_cam).A
        for shape, p0, p1 in self._eye_bracket:
            a = (T_flange * SE3.Trans(*p0)).t
            b = (T_flange * SE3.Trans(*p1)).t
            shape.T = self._segment_pose(np.asarray(a, float), np.asarray(b, float))

    def step(self, dt: float = 0.02):
        """Update the scene and advance Swift. Use as the controller callback:
        MotionController(..., on_step=lambda c: workcell.step(c.dt))"""
        self.update()
        if self.env is not None:
            self.env.step(dt)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ------------------------------------------------------------------
    # External camera rendering (wireframe of the whole workcell)
    # ------------------------------------------------------------------
    @staticmethod
    def _px(p) -> int:
        return int(np.clip(round(float(p)), -20000, 20000))

    def _draw_line(self, img, camera, T_cam_world, a, b, color, thickness=1):
        pix, depth = project_points(camera.K, T_cam_world, np.array([a, b]))
        if np.all(depth > camera.near) and np.all(np.isfinite(pix)):
            cv2.line(img, (self._px(pix[0, 0]), self._px(pix[0, 1])),
                    (self._px(pix[1, 0]), self._px(pix[1, 1])),
                    color, thickness, cv2.LINE_AA)

    def render_external(self, camera: PinholeCamera | None = None, q=None) -> np.ndarray:
        """BGR image of the whole workcell as seen by the external camera.

        Lines that cross the camera near plane are skipped (no clipping).
        """
        cam = camera or self.external_camera
        q = np.asarray(self.robot.q if q is None else q, dtype=float)
        T_world_cam = cam.world_pose()
        T_cam_world = T_world_cam.inv()
        w, h = cam.intrinsics.width, cam.intrinsics.height
        img = np.full((h, w, 3), 25, dtype=np.uint8)

        # Static boxes (table, mannequin, barriers, restricted zones...)
        signs = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                if bin(i ^ j).count("1") == 1]
        for center, size, color in self._wire_boxes:
            corners = center + signs * size / 2.0
            for i, j in edges:
                self._draw_line(img, cam, T_cam_world, corners[i], corners[j], color, 1)

        # Head sphere
        for center, radius, color in self._wire_spheres:
            pix, depth = project_points(cam.K, T_cam_world, center[None, :])
            if depth[0] > cam.near and np.all(np.isfinite(pix)):
                r = max(2, int(cam.intrinsics.fx * radius / depth[0]))
                cv2.circle(img, (self._px(pix[0, 0]), self._px(pix[0, 1])), r,
                        color, 1, cv2.LINE_AA)

        # Robot links and instrument
        pts = self._joint_points(q)
        for a, b in zip(pts[:-1], pts[1:]):
            self._draw_line(img, cam, T_cam_world, a, b, (60, 140, 255), 4)
        T_flange = self.robot.fkine(q)
        tip = np.asarray(T_flange.t) + T_flange.R[:, 2] * self.cfg.instrument_length
        self._draw_line(img, cam, T_cam_world, np.asarray(T_flange.t), tip, (235, 235, 235), 2)

        # Markers
        pixels, depths, visible = cam.project(T_world_cam, self.marker_points)
        for i in range(len(pixels)):
            if visible[i]:
                r = max(2, int(cam.intrinsics.fx * 0.006 / depths[i]))
                cv2.circle(img, (self._px(pixels[i, 0]), self._px(pixels[i, 1])), r,
                        marker_bgr(i), -1, cv2.LINE_AA)

        cv2.putText(img, f"{cam.name} camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return img


if __name__ == "__main__":
    wc = Workcell()
    print("start pose reachable:", wc.go_to_start())
    for k in range(300):
        wc.step(0.02)
        if k % 5 == 0:
            cv2.imshow("external camera", wc.render_external())
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    wc.close()