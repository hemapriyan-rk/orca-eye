"""
ORCA EYE — Stage A: Physical Camera Calibration & Egocentric Coordinates
=======================================================================
Establishes physical camera geometry, intrinsics matrix, distortion parameters,
measured horizontal FOV, and coordinate transformations between the optical
camera frame and the wearer's egocentric ground-plane frame.

Coordinate Conventions:
  Camera Optical Frame (OpenCV convention):
    X_c: right
    Y_c: down
    Z_c: forward (optical axis)

  Wearer Egocentric Frame:
    x: lateral (right positive)
    y: forward along ground plane (ahead positive)
    z: vertical (up positive)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraCalibration:
    """
    Physical camera intrinsics and geometric projection model.
    """
    frame_width: int
    frame_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    hfov_deg: float = field(init=False)
    vfov_deg: float = field(init=False)
    half_hfov_rad: float = field(init=False)
    half_vfov_rad: float = field(init=False)

    def __post_init__(self) -> None:
        # Calibrated half-angle horizontal and vertical FOV
        self.half_hfov_rad = math.atan2(self.frame_width / 2.0, self.fx)
        self.half_vfov_rad = math.atan2(self.frame_height / 2.0, self.fy)
        self.hfov_deg = math.degrees(2.0 * self.half_hfov_rad)
        self.vfov_deg = math.degrees(2.0 * self.half_vfov_rad)

    @classmethod
    def from_fov(
        cls,
        frame_width: int,
        frame_height: int,
        hfov_deg: float = 65.0,
        vfov_deg: Optional[float] = None,
    ) -> "CameraCalibration":
        """
        Construct calibration from explicit horizontal (and optional vertical) FOV.
        Focal lengths are derived from pinhole geometry:
          fx = (W / 2) / tan(hfov / 2)
          fy = (H / 2) / tan(vfov / 2)
        """
        half_h_rad = math.radians(hfov_deg / 2.0)
        fx = (frame_width / 2.0) / math.tan(half_h_rad)

        if vfov_deg is not None:
            half_v_rad = math.radians(vfov_deg / 2.0)
            fy = (frame_height / 2.0) / math.tan(half_v_rad)
        else:
            fy = fx  # square pixel assumption

        cx = frame_width / 2.0
        cy = frame_height / 2.0

        return cls(
            frame_width=frame_width,
            frame_height=frame_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], default_w: int = 640, default_h: int = 480) -> "CameraCalibration":
        """Factory from configuration dictionary."""
        w = int(d.get("frame_width", default_w))
        h = int(d.get("frame_height", default_h))
        hfov = float(d.get("hfov_deg", 65.0))
        vfov = d.get("vfov_deg", None)
        if vfov is not None:
            vfov = float(vfov)

        if "fx" in d and "fy" in d:
            fx = float(d["fx"])
            fy = float(d["fy"])
            cx = float(d.get("cx", w / 2.0))
            cy = float(d.get("cy", h / 2.0))
            k1 = float(d.get("k1", 0.0))
            k2 = float(d.get("k2", 0.0))
            p1 = float(d.get("p1", 0.0))
            p2 = float(d.get("p2", 0.0))
            return cls(w, h, fx, fy, cx, cy, k1, k2, p1, p2)

        return cls.from_fov(w, h, hfov_deg=hfov, vfov_deg=vfov)

    def pixel_to_ray(self, u: float, v: float) -> Tuple[float, float, float]:
        """
        Compute unit bearing ray [rx, ry, rz] in optical camera coordinates.
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        norm = math.sqrt(x * x + y * y + 1.0)
        return (x / norm, y / norm, 1.0 / norm)

    def pixel_and_depth_to_camera_point(self, u: float, v: float, depth_m: float) -> Tuple[float, float, float]:
        """
        Project pixel (u, v) and metric depth Z to 3D point in camera optical frame.
        """
        x_c = ((u - self.cx) / self.fx) * depth_m
        y_c = ((v - self.cy) / self.fy) * depth_m
        z_c = depth_m
        return (x_c, y_c, z_c)

    def camera_to_wearer_ego(self, x_c: float, y_c: float, z_c: float) -> Tuple[float, float, float]:
        """
        Transform from camera optical frame to wearer egocentric frame:
          x (lateral right) = X_c
          y (forward ground) = Z_c
          z (vertical up)    = -Y_c
        """
        return (x_c, z_c, -y_c)

    def is_in_horizontal_fov(self, bearing_rad: float) -> bool:
        """Evaluate if horizontal bearing is within calibrated FOV."""
        return abs(bearing_rad) <= self.half_hfov_rad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fx": round(self.fx, 2),
            "fy": round(self.fy, 2),
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
            "hfov_deg": round(self.hfov_deg, 2),
            "vfov_deg": round(self.vfov_deg, 2),
            "half_hfov_rad": round(self.half_hfov_rad, 4),
        }
