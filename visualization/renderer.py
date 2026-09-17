"""
ORCA EYE — Stage 9: Visualization Renderer
============================================
4-panel composite debug window:

  Panel 1 (top-left)   : Live camera feed + YOLO bboxes + Projected Safe Path Corridor Lines
  Panel 2 (top-right)  : Free-space colour overlay on camera
  Panel 3 (bottom-left): B&W camera background + 2D Nav Grid + Candidate & Selected Path Lines
  Panel 4 (bottom-right): Decision recommendation + Path Calculation Scoring & Metrics

All panels are the same resolution as the configured camera frame (640x480).
Disclaimer banner runs across the bottom.
Optional pyttsx3 TTS.
"""

import logging
import math
import time
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Renderer:
    """
    Multi-panel OpenCV visualization for ORCA EYE.

    Configuration keys (from cfg['visualization']):
      window_name, show_freespace, show_grid, show_paths,
      show_metrics, colors (dict), freespace_alpha, grid_alpha,
      font_scale, fps_display, tts_enabled
    """

    _CMD_COLORS = {
        "STRAIGHT":     (50, 220, 50),
        "SLIGHT_LEFT":  (50, 200, 130),
        "SLIGHT_RIGHT": (50, 200, 130),
        "LEFT":         (30, 140, 255),
        "RIGHT":        (30, 140, 255),
        "STOP":         (50, 50, 220),
        "CAUTION":      (30, 165, 255),
    }

    def __init__(self, cfg: dict, frame_w: int, frame_h: int) -> None:
        vis = cfg.get("visualization", {})
        raw_name = vis.get("window_name", "ORCA EYE - Research Prototype")
        self.window_name: str = raw_name.replace("—", "-").replace("\u2014", "-").replace("\\u2014", "-")
        self.layout_mode: str = vis.get("layout", "auto")
        self.show_freespace: bool = vis.get("show_freespace", True)
        self.show_grid: bool = vis.get("show_grid", True)
        self.show_paths: bool = vis.get("show_paths", True)
        self.show_metrics: bool = vis.get("show_metrics", True)
        self.tts_enabled: bool = vis.get("tts_enabled", False)

        colors = vis.get("colors", {})
        self.color_free     = tuple(colors.get("free",          [0, 200, 0]))
        self.color_obstacle = tuple(colors.get("obstacle",      [0, 0, 220]))
        self.color_unknown  = tuple(colors.get("unknown",       [0, 165, 255]))
        self.color_selected = tuple(colors.get("selected_path", [0, 255, 255]))
        self.color_cand     = tuple(colors.get("candidate_path",[120, 120, 120]))
        self.color_bbox     = tuple(colors.get("bbox",          [0, 255, 0]))
        self.color_track    = tuple(colors.get("track_id",      [255, 165, 0]))

        self.fs_alpha   : float = vis.get("freespace_alpha", 0.45)
        self.grid_alpha : float = vis.get("grid_alpha", 0.30)
        self.font_scale : float = vis.get("font_scale", 0.55)

        # Panel resolution
        self.panel_w = frame_w
        self.panel_h = frame_h

        # Screen dimension detection for auto-fitting and centering
        self.screen_w = 1920
        self.screen_h = 1080
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            self.screen_w = int(user32.GetSystemMetrics(0))
            self.screen_h = int(user32.GetSystemMetrics(1))
        except Exception:
            pass

        self._window_initialized = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # TTS
        self._tts_engine = None
        self._last_tts_cmd: str = ""
        if self.tts_enabled:
            self._init_tts()

        logger.info("Renderer initialized: %dx%d panels (screen: %dx%d, layout: %s)",
                    frame_w, frame_h, self.screen_w, self.screen_h, self.layout_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        detection_result,
        tracks: List,
        freespace_result,
        depth_result,
        spatial_map,
        candidates: List,
        decision,
        fps: float = 0.0,
        frame_id: int = 0,
        latency_ms: float = 0.0,
        spatial_entities: Optional[List] = None,
        dynamic_conflicts: Optional[List] = None,
        wall_proximity: Optional[dict] = None,
        geometry_3d_result: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Compose all four panels into a single BGR image.
        """
        p1 = self._panel_camera(frame, detection_result, tracks, spatial_map,
                                 candidates, decision, dynamic_conflicts,
                                 wall_proximity=wall_proximity, fps=fps, frame_id=frame_id)
        p2 = self._panel_freespace(frame, freespace_result)
        p3 = self._panel_navigation_bw(frame, spatial_map, candidates, decision,
                                       geometry_3d_result=geometry_3d_result)
        p4 = self._panel_decision(decision, fps, frame_id, latency_ms,
                                  detection_result, freespace_result, candidates,
                                  wall_proximity=wall_proximity,
                                  geometry_3d_result=geometry_3d_result)

        # Resize all panels to identical size before stacking
        p1 = self._ensure_size(p1)
        p2 = self._ensure_size(p2)
        p3 = self._ensure_size(p3)
        p4 = self._ensure_size(p4)

        # Adaptive layout:
        # If portrait (H > W, e.g. 360x640 phone video): arrange 1x4 horizontal to fit widescreen
        # If landscape (W >= H, e.g. 640x480 webcam): arrange 2x2 grid
        is_portrait = (self.panel_h > self.panel_w)
        use_horizontal = (self.layout_mode == "horizontal") or (self.layout_mode == "auto" and is_portrait)

        if use_horizontal:
            composite = np.hstack([p1, p2, p3, p4])
        else:
            top = np.hstack([p1, p2])
            bot = np.hstack([p3, p4])
            composite = np.vstack([top, bot])

        self._draw_disclaimer(composite)

        if self.tts_enabled and decision.command != self._last_tts_cmd:
            self._speak(decision.command)
            self._last_tts_cmd = decision.command

        return composite

    def show(self, composite: np.ndarray) -> int:
        if not self._window_initialized:
            comp_h, comp_w = composite.shape[:2]
            max_w = max(640, self.screen_w - 40)
            max_h = max(480, self.screen_h - 100) # reserve room for title bar and Windows taskbar

            scale = min(1.0, max_w / float(comp_w), max_h / float(comp_h))
            target_w = int(comp_w * scale)
            target_h = int(comp_h * scale)

            cv2.resizeWindow(self.window_name, target_w, target_h)
            pos_x = max(0, (self.screen_w - target_w) // 2)
            pos_y = max(0, (self.screen_h - target_h - 60) // 2)
            cv2.moveWindow(self.window_name, pos_x, pos_y)
            self._window_initialized = True

        cv2.imshow(self.window_name, composite)
        return cv2.waitKey(1)

    def destroy(self) -> None:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Panel 1 — Live Camera + Detections + AR Path Corridor Lines
    # ------------------------------------------------------------------

    def _panel_camera(
        self,
        frame: np.ndarray,
        detection_result,
        tracks: List,
        spatial_map=None,
        candidates: Optional[List] = None,
        decision=None,
        dynamic_conflicts: Optional[List] = None,
        wall_proximity: Optional[dict] = None,
        fps: float = 0.0,
        frame_id: int = 0,
    ) -> np.ndarray:
        panel = self._resize_panel(frame.copy())
        H, W = panel.shape[:2]
        oh, ow = frame.shape[:2]
        sx, sy = W / max(ow, 1), H / max(oh, 1)

        conflict_track_ids = set()
        if dynamic_conflicts:
            conflict_track_ids = {
                getattr(c, "track_id", -1) for c in dynamic_conflicts
                if getattr(c, "has_conflict", False)
            }

        # --- 1. Dynamic Automotive Reversing Camera Guidelines (on camera feed)
        if self.show_paths and candidates and decision:
            self._draw_reversing_camera_guidelines(panel, spatial_map, candidates, decision, wall_proximity)

        # --- 2. YOLO bounding boxes
        for obj in detection_result.objects:
            x1 = int(obj.bbox[0] * sx); y1 = int(obj.bbox[1] * sy)
            x2 = int(obj.bbox[2] * sx); y2 = int(obj.bbox[3] * sy)
            tid = getattr(obj, "track_id", None)
            is_conflict = tid in conflict_track_ids if tid is not None else False
            box_col = (0, 0, 255) if is_conflict else self.color_bbox

            # Filled semi-transparent box
            overlay = panel.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_col, -1)
            cv2.addWeighted(overlay, 0.15, panel, 0.85, 0, panel)
            # Solid border
            cv2.rectangle(panel, (x1, y1), (x2, y2), box_col, 2)

            # Ground contact point
            if hasattr(obj, "bottom_center"):
                bc_x = int(obj.bottom_center[0] * sx)
                bc_y = int(obj.bottom_center[1] * sy)
                cv2.circle(panel, (bc_x, bc_y), 4, (255, 200, 0), -1)

            # Label pill background
            label = f"{obj.class_name} {obj.confidence:.0%}"
            if tid is not None:
                label = f"#{tid} {label}"
            if is_conflict:
                label = f"⚠️ {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx1, ly1 = x1, max(0, y1 - th - 6)
            cv2.rectangle(panel, (lx1, ly1), (lx1 + tw + 6, ly1 + th + 6), box_col, -1)
            cv2.putText(panel, label, (lx1 + 3, ly1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # --- 3. Track velocity arrows & semantic motion
        for t in tracks:
            cx = int(t.center[0] * sx); cy = int(t.center[1] * sy)
            vx, vy = t.velocity
            tid = getattr(t, "track_id", -1)
            t_col = (0, 0, 255) if tid in conflict_track_ids else self.color_track

            if abs(vx) + abs(vy) > 1.0:
                ax = int(cx + vx * 6); ay = int(cy + vy * 6)
                cv2.arrowedLine(panel, (cx, cy), (ax, ay),
                                t_col, 2, tipLength=0.35)
            cv2.circle(panel, (cx, cy), 5, t_col, -1)
            cv2.circle(panel, (cx, cy), 5, (0, 0, 0), 1)

            motion = getattr(t, "motion_state", "")
            if motion and motion != "stationary":
                cv2.putText(panel, motion[:4].upper(), (cx + 6, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 255), 1)

        self._draw_panel_title(panel, "Live Camera")
        # Statistics label moved to bottom-left to give full prominence to top-right HUD
        self._bottom_left_label(panel, f"{detection_result.num_detections} obj  {len(tracks)} tracks")

        # --- 4. Prominent Top-Right Decision HUD
        if decision:
            self._draw_top_right_decision_hud(panel, decision, fps=fps, frame_id=frame_id, wall_proximity=wall_proximity)

        return panel

    def _draw_reversing_camera_guidelines(
        self,
        panel: np.ndarray,
        spatial_map,
        candidates: Optional[List],
        decision,
        wall_proximity: Optional[dict] = None,
    ) -> None:
        """
        Draw dynamic automotive reversing camera-style guidelines on the camera feed.
        Features:
          - Continuous curved guide rails responding to steering angle
          - 3-zone color grading: Red (<1.0m danger), Amber (1.0-2.5m caution), Neon Green (>2.5m safe)
          - Automotive transverse distance rungs with distance markers
          - Shaded perspective ground carpet
          - Obstacle / wall collision stop barrier bar
        """
        if not candidates or decision is None:
            return

        H, W = panel.shape[:2]

        # Selected candidate (or fallback to straight / first candidate)
        sel = next((c for c in candidates if c.direction == decision.selected_path_direction and not c.is_stop), None)
        if sel is None:
            sel = next((c for c in candidates if not c.is_stop), candidates[0])

        angle_deg = getattr(sel, "angle_deg", 0.0)
        theta = math.radians(angle_deg)

        # Generate parametric curve samples from camera bottom to horizon lookahead
        # Bottom of screen = user feet (s = 0.0); Top = lookahead horizon (s = 1.0)
        N = 36
        y_bottom = H - 2
        y_top = int(H * 0.44)
        total_dy = y_bottom - y_top

        left_pts = []
        right_pts = []
        center_pts = []
        s_vals = []

        is_stop = (decision.command == "STOP")
        is_wall_coll = bool(wall_proximity and wall_proximity.get("is_frontal_collision"))
        is_caution = (decision.command == "CAUTION")

        for i in range(N):
            s = i / float(N - 1)
            s_vals.append(s)
            y = int(y_bottom - s * total_dy)

            # Dynamic lateral deflection (steering curve like reverse camera)
            dx = math.tan(theta) * (s * total_dy * 0.90) + 0.35 * math.sin(theta) * (s ** 2) * W
            cx = W / 2.0 + dx

            # Perspective width: wide at feet (44% W), converging to 16% W at horizon
            half_w = (W * (0.44 - 0.28 * s)) / 2.0

            lx = max(0, min(W - 1, int(cx - half_w)))
            rx = max(0, min(W - 1, int(cx + half_w)))
            cx_int = max(0, min(W - 1, int(cx)))

            left_pts.append((lx, y))
            right_pts.append((rx, y))
            center_pts.append((cx_int, y))

        # 1. Subtle semi-transparent ground carpet
        carpet_poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        carpet_overlay = panel.copy()
        if is_stop or is_wall_coll:
            carpet_color = (0, 30, 220)    # Red tint
        elif is_caution:
            carpet_color = (0, 140, 220)   # Amber tint
        else:
            carpet_color = (0, 210, 130)   # Green tint
        cv2.fillPoly(carpet_overlay, [carpet_poly], carpet_color)
        cv2.addWeighted(carpet_overlay, 0.16, panel, 0.84, 0, panel)

        # 2. Draw curved guide rails segment by segment with 3-zone color grading
        for i in range(N - 1):
            s_mid = (s_vals[i] + s_vals[i + 1]) / 2.0
            if is_stop or is_wall_coll:
                seg_color = (0, 0, 240)    # STOP: All Red
            elif is_caution:
                seg_color = (0, 0, 240) if s_mid <= 0.22 else (0, 200, 255)
            else:
                if s_mid <= 0.22:
                    seg_color = (0, 0, 240)    # Zone 1 (Close/Danger): Red
                elif s_mid <= 0.52:
                    seg_color = (0, 215, 255)  # Zone 2 (Caution): Amber/Yellow
                else:
                    seg_color = (50, 240, 50)  # Zone 3 (Safe): Neon Green

            cv2.line(panel, left_pts[i], left_pts[i + 1], seg_color, 3, cv2.LINE_AA)
            cv2.line(panel, right_pts[i], right_pts[i + 1], seg_color, 3, cv2.LINE_AA)

        # 3. Transverse distance rungs (automotive parking lines)
        rung_indices = [
            (int(0.06 * (N - 1)), (0, 0, 240), "0.5m"),
            (int(0.22 * (N - 1)), (0, 180, 255), "1.0m"),
            (int(0.52 * (N - 1)), (0, 230, 255), "2.0m"),
            (int(0.80 * (N - 1)), (50, 240, 50), "3.0m"),
            (N - 1, (50, 240, 50), ""),
        ]
        for idx, r_col, label in rung_indices:
            idx = min(idx, N - 1)
            lp, rp = left_pts[idx], right_pts[idx]
            color = (0, 0, 240) if (is_stop or is_wall_coll) else r_col
            # Main crossbar
            cv2.line(panel, lp, rp, color, 2, cv2.LINE_AA)
            # Outer tick marks
            cv2.line(panel, (max(0, lp[0] - 8), lp[1]), lp, color, 2, cv2.LINE_AA)
            cv2.line(panel, rp, (min(W - 1, rp[0] + 8), rp[1]), color, 2, cv2.LINE_AA)
            if label:
                cv2.putText(panel, label, (max(2, lp[0] - 32), lp[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

        # 4. Center trajectory line with forward chevrons
        center_col = (0, 0, 240) if (is_stop or is_wall_coll) else (0, 255, 255)
        for i in range(N - 1):
            if i % 3 != 0:  # dashed center guide
                cv2.line(panel, center_pts[i], center_pts[i + 1], center_col, 2, cv2.LINE_AA)

        # Forward chevrons at 1/3 and 2/3 distance
        for k in (int(0.35 * (N - 1)), int(0.70 * (N - 1))):
            cx, cy = center_pts[k]
            chev_left = (cx - 7, cy + 6)
            chev_right = (cx + 7, cy + 6)
            cv2.line(panel, chev_left, (cx, cy), center_col, 2, cv2.LINE_AA)
            cv2.line(panel, chev_right, (cx, cy), center_col, 2, cv2.LINE_AA)

        # 5. Obstacle / Wall Collision Stop Barrier Bar
        clr = getattr(decision, "clearance", 1.0)
        has_barrier = is_stop or is_wall_coll or (clr < 0.80)
        if has_barrier:
            if is_stop or is_wall_coll:
                bar_s = min(0.25, max(0.08, clr * 0.4))
            else:
                bar_s = max(0.15, min(0.92, clr))
            bar_idx = min(N - 1, max(0, int(bar_s * (N - 1))))
            blp = left_pts[bar_idx]
            brp = right_pts[bar_idx]

            # Bold red barrier bar
            cv2.line(panel, blp, brp, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.line(panel, blp, brp, (255, 255, 255), 2, cv2.LINE_AA)

            # Warning text pill at the barrier center
            mid_x = (blp[0] + brp[0]) // 2
            mid_y = (blp[1] + brp[1]) // 2
            if is_wall_coll:
                bar_txt = "WALL HAZARD"
            elif is_stop:
                bar_txt = "STOP"
            else:
                bar_txt = f"OBSTACLE ({clr * 3.5:.1f}m)"

            (tw, th), _ = cv2.getTextSize(bar_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(panel, (mid_x - tw // 2 - 6, mid_y - th - 5),
                          (mid_x + tw // 2 + 6, mid_y + 4), (0, 0, 220), -1)
            cv2.rectangle(panel, (mid_x - tw // 2 - 6, mid_y - th - 5),
                          (mid_x + tw // 2 + 6, mid_y + 4), (255, 255, 255), 1)
            cv2.putText(panel, bar_txt, (mid_x - tw // 2, mid_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_top_right_decision_hud(
        self,
        panel: np.ndarray,
        decision,
        fps: float = 0.0,
        frame_id: int = 0,
        wall_proximity: Optional[dict] = None,
    ) -> None:
        """
        Render a high-contrast automotive navigation HUD badge in the top-right corner of Panel 1.
        """
        if decision is None:
            return

        H, W = panel.shape[:2]

        is_wall_coll = bool(wall_proximity and wall_proximity.get("is_frontal_collision"))
        lat_warn = (wall_proximity.get("lateral_warning") if wall_proximity else None) or getattr(decision, "wall_warning", None)
        has_wall_alert = is_wall_coll or bool(lat_warn)

        hud_w = 230
        hud_h = 92 if has_wall_alert else 76
        x2 = W - 10
        x1 = x2 - hud_w
        y1 = 10
        y2 = y1 + hud_h

        # Command color mapping
        cmd = decision.command
        if cmd == "STOP" or is_wall_coll:
            border_col = (0, 0, 240)    # Red
            txt_col    = (50, 50, 255)
            cmd_display = "STOP"
        elif cmd == "CAUTION":
            border_col = (0, 165, 255)  # Amber
            txt_col    = (0, 200, 255)
            cmd_display = "CAUTION"
        elif cmd == "STRAIGHT":
            border_col = (50, 220, 50)  # Green
            txt_col    = (70, 255, 70)
            cmd_display = "GO STRAIGHT"
        elif cmd in ("SLIGHT_LEFT", "LEFT"):
            border_col = (255, 180, 30)  # Cyan
            txt_col    = (255, 210, 50)
            cmd_display = "SLIGHT LEFT" if cmd == "SLIGHT_LEFT" else "TURN LEFT"
        elif cmd in ("SLIGHT_RIGHT", "RIGHT"):
            border_col = (255, 180, 30)  # Cyan
            txt_col    = (255, 210, 50)
            cmd_display = "SLIGHT RIGHT" if cmd == "SLIGHT_RIGHT" else "TURN RIGHT"
        else:
            border_col = (180, 180, 180)
            txt_col    = (220, 220, 220)
            cmd_display = cmd

        # Semi-transparent dark background
        overlay = panel.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)

        # High-contrast border
        cv2.rectangle(panel, (x1, y1), (x2, y2), border_col, 2)
        cv2.rectangle(panel, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 0, 0), 1)

        # Line 1: Header + FPS
        cv2.putText(panel, "NAVIGATION DECISION", (x1 + 8, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 190), 1, cv2.LINE_AA)
        fps_text = f"{fps:.0f} FPS"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(panel, fps_text, (x2 - fw - 8, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 220, 140), 1, cv2.LINE_AA)

        # Line 2: Large Bold Command Display
        cv2.putText(panel, cmd_display, (x1 + 8, y1 + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, txt_col, 2, cv2.LINE_AA)

        # Line 3: Clearance and Score metrics
        clr_pct = int(getattr(decision, "clearance", 0.0) * 100)
        score_val = getattr(decision, "score", 0.0)
        metric_txt = f"Clearance: {clr_pct}%  |  Score: {score_val:.2f}"
        cv2.putText(panel, metric_txt, (x1 + 8, y1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 220), 1, cv2.LINE_AA)

        # Line 4 (if wall alert):
        if is_wall_coll:
            alert_txt = "WALL COLLISION HAZARD!"
            cv2.putText(panel, alert_txt, (x1 + 8, y1 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 255), 1, cv2.LINE_AA)
        elif lat_warn:
            alert_txt = f"WALL PROXIMITY: {lat_warn}"
            cv2.putText(panel, alert_txt, (x1 + 8, y1 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)

    def _bottom_left_label(self, panel: np.ndarray, text: str) -> None:
        """Draw small statistics pill in bottom-left corner."""
        H, _ = panel.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(panel, (6, H - th - 10), (14 + tw, H - 4), (0, 0, 0), -1)
        cv2.putText(panel, text, (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Panel 2 — Free-Space Colour Overlay
    # ------------------------------------------------------------------

    def _panel_freespace(
        self,
        frame: np.ndarray,
        freespace_result,
    ) -> np.ndarray:
        panel = self._resize_panel(frame.copy())

        if freespace_result is not None and self.show_freespace:
            lm = cv2.resize(
                freespace_result.label_map,
                (self.panel_w, self.panel_h),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = np.zeros_like(panel)
            overlay[lm == 0] = self.color_free      # FREE   → green
            overlay[lm == 1] = self.color_obstacle  # OBS    → red
            overlay[lm == 2] = self.color_unknown   # UNKNOWN→ amber

            cv2.addWeighted(overlay, self.fs_alpha, panel, 1 - self.fs_alpha, 0, panel)

            # Legend
            legends = [("FREE",    self.color_free),
                       ("OBSTACLE",self.color_obstacle),
                       ("UNKNOWN", self.color_unknown)]
            lx = 10
            for name, col in legends:
                cv2.rectangle(panel, (lx, self.panel_h - 22),
                              (lx + 14, self.panel_h - 8), col, -1)
                cv2.putText(panel, name, (lx + 17, self.panel_h - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                lx += 90

            stats = freespace_result.statistics
            self._corner_label(
                panel,
                f"Free {stats['free_fraction']*100:.0f}%  "
                f"Obs {stats['obstacle_fraction']*100:.0f}%  "
                f"Unk {stats['unknown_fraction']*100:.0f}%",
            )

        self._draw_panel_title(panel, "Free-Space Overlay")
        return panel

    # ------------------------------------------------------------------
    # Panel 3 — B&W Camera Background + 2D Nav Grid + Path Calculation Lines
    # ------------------------------------------------------------------

    def _panel_navigation_bw(
        self,
        frame: np.ndarray,
        spatial_map,
        candidates: List,
        decision,
        geometry_3d_result: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Background: greyscale camera frame with edge-enhanced detection cues.
        Foreground: semi-transparent coloured grid + candidate corridors and obstacle conflict lines.
        """
        H, W = self.panel_h, self.panel_w
        panel_base = self._resize_panel(frame.copy())

        # --- 1. Convert to B&W with boosted contrast
        grey = cv2.cvtColor(panel_base, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        edges = cv2.Canny(grey, 40, 120)
        edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_col[:, :, 0] = 0; edges_col[:, :, 2] = 0  # faint green edge accents
        panel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(edges_col, 0.35, panel, 1.0, 0, panel)

        if spatial_map is None:
            self._draw_panel_title(panel, "B&W + Navigation Grid")
            return panel

        rows = spatial_map.rows
        cols = spatial_map.cols
        cw = W // cols
        ch = H // rows

        # --- 2. Grid overlay (cells coloured by occupancy and free probability)
        grid_overlay = panel.copy()
        for r in range(rows):
            for c in range(cols):
                cell = spatial_map.get_cell(r, c)
                if cell is None:
                    continue
                fp   = cell.free_prob
                occ  = cell.occupancy
                unc  = cell.uncertainty
                x1, y1 = c * cw, r * ch
                x2, y2 = x1 + cw - 1, y1 + ch - 1

                # Cell colour: green=free, red=occupied, amber=uncertain
                g = int(fp  * 160)
                b = int(occ * 160)
                a = int(unc * 80)
                color = (b, g, a)
                cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), (50, 50, 50), 1)

        cv2.addWeighted(grid_overlay, self.grid_alpha + 0.15, panel, 1 - self.grid_alpha - 0.15, 0, panel)

        # --- 3. Path Calculation Lines: Candidate Corridors & Conflict Detection
        if self.show_paths and candidates:
            # First pass: Non-selected candidates (draw trajectory lines + obstacle conflict marks)
            for cand in candidates:
                if cand.is_stop or not cand.points:
                    continue
                if cand.direction == decision.selected_path_direction:
                    continue  # selected path drawn prominently in pass 2

                pts_px = [(c * cw + cw // 2, r * ch + ch // 2) for r, c in cand.points]
                # Subtle line for unselected candidate
                for i in range(len(pts_px) - 1):
                    cv2.line(panel, pts_px[i], pts_px[i + 1], (90, 90, 110), 1, cv2.LINE_AA)

                # End tag with angle and 3D metric clearance
                end_x, end_y = pts_px[-1]
                c3d_val = getattr(cand, "corridor_3d_clearance_m", 0.0)
                tag_txt = f"{cand.angle_deg:+.0f}d ({c3d_val:.1f}m)" if c3d_val > 0.0 else f"{cand.angle_deg:+.0f}d"
                cv2.putText(panel, tag_txt, (end_x - 14, end_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 180, 220), 1)

                # If this path had high risk or collided with an obstacle, draw a red collision cross
                if cand.risk > 0.30 or cand.clearance < 0.25:
                    # Find first high-risk cell along the trajectory
                    for r, c in cand.points:
                        cell = spatial_map.get_cell(r, c)
                        if cell and cell.occupancy > 0.40:
                            ox, oy = c * cw + cw // 2, r * ch + ch // 2
                            # Red 'X' conflict marker
                            cv2.line(panel, (ox - 4, oy - 4), (ox + 4, oy + 4), (0, 0, 220), 2)
                            cv2.line(panel, (ox + 4, oy - 4), (ox - 4, oy + 4), (0, 0, 220), 2)
                            break

            # Second pass: SELECTED Candidate (Full walking corridor with boundaries and rungs)
            sel_cand = next((c for c in candidates if c.direction == decision.selected_path_direction and not c.is_stop), None)
            if sel_cand and len(sel_cand.points) >= 2:
                half_w = max(int((sel_cand.width * cw) / 2.0), 10)
                left_pts = [(c * cw + cw // 2 - half_w, r * ch + ch // 2) for r, c in sel_cand.points]
                right_pts = [(c * cw + cw // 2 + half_w, r * ch + ch // 2) for r, c in sel_cand.points]
                center_pts = [(c * cw + cw // 2, r * ch + ch // 2) for r, c in sel_cand.points]

                # Corridor polygon fill
                poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
                overlay = panel.copy()
                corridor_color = (0, 255, 255)  # Vibrant Neon Yellow
                cv2.fillPoly(overlay, [poly], (0, 180, 220))
                cv2.addWeighted(overlay, 0.25, panel, 0.75, 0, panel)

                # Corridor Left & Right boundary lines
                cv2.polylines(panel, [np.array(left_pts, dtype=np.int32)], False, corridor_color, 2, cv2.LINE_AA)
                cv2.polylines(panel, [np.array(right_pts, dtype=np.int32)], False, corridor_color, 2, cv2.LINE_AA)

                # Centerline with forward chevrons
                for i in range(len(center_pts) - 1):
                    cv2.line(panel, center_pts[i], center_pts[i + 1], corridor_color, 2, cv2.LINE_AA)
                for px, py in center_pts:
                    cv2.circle(panel, (px, py), 3, (0, 255, 0), -1)

                # Distance rungs across the corridor
                for i in range(0, len(left_pts), 2):
                    cv2.line(panel, left_pts[i], right_pts[i], (0, 200, 255), 1, cv2.LINE_AA)

                # Score pill at midpoint
                mid = center_pts[len(center_pts) // 2]
                c3d_sel = getattr(sel_cand, "corridor_3d_clearance_m", 0.0)
                c3d_str = f" | 3D: {c3d_sel:.1f}m" if c3d_sel > 0.0 else ""
                score_txt = f"{sel_cand.direction} ({sel_cand.score:.2f}{c3d_str})"
                (tw, th), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(panel, (mid[0] + 6, mid[1] - th - 6), (mid[0] + tw + 14, mid[1] + 4), (0, 0, 0), -1)
                cv2.putText(panel, score_txt, (mid[0] + 10, mid[1] - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, corridor_color, 1, cv2.LINE_AA)

        # --- 4. User position dot (bottom-centre)
        ux = (cols // 2) * cw + cw // 2
        uy = (rows - 1) * ch + ch // 2
        cv2.circle(panel, (ux, uy), 10, (0, 255, 255), -1)
        cv2.circle(panel, (ux, uy), 10, (0, 0, 0),     2)
        cv2.putText(panel, "YOU", (ux - 14, uy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        # --- 5. Direction arrow overlay (large, centred)
        self._draw_direction_arrow(panel, decision.command)

        self._draw_panel_title(panel, "3D Geometry & Corridor Lines")
        return panel

    # ------------------------------------------------------------------
    # Panel 4 — Decision Recommendation + Path Scoring Calculation Breakdown
    # ------------------------------------------------------------------

    def _panel_decision(
        self,
        decision,
        fps: float,
        frame_id: int,
        latency_ms: float,
        detection_result,
        freespace_result,
        candidates: Optional[List] = None,
        wall_proximity: Optional[dict] = None,
        geometry_3d_result: Optional[Any] = None,
    ) -> np.ndarray:
        panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        panel[:] = (12, 12, 28)  # very dark navy

        cmd   = decision.command
        color = self._CMD_COLORS.get(cmd, (200, 200, 200))
        font  = cv2.FONT_HERSHEY_DUPLEX

        # Glow effect: draw command in slightly larger blurred text first
        glow = panel.copy()
        cv2.putText(glow, cmd, (16, 52), font, 1.4, color, 6)
        cv2.GaussianBlur(glow, (11, 11), 0, glow)
        cv2.addWeighted(glow, 0.5, panel, 1.0, 0, panel)

        # Solid command text
        cv2.putText(panel, cmd, (16, 52), font, 1.4, color, 2)

        # Recommendation header pill
        cv2.putText(panel, "ACTION COMMAND", (16, 22), font, 0.46, (150, 150, 190), 1)

        # Score & Clearance bars
        bar_x0, bar_x1 = 16, self.panel_w - 16
        bar_y0, bar_y1 = 66, 78
        sf = max(0.0, min(1.0, decision.score))
        cv2.rectangle(panel, (bar_x0, bar_y0), (bar_x1, bar_y1), (35, 35, 55), -1)
        cv2.rectangle(panel, (bar_x0, bar_y0),
                      (bar_x0 + int((bar_x1 - bar_x0) * sf), bar_y1),
                      color, -1)
        cv2.putText(panel, f"Selected Score: {decision.score:.3f} | Clearance: {decision.clearance:.2f}",
                    (bar_x0, bar_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 220), 1)

        # State and Dynamic Conflict Banner
        y = 112
        dec_type = getattr(decision, "decision", "GO")
        conflict_txt = ""
        if getattr(decision, "dynamic_conflict", False):
            ttc = getattr(decision, "time_to_possible_conflict", None)
            conflict_txt = f" | [CONFLICT] TTC {ttc:.1f}s" if ttc is not None else " | [CONFLICT]"
        reason_txt = getattr(decision, "reason", "")
        cv2.putText(panel, f"STATE: {dec_type}{conflict_txt} | {reason_txt[:38]}",
                    (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # Wall proximity indicator + 3D metric clearance
        if wall_proximity:
            lat_w = wall_proximity.get("lateral_warning")
            front_coll = wall_proximity.get("is_frontal_collision", False)
            min_free = wall_proximity.get("min_frontal_free", 1.0)
            f3d = wall_proximity.get("frontal_clearance_3d_m")
            f3d_str = f" | 3D: {f3d:.1f}m" if f3d is not None else ""
            wall_str = f"WALL MONITOR: Front Free {min_free*100:.0f}%{f3d_str}"
            if front_coll:
                wall_str += " | [HAZARD] COLLISION"
                w_col = (50, 50, 255)
            elif lat_w:
                wall_str += f" | [CLOSE] {lat_w}"
                w_col = (0, 165, 255)
            else:
                wall_str += " | CLEAR"
                w_col = (100, 220, 100)
            y += 16
            cv2.putText(panel, wall_str, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, w_col, 1)

        # 3D Physical Geometry Pose & Recommendation
        if geometry_3d_result is not None:
            y += 16
            g_str = f"3D POSE: Floor ~{geometry_3d_result.floor_height_m:.1f}m | Pitch: {geometry_3d_result.pitch_deg:+.1f}d | 3D Rec: {geometry_3d_result.best_direction}"
            cv2.putText(panel, g_str, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 255), 1)

        # Performance & Stats row
        y += 18
        stats_str = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.0f}ms  |  Detections: {detection_result.num_detections}"
        cv2.putText(panel, stats_str, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 200, 140), 1)

        # Divider
        y += 10
        cv2.line(panel, (bar_x0, y), (bar_x1, y), (50, 50, 80), 1)

        # Stage A: Perceptual Support & Observation Validity Diagnostics
        crit_ids = getattr(decision, "critical_entities", [])
        cam_sup = getattr(decision, "camera_support", 1.0)
        admiss = getattr(decision, "admissibility_status", "ADMISSIBLE")
        crit_str = f"Ek:{crit_ids}" if crit_ids else "Ek:None"
        cam_col = (100, 255, 100) if admiss == "ADMISSIBLE" else (50, 150, 255)
        y += 16
        sup_str = f"CAM SUPPORT: {cam_sup:.2f} | {crit_str} | {admiss}"
        cv2.putText(panel, sup_str, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cam_col, 1)

        # Table Header
        y += 16
        cv2.putText(panel, "CANDIDATE CORRIDOR RANKING:", (bar_x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        y += 16
        col_hdr = f"{'DIR':10s} {'SCORE':5s} {'3D(m)':5s} {'CLEAR':5s} {'STATUS'}"
        cv2.putText(panel, col_hdr, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (120, 120, 160), 1)
        y += 16

        # Candidate table entries
        cand_dict = {c.direction: c for c in candidates} if candidates else {}
        corridor_records = getattr(decision, "corridor_support_records", {})

        for s in decision.all_scores:
            d_name = s["direction"]
            score_val = s["score"]
            is_sel = (d_name == decision.selected_path_direction)

            cand_obj = cand_dict.get(d_name)
            clr_val = cand_obj.clearance if cand_obj else 0.0
            risk_val = cand_obj.risk if cand_obj else 0.0
            c3d_val = getattr(cand_obj, "corridor_3d_clearance_m", 0.0) if cand_obj else 0.0
            sup_meta = corridor_records.get(d_name, {})
            rho_val = sup_meta.get("support", 1.0)

            if is_sel:
                status = "BEST (SELECTED)"
                row_col = (0, 255, 255)
                prefix = "> "
            elif sup_meta.get("status") == "INADMISSIBLE_LOW_SUPPORT":
                status = "LOW_RHO (INADM)"
                row_col = (50, 150, 255)
                prefix = "  "
            elif risk_val > 0.35 or clr_val < 0.20:
                status = "BLOCKED (OBS)"
                row_col = (70, 70, 220)
                prefix = "  "
            else:
                status = "AVAILABLE"
                row_col = (140, 140, 170)
                prefix = "  "

            row_str = f"{prefix}{d_name:8s} {score_val:5.2f} {c3d_val:4.1f}m {clr_val:5.2f} {status}"
            y += 15
            cv2.putText(panel, row_str, (bar_x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, row_col, 1)
            y += 16
            if y > self.panel_h - 20:
                break

        self._draw_panel_title(panel, "Decision & Path Calculation Metrics")
        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_direction_arrow(self, panel: np.ndarray, command: str) -> None:
        """Draw a large faint direction arrow in the centre of the panel."""
        H, W = panel.shape[:2]
        cx, cy = W // 2, H // 2
        color = self._CMD_COLORS.get(command, (200, 200, 200))
        overlay = panel.copy()
        size = min(W, H) // 5

        if command == "STRAIGHT":
            cv2.arrowedLine(overlay, (cx, cy + size), (cx, cy - size),
                            color, 6, tipLength=0.35)
        elif command in ("SLIGHT_LEFT", "LEFT"):
            ang = 0.5 if "SLIGHT" in command else 1.0
            ex = int(cx - size * ang); ey = cy - size // 2
            cv2.arrowedLine(overlay, (cx + size // 2, cy + size // 2),
                            (ex, ey), color, 6, tipLength=0.35)
        elif command in ("SLIGHT_RIGHT", "RIGHT"):
            ang = 0.5 if "SLIGHT" in command else 1.0
            ex = int(cx + size * ang); ey = cy - size // 2
            cv2.arrowedLine(overlay, (cx - size // 2, cy + size // 2),
                            (ex, ey), color, 6, tipLength=0.35)
        elif command in ("STOP", "CAUTION"):
            # X marker
            cv2.line(overlay, (cx - size, cy - size), (cx + size, cy + size), color, 7)
            cv2.line(overlay, (cx + size, cy - size), (cx - size, cy + size), color, 7)

        cv2.addWeighted(overlay, 0.22, panel, 0.78, 0, panel)

    def _draw_panel_title(self, panel: np.ndarray, title: str) -> None:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(panel, title, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 230), 1)
        cv2.line(panel, (0, 28), (panel.shape[1], 28), (50, 50, 70), 1)

    def _corner_label(self, panel: np.ndarray, text: str) -> None:
        """Small label in top-right corner."""
        H, W = panel.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(panel, (W - tw - 10, 2), (W - 2, th + 8), (0, 0, 0), -1)
        cv2.putText(panel, text, (W - tw - 6, th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 200), 1)

    def _draw_disclaimer(self, composite: np.ndarray) -> None:
        h, w = composite.shape[:2]
        cv2.rectangle(composite, (0, h - 24), (w, h), (8, 8, 40), -1)
        msg = "RESEARCH PROTOTYPE - NOT FOR REAL-WORLD MOBILITY USE"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(composite, msg, (w // 2 - tw // 2, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 220), 1)

    def _resize_panel(self, img: np.ndarray) -> np.ndarray:
        if img.shape[1] != self.panel_w or img.shape[0] != self.panel_h:
            img = cv2.resize(img, (self.panel_w, self.panel_h))
        return img

    def _ensure_size(self, img: np.ndarray) -> np.ndarray:
        return self._resize_panel(img)

    def _init_tts(self) -> None:
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 150)
            logger.info("TTS engine initialized.")
        except Exception as exc:
            logger.warning("TTS init failed (%s) — disabling.", exc)
            self.tts_enabled = False

    def _speak(self, text: str) -> None:
        if self._tts_engine:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS speak error: %s", exc)
