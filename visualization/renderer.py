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
            box = getattr(obj, "bbox", getattr(obj, "box", None))
            if box is None or len(box) < 4:
                continue
            x1 = int(box[0] * sx); y1 = int(box[1] * sy)
            x2 = int(box[2] * sx); y2 = int(box[3] * sy)
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

    # =========================================================================
    # STAGE B: DUAL-CAMERA MERGED DASHBOARD
    # =========================================================================

    def render_dual(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        detection_result0,
        tracks0: List,
        detection_result1,
        tracks1: List,
        entity_states: dict,
        associations: List,
        candidates: List,
        decision,
        corridor_support_records: dict,
        fps: float = 0.0,
        frame_id: int = 0,
        latency_ms: float = 0.0,
        freespace_result0: Optional[Any] = None,
        freespace_result1: Optional[Any] = None,
        spatial_map: Optional[Any] = None,
        geometry_3d_result: Optional[Any] = None,
        wall_proximity: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Renders the Stage B Dual-Camera Merged Dashboard:
          Top: CAM0 (Primary Left, -17.5 deg) and CAM1 (Primary Right, +17.5 deg) feeds side-by-side
               with free-space segmentation, automotive reversing corridor guidelines,
               visual overlap seam shading, and cross-camera track association links.
          Bottom-Left: Unified 2D Egocentric Occupancy Grid & Dual FOV Cones with path curves.
          Bottom-Right: Stage B predictive handover telemetry, candidate corridor scoring table,
                        and corridor admissibility status.
        """
        top_w = 675
        top_h = 380

        # 1. Render Camera 0 & Camera 1 panels
        p_cam0 = self._panel_dual_feed(
            frame0, detection_result0, tracks0,
            camera_id="CAM0", title="CAM0: PRIMARY LEFT (Yaw -17.5 deg, FOV [-50 to +15 deg])",
            overlap_side="right", target_w=top_w, target_h=top_h,
            entity_states=entity_states,
            freespace_result=freespace_result0,
            candidates=candidates,
            decision=decision,
            wall_proximity=wall_proximity,
            fps=fps,
            frame_id=frame_id,
            spatial_map=spatial_map,
        )
        p_cam1 = self._panel_dual_feed(
            frame1, detection_result1, tracks1,
            camera_id="CAM1", title="CAM1: PRIMARY RIGHT (Yaw +17.5 deg, FOV [-15 to +50 deg])",
            overlap_side="left", target_w=top_w, target_h=top_h,
            entity_states=entity_states,
            freespace_result=freespace_result1,
            candidates=candidates,
            decision=decision,
            wall_proximity=wall_proximity,
            fps=fps,
            frame_id=frame_id,
            spatial_map=spatial_map,
        )

        # Center separator bar
        sep = np.zeros((top_h, 10, 3), dtype=np.uint8)
        sep[:] = (40, 40, 45)
        top_composite = np.hstack([p_cam0, sep, p_cam1])

        # Draw cross-camera association lines connecting matching boxes
        self._draw_cross_camera_links(
            top_composite, top_w, 10, tracks0, tracks1, associations, detection_result0, detection_result1
        )

        # 2. Render Bottom Panels
        bot_bev_w = 580
        bot_telem_w = 1360 - bot_bev_w
        bot_h = 360

        p_bev = self._panel_dual_bev(
            entity_states=entity_states,
            candidates=candidates,
            decision=decision,
            target_w=bot_bev_w,
            target_h=bot_h,
            spatial_map=spatial_map,
        )
        p_telem = self._panel_dual_telemetry(
            entity_states=entity_states,
            corridor_support_records=corridor_support_records,
            decision=decision,
            candidates=candidates,
            wall_proximity=wall_proximity,
            geometry_3d_result=geometry_3d_result,
            fps=fps,
            frame_id=frame_id,
            latency_ms=latency_ms,
            target_w=bot_telem_w,
            target_h=bot_h,
        )

        bot_composite = np.hstack([p_bev, p_telem])

        # Stack into full 1360x768 composite
        composite = np.vstack([top_composite, bot_composite])
        self._draw_disclaimer(composite)

        if self.tts_enabled and decision and decision.command != self._last_tts_cmd:
            self._speak(decision.command)
            self._last_tts_cmd = decision.command

        return composite

    def _panel_dual_feed(
        self,
        frame: np.ndarray,
        det_result,
        tracks: List,
        camera_id: str,
        title: str,
        overlap_side: str,
        target_w: int,
        target_h: int,
        entity_states: dict,
        freespace_result: Optional[Any] = None,
        candidates: Optional[List] = None,
        decision: Optional[Any] = None,
        wall_proximity: Optional[dict] = None,
        fps: float = 0.0,
        frame_id: int = 0,
        spatial_map: Optional[Any] = None,
    ) -> np.ndarray:
        p = cv2.resize(frame, (target_w, target_h))
        orig_h, orig_w = frame.shape[:2]
        sx = target_w / float(orig_w)
        sy = target_h / float(orig_h)

        # 1. Free-space colour overlay (translucent green on navigable floor)
        if freespace_result is not None and self.show_freespace and hasattr(freespace_result, "label_map"):
            lm = cv2.resize(
                freespace_result.label_map,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            fs_overlay = np.zeros_like(p)
            fs_overlay[lm == 0] = self.color_free      # FREE -> green
            fs_overlay[lm == 1] = self.color_obstacle  # OBS -> red
            cv2.addWeighted(fs_overlay, self.fs_alpha, p, 1.0 - self.fs_alpha, 0, p)

        # 2. Draw subtle overlap seam zone
        overlay = p.copy()
        if overlap_side == "right":
            # Right ~25% corresponds to [+17.5 deg, +32.5 deg]
            x_start = int(target_w * 0.74)
            cv2.rectangle(overlay, (x_start, 28), (target_w, target_h), (0, 180, 240), -1)
            cv2.putText(p, "OVERLAP SEAM [+17.5 deg, +32.5 deg]", (x_start + 6, target_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.line(p, (x_start, 28), (x_start, target_h), (0, 200, 255), 1, cv2.LINE_AA)
        else:
            # Left ~25% corresponds to [-32.5 deg, -17.5 deg] local
            x_end = int(target_w * 0.26)
            cv2.rectangle(overlay, (0, 28), (x_end, target_h), (0, 180, 240), -1)
            cv2.putText(p, "OVERLAP SEAM", (8, target_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.line(p, (x_end, 28), (x_end, target_h), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, p, 0.85, 0, p)

        # 3. Dynamic perspective corridor guidelines & Decision HUD on CAM0
        if candidates and decision and camera_id == "CAM0":
            self._draw_reversing_camera_guidelines(
                p, spatial_map=spatial_map, candidates=candidates,
                decision=decision, wall_proximity=wall_proximity
            )
            self._draw_dual_decision_hud(
                p, decision=decision, fps=fps, frame_id=frame_id, wall_proximity=wall_proximity
            )

        # 4. Draw bounding boxes
        if det_result and hasattr(det_result, "objects"):
            for obj in det_result.objects:
                box = getattr(obj, "bbox", getattr(obj, "box", None))
                if box is None or len(box) < 4:
                    continue
                x1 = int(box[0] * sx)
                y1 = int(box[1] * sy)
                x2 = int(box[2] * sx)
                y2 = int(box[3] * sy)

                # Look up entity responsibility state if tracked
                box_color = (0, 230, 0)
                state_badge = ""
                tid = getattr(obj, "track_id", None)
                if tid is not None:
                    for ent in entity_states.values():
                        obs = ent.observations.get(camera_id)
                        if obs and obs.track_id == tid:
                            is_overlap = (
                                "CAM0" in ent.observations and "CAM1" in ent.observations
                                and getattr(ent.observations["CAM0"], "in_fov", False)
                                and getattr(ent.observations["CAM1"], "in_fov", False)
                            )
                            if is_overlap:
                                box_color = (0, 215, 255)  # Gold
                                state_badge = " [REAL OVERLAP]"
                            else:
                                st = ent.responsibility_state
                                if st == "PRE_ARM":
                                    box_color = (0, 215, 255)
                                    state_badge = " [PRE-ARM]"
                                elif st == "TRANSFER":
                                    box_color = (255, 50, 255)
                                    state_badge = " [TRANSFER]"
                                elif st == "SECONDARY" and camera_id == "CAM1":
                                    box_color = (0, 255, 120)
                                    state_badge = " [RESPONS]"
                            break

                cv2.rectangle(p, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"{obj.class_name}:{obj.confidence:.2f}{state_badge}"
                cv2.rectangle(p, (x1, max(28, y1 - 20)), (x1 + len(lbl) * 8 + 6, max(48, y1)), (0, 0, 0), -1)
                cv2.putText(p, lbl, (x1 + 4, max(42, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)

        # Title bar
        cv2.rectangle(p, (0, 0), (target_w, 28), (20, 20, 25), -1)
        cv2.putText(p, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 240), 1, cv2.LINE_AA)

        # Draw Overlap zone indicator boundary on camera view
        mid_x = int(target_w * 0.50)
        if camera_id == "CAM0":
            cv2.line(p, (mid_x, 28), (mid_x, target_h), (0, 180, 230), 1, cv2.LINE_AA)
            cv2.putText(p, "CENTER OVERLAP ->", (target_w - 150, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
        elif camera_id == "CAM1":
            cv2.line(p, (mid_x, 28), (mid_x, target_h), (0, 180, 230), 1, cv2.LINE_AA)
            cv2.putText(p, "<- CENTER OVERLAP", (target_w - 150, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

        # Bottom Sub-banner (Camera FOV specs)
        cv2.rectangle(p, (0, target_h - 22), (target_w, target_h), (15, 15, 20), -1)
        if camera_id == "CAM0":
            spec_txt = "CAM0 | PRIMARY LEFT | Yaw: -17.5 deg | FOV: [-50 deg, +15 deg]"
            cv2.putText(p, spec_txt, (10, target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 200, 0), 1)
        else:
            spec_txt = "CAM1 | PRIMARY RIGHT | Yaw: +17.5 deg | FOV: [-15 deg, +50 deg]"
            cv2.putText(p, spec_txt, (10, target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 80, 200), 1)

        return p

    def _draw_dual_decision_hud(
        self,
        panel: np.ndarray,
        decision,
        fps: float = 0.0,
        frame_id: int = 0,
        wall_proximity: Optional[dict] = None,
    ) -> None:
        """Render navigation decision pill on top-left of CAM0 feed."""
        if decision is None:
            return

        cmd = getattr(decision, "command", "STOP")
        is_wall_coll = bool(wall_proximity and wall_proximity.get("is_frontal_collision"))
        lat_warn = (wall_proximity.get("lateral_warning") if wall_proximity else None) or getattr(decision, "wall_warning", None)
        has_wall_alert = is_wall_coll or bool(lat_warn)

        hud_w = 235
        hud_h = 76 if not has_wall_alert else 92
        x1, y1 = 12, 34
        x2, y2 = x1 + hud_w, y1 + hud_h

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

        overlay = panel.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)

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
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, txt_col, 2, cv2.LINE_AA)

        # Line 3: Clearance and Score metrics
        clr_pct = int(getattr(decision, "clearance", 0.0) * 100)
        score_val = getattr(decision, "score", 0.0)
        metrics_text = f"Clearance: {clr_pct}% | Score: {score_val:.2f}"
        cv2.putText(panel, metrics_text, (x1 + 8, y1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 210), 1, cv2.LINE_AA)

        # Line 4: Optional Wall alert
        if is_wall_coll:
            if getattr(decision, "decision", "") == "TURN":
                alert_txt = f"STEERING AROUND HAZARD -> {cmd}"
                cv2.putText(panel, alert_txt, (x1 + 8, y1 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1, cv2.LINE_AA)
            else:
                alert_txt = "HAZARD: WALL COLLISION"
                cv2.putText(panel, alert_txt, (x1 + 8, y1 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 255), 1, cv2.LINE_AA)
        elif lat_warn:
            alert_txt = f"WALL PROXIMITY: {lat_warn}"
            cv2.putText(panel, alert_txt, (x1 + 8, y1 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

    def _draw_cross_camera_links(
        self,
        top_composite: np.ndarray,
        cam_w: int,
        sep_w: int,
        tracks0: List,
        tracks1: List,
        associations: List,
        det0,
        det1,
    ) -> None:
        """Draws association lines connecting corresponding objects across CAM0 and CAM1."""
        if not associations:
            return

        for m in associations:
            o0, o1, conf = m
            if conf < 0.50:
                continue

            # Find matching bboxes
            b0_center = None
            b1_center = None

            if det0 and hasattr(det0, "objects"):
                for obj in det0.objects:
                    if getattr(obj, "track_id", None) == o0.track_id:
                        box0 = getattr(obj, "bbox", getattr(obj, "box", None))
                        if box0 is not None and len(box0) >= 4:
                            b0_center = (
                                int(((box0[0] + box0[2]) / 2.0) * (cam_w / 640.0)),
                                int(((box0[1] + box0[3]) / 2.0) * (380.0 / 480.0)),
                            )
                        break

            if det1 and hasattr(det1, "objects"):
                for obj in det1.objects:
                    if getattr(obj, "track_id", None) == o1.track_id:
                        box1 = getattr(obj, "bbox", getattr(obj, "box", None))
                        if box1 is not None and len(box1) >= 4:
                            b1_center = (
                                cam_w + sep_w + int(((box1[0] + box1[2]) / 2.0) * (cam_w / 640.0)),
                                int(((box1[1] + box1[3]) / 2.0) * (380.0 / 480.0)),
                            )
                        break

            if b0_center and b1_center:
                # Draw connecting line
                cv2.line(top_composite, b0_center, b1_center, (0, 255, 255), 2, cv2.LINE_AA)
                # Midpoint tag
                mx = (b0_center[0] + b1_center[0]) // 2
                my = (b0_center[1] + b1_center[1]) // 2
                tag = f"MATCH A01: {conf*100:.0f}%"
                cv2.rectangle(top_composite, (mx - 48, my - 12), (mx + 48, my + 10), (0, 0, 0), -1)
                cv2.rectangle(top_composite, (mx - 48, my - 12), (mx + 48, my + 10), (0, 255, 255), 1)
                cv2.putText(top_composite, tag, (mx - 44, my + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)

    def _panel_dual_bev(
        self,
        entity_states: dict,
        candidates: List,
        decision,
        target_w: int,
        target_h: int,
        spatial_map: Optional[Any] = None,
    ) -> np.ndarray:
        p = np.full((target_h, target_w, 3), 15, dtype=np.uint8)

        # Title bar
        cv2.rectangle(p, (0, 0), (target_w, 28), (25, 25, 30), -1)
        cv2.putText(p, "UNIFIED 2D SPATIAL MAP & DUAL FOV CONES", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 230), 1, cv2.LINE_AA)
        cv2.line(p, (0, 28), (target_w, 28), (60, 60, 70), 1)

        ox = target_w // 2
        oy = target_h - 25
        scale = 55.0  # pixels per meter

        # 1. Draw 2D Occupancy Grid cells from SpatialMap
        if spatial_map is not None:
            rows = spatial_map.rows
            cols = spatial_map.cols
            grid_layer = p.copy()

            for r in range(rows):
                y_m = ((rows - 1 - r) + 0.5) * (5.0 / float(rows))
                cy_px = int(oy - y_m * scale)
                hh = max(2, int(0.5 * (5.0 / float(rows)) * scale))

                for c in range(cols):
                    x_m = (c - (cols - 1) / 2.0) * (3.2 / float(cols))
                    cx_px = int(ox + x_m * scale)
                    hw = max(2, int(0.5 * (3.2 / float(cols)) * scale))

                    cell = spatial_map.get_cell(r, c)
                    if cell is None:
                        continue
                    fp = cell.free_prob
                    occ = cell.occupancy
                    unc = cell.uncertainty

                    if occ > 0.35:
                        c_col = (30, 30, int(min(255, occ * 200 + 55)))  # Red obstacle
                    elif fp > 0.45:
                        c_col = (20, int(min(255, fp * 190 + 50)), 35)   # Green free space
                    else:
                        c_col = (10, int(unc * 80), int(unc * 140))       # Amber uncertain

                    x1, y1 = cx_px - hw, cy_px - hh
                    x2, y2 = cx_px + hw, cy_px + hh
                    cv2.rectangle(grid_layer, (x1, y1), (x2, y2), c_col, -1)
                    cv2.rectangle(grid_layer, (x1, y1), (x2, y2), (35, 35, 45), 1)

            cv2.addWeighted(grid_layer, 0.42, p, 0.58, 0, p)

        # 2. Concentric distance arcs (1m to 5m)
        for dist_m in range(1, 6):
            r_px = int(dist_m * scale)
            cv2.ellipse(p, (ox, oy), (r_px, r_px), 0, 180, 360, (50, 50, 65), 1)
            cv2.putText(p, f"{dist_m}m", (ox + 4, oy - r_px + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 110), 1)

        # 3. FOV Cones: Max distance 5.2m
        max_r = int(5.2 * scale)

        # CAM0: [-50.0 deg, +15.0 deg] (Cyan / Left Primary yaw = -17.5 deg)
        ang_c0_l = math.radians(-50.0)
        ang_c0_r = math.radians(+15.0)
        pt_c0_l = (int(ox + max_r * math.sin(ang_c0_l)), int(oy - max_r * math.cos(ang_c0_l)))
        pt_c0_r = (int(ox + max_r * math.sin(ang_c0_r)), int(oy - max_r * math.cos(ang_c0_r)))
        cv2.line(p, (ox, oy), pt_c0_l, (220, 200, 0), 1, cv2.LINE_AA)
        cv2.line(p, (ox, oy), pt_c0_r, (220, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(p, "CAM0 (-17.5 deg)", (pt_c0_l[0] - 25, pt_c0_l[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 200, 0), 1)

        # CAM1: [-15.0 deg, +50.0 deg] (Magenta / Right Primary yaw = +17.5 deg)
        ang_c1_l = math.radians(-15.0)
        ang_c1_r = math.radians(+50.0)
        pt_c1_l = (int(ox + max_r * math.sin(ang_c1_l)), int(oy - max_r * math.cos(ang_c1_l)))
        pt_c1_r = (int(ox + max_r * math.sin(ang_c1_r)), int(oy - max_r * math.cos(ang_c1_r)))
        cv2.line(p, (ox, oy), pt_c1_l, (200, 80, 200), 1, cv2.LINE_AA)
        cv2.line(p, (ox, oy), pt_c1_r, (200, 80, 200), 1, cv2.LINE_AA)
        cv2.putText(p, "CAM1 (+17.5 deg)", (pt_c1_r[0] - 20, pt_c1_r[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 80, 200), 1)

        # Real Overlap Entities in center [-15 deg, +15 deg]
        real_overlap_ents = [
            ent for ent in entity_states.values()
            if "CAM0" in ent.observations and "CAM1" in ent.observations
            and getattr(ent.observations["CAM0"], "in_fov", False)
            and getattr(ent.observations["CAM1"], "in_fov", False)
        ]
        num_overlap = len(real_overlap_ents)

        # Dynamic Overlap Wedge: [-15.0 deg, +15.0 deg] centered at 0.0 deg
        ov_poly = np.array([
            (ox, oy),
            pt_c1_l,
            (int(ox + max_r * math.sin(0.0)), int(oy - max_r * math.cos(0.0))),
            pt_c0_r,
        ], dtype=np.int32)
        ov_overlay = p.copy()
        ov_col = (0, 200, 255) if num_overlap > 0 else (0, 140, 220)
        ov_alpha = 0.28 if num_overlap > 0 else 0.14
        cv2.fillConvexPoly(ov_overlay, ov_poly, ov_col)
        cv2.addWeighted(ov_overlay, ov_alpha, p, 1.0 - ov_alpha, 0, p)

        # Optical center forward axis (0 deg)
        cv2.line(p, (ox, oy), (int(ox), int(oy - max_r)), (0, 220, 255), 1, cv2.LINE_AA)
        ov_banner = f"REAL OVERLAP [-15..+15 deg]: {num_overlap} FUSED TARGETS" if num_overlap > 0 else "CENTER OVERLAP [-15..+15 deg]: 0 TARGETS"
        cv2.putText(p, ov_banner, (ox - 95, oy - max_r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1)

        # 4. Draw Candidate Path Trajectories
        cmd_name = decision.command if decision else "STOP"
        sel_dir = getattr(decision, "selected_path_direction", cmd_name)
        rows = spatial_map.rows if spatial_map else 12
        cols = spatial_map.cols if spatial_map else 20

        if candidates:
            # Pass 1: Draw unselected paths (cool slate grey)
            for cand in candidates:
                if cand.is_stop or not cand.points:
                    continue
                if cand.direction == sel_dir:
                    continue
                pts_screen = [(ox, oy)]
                for r, c in cand.points:
                    y_m = ((rows - 1 - r) + 0.5) * (5.0 / float(rows))
                    x_m = (c - (cols - 1) / 2.0) * (3.2 / float(cols))
                    pts_screen.append((int(ox + x_m * scale), int(oy - y_m * scale)))

                if len(pts_screen) > 1:
                    cv2.polylines(p, [np.array(pts_screen, dtype=np.int32)], False, (80, 80, 100), 1, cv2.LINE_AA)
                    end_pt = pts_screen[-1]
                    c3d = getattr(cand, "corridor_3d_clearance_m", 0.0)
                    tag = f"{cand.direction[:2]} ({c3d:.1f}m)" if c3d > 0 else cand.direction[:2]
                    cv2.putText(p, tag, (end_pt[0] - 14, end_pt[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 140), 1, cv2.LINE_AA)

            # Pass 2: Draw selected path prominently
            for cand in candidates:
                if cand.direction != sel_dir or cand.is_stop or not cand.points:
                    continue
                pts_screen = [(ox, oy)]
                for r, c in cand.points:
                    y_m = ((rows - 1 - r) + 0.5) * (5.0 / float(rows))
                    x_m = (c - (cols - 1) / 2.0) * (3.2 / float(cols))
                    pts_screen.append((int(ox + x_m * scale), int(oy - y_m * scale)))

                if len(pts_screen) > 1:
                    path_col = (0, 240, 255) if decision and decision.decision == "TURN" else (0, 255, 120)
                    cv2.polylines(p, [np.array(pts_screen, dtype=np.int32)], False, path_col, 3, cv2.LINE_AA)
                    for pt in pts_screen[1:]:
                        cv2.circle(p, pt, 3, (255, 255, 255), -1)
                    end_pt = pts_screen[-1]
                    c3d = getattr(cand, "corridor_3d_clearance_m", 0.0)
                    tag = f"{cand.direction} ({c3d:.1f}m)" if c3d > 0 else cand.direction
                    cv2.putText(p, tag, (end_pt[0] - 25, end_pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, path_col, 1, cv2.LINE_AA)

        # 5. Draw Wearer Reference Position
        cv2.circle(p, (ox, oy), 7, (0, 255, 255), -1)
        cv2.circle(p, (ox, oy), 11, (180, 180, 200), 1)
        cv2.putText(p, "WEARER", (ox - 18, oy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)

        # 6. Draw Tracked Multi-Camera Entities with Dynamic Real Overlap Sightlines
        for ent in entity_states.values():
            w_pos = None
            for obs in ent.observations.values():
                if obs and obs.in_fov and obs.world_pos:
                    w_pos = obs.world_pos
                    break

            if w_pos is not None:
                gx, gy = w_pos
                ex = int(ox + gx * scale)
                ey = int(oy - gy * scale)

                is_real_overlap = (
                    "CAM0" in ent.observations and "CAM1" in ent.observations
                    and getattr(ent.observations["CAM0"], "in_fov", False)
                    and getattr(ent.observations["CAM1"], "in_fov", False)
                )

                if is_real_overlap:
                    # Draw converging dual-camera sightlines
                    cv2.line(p, (ox - 15, oy), (ex, ey), (220, 200, 0), 1, cv2.LINE_AA)
                    cv2.line(p, (ox + 15, oy), (ex, ey), (200, 80, 200), 1, cv2.LINE_AA)
                    ecol = (0, 235, 255)  # Glowing Amber-Gold
                    lbl = f"#{ent.global_entity_id} [REAL OVERLAP]"
                else:
                    st = ent.responsibility_state
                    if st == "PRIMARY":
                        ecol = (240, 160, 0)
                    elif st == "PRE_ARM":
                        ecol = (0, 215, 255)
                    elif st == "TRANSFER":
                        ecol = (255, 50, 255)
                    else:
                        ecol = (0, 255, 120)
                    lbl = f"#{ent.global_entity_id} [{st}]"

                cv2.circle(p, (ex, ey), 8, ecol, -1)
                cv2.circle(p, (ex, ey), 10, (255, 255, 255), 1)
                cv2.putText(p, lbl, (ex + 12, ey + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, ecol, 1, cv2.LINE_AA)

        return p

    def _panel_dual_telemetry(
        self,
        entity_states: dict,
        corridor_support_records: dict,
        decision,
        candidates: Optional[List] = None,
        wall_proximity: Optional[dict] = None,
        geometry_3d_result: Optional[Any] = None,
        fps: float = 0.0,
        frame_id: int = 0,
        latency_ms: float = 0.0,
        target_w: int = 780,
        target_h: int = 360,
    ) -> np.ndarray:
        p = np.full((target_h, target_w, 3), 18, dtype=np.uint8)

        # Title bar
        cv2.rectangle(p, (0, 0), (target_w, 28), (25, 25, 30), -1)
        cv2.putText(p, "STAGE B: PREDICTIVE HANDOVER TELEMETRY & ADMISSIBILITY", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 230), 1, cv2.LINE_AA)
        cv2.line(p, (0, 28), (target_w, 28), (60, 60, 70), 1)

        col_w = target_w // 2 - 10

        # ── Left Column: Multi-Camera Responsibility & Geometry ──────
        cv2.putText(p, "ENTITY RESPONSIBILITY CONTROLLER", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(p, (12, 53), (col_w, 53), (50, 50, 60), 1)

        y_ent = 70
        displayed_ents = 0
        for ent in list(entity_states.values())[:3]:
            displayed_ents += 1
            st = ent.responsibility_state
            resp = ent.responsible_camera_id or "CAM0"

            st_col = (0, 255, 120) if st == "SECONDARY" else ((255, 50, 255) if st == "TRANSFER" else ((0, 215, 255) if st == "PRE_ARM" else (240, 160, 0)))
            cv2.rectangle(p, (12, y_ent - 14), (80, y_ent + 4), st_col, -1)
            cv2.putText(p, st, (16, y_ent), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(p, f"Entity #{ent.global_entity_id} | Resp: {resp}", (88, y_ent),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

            # Observation rho bars
            obs0 = ent.observations.get("CAM0")
            obs1 = ent.observations.get("CAM1")
            r0 = obs0.rho_fov if (obs0 and obs0.in_fov) else 0.0
            r1 = obs1.rho_fov if (obs1 and obs1.in_fov) else 0.0

            # CAM0 bar
            y_bar = y_ent + 16
            cv2.putText(p, f"CAM0 rho: {r0:.2f}", (20, y_bar), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 0), 1)
            cv2.rectangle(p, (108, y_bar - 10), (108 + int(r0 * 95), y_bar - 2), (180, 180, 0), -1)
            cv2.rectangle(p, (108, y_bar - 10), (203, y_bar - 2), (60, 60, 70), 1)

            # CAM1 bar
            y_bar2 = y_ent + 31
            cv2.putText(p, f"CAM1 rho: {r1:.2f}", (20, y_bar2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 80, 200), 1)
            cv2.rectangle(p, (108, y_bar2 - 10), (108 + int(r1 * 95), y_bar2 - 2), (200, 80, 200), -1)
            cv2.rectangle(p, (108, y_bar2 - 10), (203, y_bar2 - 2), (60, 60, 70), 1)

            # Association
            assoc_txt = f"A01: {ent.association_confidence*100:.0f}%" if ent.association_confidence > 0 else "A01: --"
            cv2.putText(p, assoc_txt, (218, y_bar), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 220, 255), 1)

            y_ent += 54

        if displayed_ents == 0:
            cv2.putText(p, "No critical entities in transition zone", (20, y_ent + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1)
            y_ent += 24

        # Wall & 3D Egocentric Geometry Monitor
        y_geo = max(y_ent + 10, 225)
        cv2.putText(p, "WALL MONITOR & 3D POSE", (12, y_geo),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(p, (12, y_geo + 5), (col_w, y_geo + 5), (50, 50, 60), 1)

        y_geo += 22
        if wall_proximity:
            lat_w = wall_proximity.get("lateral_warning")
            front_coll = wall_proximity.get("is_frontal_collision", False)
            min_free = wall_proximity.get("min_frontal_free", 1.0)
            f3d = wall_proximity.get("frontal_clearance_3d_m")
            f3d_str = f" | 3D: {f3d:.1f}m" if f3d is not None else ""
            wall_str = f"Wall: Front Free {min_free*100:.0f}%{f3d_str}"
            if front_coll:
                wall_str += " | [HAZARD] COLLISION"
                w_col = (50, 50, 255)
            elif lat_w:
                wall_str += f" | [CLOSE] {lat_w}"
                w_col = (0, 165, 255)
            else:
                wall_str += " | CLEAR"
                w_col = (100, 220, 100)
            cv2.putText(p, wall_str, (16, y_geo), cv2.FONT_HERSHEY_SIMPLEX, 0.35, w_col, 1)

        y_geo += 18
        if geometry_3d_result is not None:
            g_str = f"3D Pose: Floor ~{geometry_3d_result.floor_height_m:.1f}m | Pitch: {geometry_3d_result.pitch_deg:+.1f}d | Rec: {geometry_3d_result.best_direction}"
            cv2.putText(p, g_str, (16, y_geo), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 210, 240), 1)

        # ── Right Column: Candidate Corridor Ranking & Support Gate ───
        right_x = col_w + 15
        cv2.putText(p, "CANDIDATE CORRIDOR RANKING & SUPPORT GATE", (right_x, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(p, (right_x, 53), (target_w - 15, 53), (50, 50, 60), 1)

        y_cor = 70
        hdr_txt = f"{'DIR':12s} {'SCORE':5s} {'3D(m)':5s} {'CLEAR':5s} {'SUP_B':5s} {'STATUS'}"
        cv2.putText(p, hdr_txt, (right_x, y_cor), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 180), 1)
        y_cor += 18

        cand_dict = {c.direction: c for c in candidates} if candidates else {}
        sel_dir = getattr(decision, "selected_path_direction", (decision.command if decision else "STOP"))

        for c_name in ["STRAIGHT", "SLIGHT_LEFT", "SLIGHT_RIGHT", "LEFT", "RIGHT", "STOP"]:
            cand = cand_dict.get(c_name)
            rec = corridor_support_records.get(c_name)
            is_sel = (c_name == sel_dir)

            score_val = cand.score if cand else (0.10 if c_name == "STOP" else 0.0)
            clr_val = cand.clearance if cand else 0.0
            c3d_val = getattr(cand, "corridor_3d_clearance_m", 0.0) if cand else 0.0
            sup_val = rec.multi_cam_support if rec else 1.0
            adm = rec.is_admissible if rec else True

            if is_sel:
                status = "BEST (SELECTED)"
                row_col = (0, 255, 255)
                prefix = "> "
            elif c_name == "STOP":
                status = "SAFE FALLBACK"
                row_col = (130, 130, 150)
                prefix = "  "
            elif not adm:
                status = "REJECTED (LOW RHO)"
                row_col = (50, 50, 220)
                prefix = "  "
            elif clr_val < 0.20 or (cand and cand.risk > 0.35):
                status = "BLOCKED (OBS)"
                row_col = (70, 70, 220)
                prefix = "  "
            else:
                status = "AVAILABLE"
                row_col = (50, 220, 50)
                prefix = "  "

            row_str = f"{prefix}{c_name:10s} {score_val:5.2f} {c3d_val:4.1f}m {clr_val:5.2f} {sup_val:5.2f} {status}"
            cv2.putText(p, row_str, (right_x, y_cor), cv2.FONT_HERSHEY_SIMPLEX, 0.35, row_col, 1, cv2.LINE_AA)
            y_cor += 19

        # ── Bottom Action & Performance Banner ────────────────────────
        cv2.line(p, (15, target_h - 75), (target_w - 15, target_h - 75), (50, 50, 65), 1)

        cmd = decision.command if decision else "STOP"
        cmd_col = self._CMD_COLORS.get(cmd, (0, 255, 255))
        cv2.rectangle(p, (15, target_h - 68), (220, target_h - 32), cmd_col, -1)
        cv2.putText(p, f"CMD: {cmd}", (25, target_h - 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

        # Subtitle metrics
        clr_pct = int(getattr(decision, "clearance", 0.0) * 100)
        dec_score = getattr(decision, "score", 0.0)
        reason_txt = getattr(decision, "reason", "")
        summary_str = f"Score: {dec_score:.2f} | Clr: {clr_pct}% | {reason_txt[:42]}"
        cv2.putText(p, summary_str, (235, target_h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        perf_str = f"FPS: {fps:.1f} | Frame: {frame_id} | Pipeline: {latency_ms:.0f}ms | Mode: B2_PREDICTIVE"
        cv2.putText(p, perf_str, (235, target_h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 190), 1, cv2.LINE_AA)

        return p


