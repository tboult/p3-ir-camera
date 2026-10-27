#!/usr/bin/env python3
"""Dual-Stream Thermal & Visible Camera Inspector (`dual-view.py`).

Combines thermal imaging (P1/P3) with a visible webcam stream, providing real-time 
8-bit iso-contour edge mapping for narrow temperature bands, lock-in analysis, 
timestamped 15-point vertical column CSV logging, and offline dataset playback/MP4 export.

Usage:
  # Live Streaming Mode (automatically reads config.json if present)
  python3 dual-view.py
  python3 dual-view.py --model p1 --min 50 --max 90 --iso-step 5.0 --high-res

  # Replay Saved Binary Dataset (defaults to high-res $640x480$ readable display)
  python3 dual-view.py --play thermal_rec_20261026_154556.bin --export-mp4 combined.mp4

Interactive Controls:
  q         - Quit application
  b         - Toggle live Binary Thermal (.bin) + Visible Video (.mp4) Auto-Sync Recording
  v         - Toggle live Composite UI MP4 Video Recording
  l / o     - Toggle live 15-point Column Temperature CSV Logging
  w         - Cycle Alignment Modes (OFF <-> RAW OVERLAY)
  c         - Cycle Colormaps (Rainbow, Ironbow, White Hot, etc.)
  r         - Rotate display 90°
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

import argparse
import json
import os
import logging
import time
import threading

import cv2
import numpy as np
from numpy.typing import NDArray

from p3_camera import (
    GainMode,
    Model,
    P3Camera,
    get_model_config,
    raw_to_celsius,
    raw_to_celsius_corrected,
)

# Suppress verbose OpenCV/libjpeg C-level warnings
os.environ["OPENCV_LOG_LEVEL"] = "OFF"


class ThreadedWebcam:
    """Non-blocking background frame fetcher with V4L2 buffer overflow protection."""

    def __init__(self, src: int | str | None, width: int = 1280, height: int = 720):
        self.cap = None
        self.running = False
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()

        if src is None:
            return

        cam_arg = int(src) if (isinstance(src, str) and str(src).isdigit()) else src

        self.cap = cv2.VideoCapture(cam_arg, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_arg)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 1. Force MJPEG FOURCC matching your camera's active hardware stream
            mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)

            # 2. Negotiate 1280x720 @ 30 FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 3. Sensor warm-up loop: Flush initial blank/black V4L2 buffer frames
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.ret, self.frame = ret, frame

            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print(f"Threaded webcam initialized on device {src} ({width}x{height} MJPG)")
        else:
            print(f"Warning: ThreadedWebcam could not open camera {src}")

    def _update(self) -> None:
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.ret, self.frame = ret, frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

def celsius_to_fahrenheit(c: float | NDArray[np.floating]) -> float | NDArray[np.float32]:
    return np.float32(c) * 1.8 + 32.0


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) / 1.8


class ColormapID(IntEnum):
    WHITE_HOT = 0
    BLACK_HOT = 1
    RAINBOW = 2
    IRONBOW = 3
    MILITARY = 4
    SEPIA = 5


class ScaleMode(IntEnum):
    OFF = 0
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3
    LANCZOS = 4


class CVLineType(IntEnum):
    FILLED = cv2.FILLED
    LINE_4 = cv2.LINE_4
    LINE_8 = cv2.LINE_8
    LINE_AA = cv2.LINE_AA


COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}
COLOR_RETICULE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)


def _cv_lut(colormap: int) -> NDArray[np.uint8]:
    gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colored = cv2.applyColorMap(gray, colormap)
    return cast(NDArray[np.uint8], colored.reshape(256, 3))


def _init_colormaps() -> None:
    global COLORMAPS
    ramp = np.arange(256, dtype=np.uint8)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = ramp
    COLORMAPS[ColormapID.WHITE_HOT] = lut

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = 255 - ramp
    COLORMAPS[ColormapID.BLACK_HOT] = lut

    COLORMAPS[ColormapID.RAINBOW] = _cv_lut(cv2.COLORMAP_JET)
    COLORMAPS[ColormapID.IRONBOW] = _cv_lut(cv2.COLORMAP_INFERNO)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.2).astype(np.uint8)
    lut[:, 1] = ramp
    lut[:, 2] = (ramp * 0.3).astype(np.uint8)
    COLORMAPS[ColormapID.MILITARY] = lut

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.4).astype(np.uint8)
    lut[:, 1] = (ramp * 0.7).astype(np.uint8)
    lut[:, 2] = ramp
    COLORMAPS[ColormapID.SEPIA] = lut


_init_colormaps()


def get_colormap(cmap_id: ColormapID | int) -> NDArray[np.uint8]:
    return COLORMAPS[ColormapID(cmap_id)]


def apply_colormap(img_u8: NDArray[np.uint8], cmap_id: ColormapID | int) -> NDArray[np.uint8]:
    lut = get_colormap(cmap_id)
    return lut[img_u8]


def draw_reticule_and_points(img: NDArray[np.uint8], scale_factor: float = 2.0) -> NDArray[np.uint8]:
    """Draw center crosshair and 15 vertical column CSV sampling points on thermal image."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    sf = scale_factor
    r_len = int(10 * sf)

    # 1. Center crosshair lines (Green)
    cv2.line(img, (cx - r_len, cy), (cx + r_len, cy), COLOR_RETICULE, max(1, int(sf)), cv2.LINE_AA)
    cv2.line(img, (cx, cy - r_len), (cx, cy + r_len), COLOR_RETICULE, max(1, int(sf)), cv2.LINE_AA)

    # 2. 15 Vertical column points (P1..P15 with P8 center point enlarged)
    pt_mask = np.zeros((h, w), dtype=np.uint8)
    step_y = 7.5 * sf  # 15px step for sf=2.0 (maps to 5 raw thermal pixels)

    for i in range(1, 16):
        offset = (i - 8) * step_y
        py = int(round(np.clip(cy + offset, 0, h - 1)))
        px = cx
        radius = max(3, int(sf) + 1) if i == 8 else max(2, int(sf))
        cv2.circle(pt_mask, (px, py), radius, 255, -1)

    # High-contrast pixel inversion
    img[pt_mask > 0] = 255 - img[pt_mask > 0]
    return img


def render_colorbar_panel(
    height: int,
    width: int,
    cmap_idx: int,
    cbar_min: float,
    cbar_max: float,
    unit_str: str,
    scale_factor: float = 1.0,
) -> NDArray[np.uint8]:
    """Render vertical colormap gradient bar and min/mid/max temperature ticks on the right margin."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    sf = scale_factor
    bar_w = int(14 * sf)
    bar_x = int(8 * sf)

    # 1. Generate active colormap gradient ramp (254 high temp -> 1 low temp)
    ramp = np.linspace(254, 1, height, dtype=np.uint8).reshape(height, 1)
    bar_bgr = apply_colormap(ramp, cmap_idx)

    panel[:, bar_x : bar_x + bar_w] = bar_bgr
    cv2.rectangle(panel, (bar_x, 0), (bar_x + bar_w - 1, height - 1), (150, 150, 150), 1)

    # 2. Temperature labels next to the colorbar
    text_x = bar_x + bar_w + int(6 * sf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38 * sf
    color = COLOR_TEXT
    thickness = max(1, int(sf))

    # Top label (max threshold)
    cv2.putText(panel, f"{cbar_max:.0f}{unit_str}", (text_x, int(15 * sf)), font, font_scale, color, thickness, cv2.LINE_AA)

    # Mid label (median threshold)
    cbar_mid = (cbar_min + cbar_max) / 2.0
    cv2.putText(panel, f"{cbar_mid:.0f}{unit_str}", (text_x, height // 2 + int(4 * sf)), font, font_scale, color, thickness, cv2.LINE_AA)

    # Bottom label (min threshold)
    cv2.putText(panel, f"{cbar_min:.0f}{unit_str}", (text_x, height - int(6 * sf)), font, font_scale, color, thickness, cv2.LINE_AA)

    return panel


def tnr(img: NDArray[np.uint16], prev_img: NDArray[np.uint16] | None, alpha: float = 0.3) -> NDArray[np.uint16]:
    if prev_img is None:
        return img
    result = alpha * img.astype(np.float32) + (1 - alpha) * prev_img.astype(np.float32)
    return result.astype(np.uint16)


class P3Viewer:
    """P3/P1 Dual-Stream Thermal Inspector & Offline Replay Engine."""

    def __init__(
        self,
        model: Model | str = Model.P3,
        cbar_min: float = 50.0,
        cbar_max: float = 90.0,
        iso_step: float = 5.0,
        use_celsius: bool = False,
        visible_cam: str | int | None = None,
        high_res: bool = False,
        logfile: str | None = None,
        bin_file: str | None = None,
    ) -> None:
        config = get_model_config(model)
        self.camera = P3Camera(config=config)
        self.model = config.model
        self.cbar_min = cbar_min
        self.cbar_max = cbar_max
        self.iso_step = iso_step
        self.use_celsius = use_celsius
        self.unit_str = "C" if use_celsius else "F"
        self.visible_cam_id = visible_cam

        self.high_res = high_res
        self.target_w = 640 if high_res else 320
        self.target_h = 480 if high_res else 240
        self.scale_factor = 2.0 if high_res else 1.0

        sf = self.scale_factor
        self.top_bar_h = int(36 * sf)
        self.bottom_bar_h = int(28 * sf)
        self.right_bar_w = int(85 * sf)

        # 15-Point CSV Logging setup
        self.logfile = logfile
        self.logging_enabled: bool = logfile is not None
        self._spot_temp_buffer: list[list[float]] = []
        self.active_logfile = self._generate_timestamped_filename(self.logfile, ".csv") if self.logging_enabled else None
        if self.logging_enabled and self.active_logfile:
            self._init_logfile(self.active_logfile)

        # Binary 8-bit thermal stream + synchronized visible video writer setup
        self.bin_file_base = bin_file
        self.recording_binary: bool = bin_file is not None
        self.bin_handle = None
        self.vis_video_writer: cv2.VideoWriter | None = None
        self.active_binfile: str | None = None
        self.active_vis_file: str | None = None

        if self.recording_binary:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            self.active_binfile = f"thermal_rec_{ts_str}.bin"
            self.active_vis_file = f"visible_rec_{ts_str}.mp4"
            self.bin_handle = open(self.active_binfile, "wb")
            print(f"Binary Thermal Recording STARTED -> {self.active_binfile}")

        # Live Composite MP4 Video Recording setup
        self.recording_video: bool = False
        self.video_writer: cv2.VideoWriter | None = None
        self.video_filename: str | None = None
        self._video_dim: tuple[int, int] | None = None

        self.rotation: int = 90
        self.colormap_idx: int = ColormapID.RAINBOW
        self.show_reticule: bool = True
        self.show_colorbar: bool = True
        self.fps: float = 0.0
        self.scale_mode: ScaleMode = ScaleMode.BICUBIC
        self.cv_linetype: CVLineType = CVLineType.LINE_AA
        self.tnr_alpha: float = 0.5
        self._fps_count: int = 0
        self._fps_time: float = time.time()
        self._last_display: NDArray[np.uint8] | None = None
        self._prev_frame: NDArray[np.uint16] | None = None
        self._current_frame_stats: dict = {}

        # Align mode: 0 = OFF, 1 = RAW OVERLAY
        self.align_mode: int = 0
        self.cap_visible = ThreadedWebcam(self.visible_cam_id)

    def _generate_timestamped_filename(self, filename: str | None = None, default_ext: str = ".csv") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        if filename:
            base, ext = os.path.splitext(filename)
            ext = ext if ext else default_ext
            return f"{base}_{ts}{ext}"
        prefix = "column_log" if default_ext == ".csv" else "thermal_rec"
        return f"{prefix}_{ts}{default_ext}"

    def _init_logfile(self, filename: str) -> None:
        try:
            with open(filename, "a") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# Vertical Column Thermal Log initialized {ts}\n")
                p_names = ["P8_Center"] + [f"P{i}" for i in range(1, 16)]
                headers = ["Timestamp"] + [f"Mean_{p}" for p in p_names] + [f"Med_{p}" for p in p_names] + [f"Min_{p}" for p in p_names] + [f"Max_{p}" for p in p_names] + ["Unit"]
                f.write(",".join(headers) + "\n")
            print(f"Column temperature logging enabled -> {filename}")
        except Exception as e:
            print(f"Warning: Could not open log file {filename}: {e}")

    def _get_column_samples(self, temp_disp: NDArray[np.float32]) -> list[float]:
        th, tw = temp_disp.shape[:2]
        cx, cy = tw // 2, th // 2
        samples = []
        for i in range(1, 16):
            offset = (i - 8) * 5
            if self.rotation in (90, 270):
                direction = 1 if self.rotation == 90 else -1
                x_raw = np.clip(cx + offset * direction, 0, tw - 1)
                y_raw = cy
            else:
                direction = 1 if self.rotation == 0 else -1
                x_raw = cx
                y_raw = np.clip(cy + offset * direction, 0, th - 1)
            samples.append(float(temp_disp[y_raw, x_raw]))
        return [samples[7]] + samples

    def _log_column_temps(self, temp_disp: NDArray[np.float32]) -> None:
        if not self.logging_enabled or not getattr(self, "active_logfile", None):
            return

        self._spot_temp_buffer.append(self._get_column_samples(temp_disp))

        if len(self._spot_temp_buffer) >= 25:
            arr = np.array(self._spot_temp_buffer, dtype=np.float32)
            all_stats = np.concatenate([np.mean(arr, axis=0), np.median(arr, axis=0), np.min(arr, axis=0), np.max(arr, axis=0)])
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            vals_str = ",".join(f"{v:.1f}" for v in all_stats)
            try:
                with open(self.active_logfile, "a") as f:
                    f.write(f"{ts},{vals_str},{self.unit_str}\n")
            except Exception as e:
                print(f"Error writing to log file {self.active_logfile}: {e}")
            self._spot_temp_buffer.clear()

    def encode_8bit_iso_frame(self, temp_disp: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Encode 50-90°F range linearly (2..253) and map out-of-range temperatures to distinct iso-bands."""
        t_min, t_max, step = self.cbar_min, self.cbar_max, max(0.5, self.iso_step)
        clamped = np.clip(temp_disp, t_min, t_max)
        out = (2 + (clamped - t_min) / (t_max - t_min) * 251.0).astype(np.uint8)

        # Cold Isotherm: alternate index 0 (Blue) and 1 (White)
        mask_cold = temp_disp < t_min
        if np.any(mask_cold):
            cold_band = np.floor((t_min - temp_disp[mask_cold]) / step).astype(np.int32) % 2
            out[mask_cold] = cold_band.astype(np.uint8)

        # Hot Isotherm: alternate index 254 (Red) and 255 (White)
        mask_hot = temp_disp > t_max
        if np.any(mask_hot):
            hot_band = np.floor((temp_disp[mask_hot] - t_max) / step).astype(np.int32) % 2
            out[mask_hot] = (254 + hot_band).astype(np.uint8)

        return out

    def _toggle_spot_logging(self) -> None:
        self.logging_enabled = not self.logging_enabled
        if self.logging_enabled:
            base = self.logfile if self.logfile else "column_log.csv"
            self.active_logfile = self._generate_timestamped_filename(base, ".csv")
            self._spot_temp_buffer.clear()
            self._init_logfile(self.active_logfile)
            print(f"CSV Logging: STARTED -> {self.active_logfile}")
        else:
            print("CSV Logging: STOPPED")

    def _toggle_binary_recording(self) -> None:
        """Toggle simultaneous 8-bit thermal binary + raw visible video recording."""
        self.recording_binary = not self.recording_binary
        if self.recording_binary:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            self.active_binfile = f"thermal_rec_{ts_str}.bin"
            self.active_vis_file = f"visible_rec_{ts_str}.mp4"
            self.bin_handle = open(self.active_binfile, "wb")
            self.vis_video_writer = None
            print(f"Binary Mode: STARTED -> {self.active_binfile} & {self.active_vis_file}")
        else:
            if self.bin_handle:
                self.bin_handle.close()
                self.bin_handle = None
            if self.vis_video_writer is not None:
                self.vis_video_writer.release()
                self.vis_video_writer = None
            print("Binary Mode: STOPPED")

    def _toggle_video_recording(self) -> None:
        """Toggle live dual-view composite video recording."""
        self.recording_video = not self.recording_video
        if self.recording_video:
            self.video_writer = None
            self._video_dim = None
            print("Composite Video Recording: STARTED")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self._video_dim = None
            print(f"Composite Video Recording: STOPPED -> {self.video_filename}")

    def run(self) -> None:
        model_name = self.model.value.upper()
        window_name = f"{model_name} Thermal Inspector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        dual_active = self.cap_visible is not None and self.cap_visible.isOpened()
        total_w = (self.target_w + self.right_bar_w) if not dual_active else (self.target_w * 2 + self.right_bar_w)
        total_h = self.target_h + self.top_bar_h + self.bottom_bar_h
        cv2.resizeWindow(window_name, total_w, total_h)

        self.camera.connect()
        self.camera.init()
        self.camera.start_streaming()

        try:
            while True:
                _, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue

                thermal = tnr(thermal, self._prev_frame, alpha=self.tnr_alpha)
                self._prev_frame = thermal.copy()

                rendered_thermal = self._render(thermal)
                final_display, record_frame = self._process_dual_camera(rendered_thermal)
                self._last_display = final_display

                if self.recording_video:
                    self._record_frame_to_video(record_frame)

                cv2.imshow(window_name, final_display)
                self._update_fps()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord("v"):
                    self._toggle_video_recording()
                elif key == ord("b"):
                    self._toggle_binary_recording()
                elif key in (ord("l"), ord("L"), ord("o")):
                    self._toggle_spot_logging()
                elif key == ord("w"):
                    self.align_mode = (self.align_mode + 1) % 2
                elif key == ord("c"):
                    self.colormap_idx = (self.colormap_idx + 1) % len(ColormapID)
                elif key == ord("r"):
                    self.rotation = (self.rotation + 90) % 360
        finally:
            if self.bin_handle:
                self.bin_handle.close()
            if self.vis_video_writer is not None:
                self.vis_video_writer.release()
            if self.recording_video and self.video_writer is not None:
                self.video_writer.release()
            if self.cap_visible is not None:
                self.cap_visible.release()
            self.camera.stop_streaming()
            cv2.destroyAllWindows()

    def _process_dual_camera(self, thermal_canvas: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        if self.cap_visible is not None and self.cap_visible.isOpened():
            ret, vis_frame = self.cap_visible.read()
            canvas_h = thermal_canvas.shape[0]
            thermal_h = max(1, canvas_h - self.top_bar_h - self.bottom_bar_h)
            thermal_w = self.target_w

            if ret and vis_frame is not None:
                thermal_color_crop = thermal_canvas[self.top_bar_h : self.top_bar_h + thermal_h, 0:thermal_w]
                vis_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR) if vis_frame.ndim == 2 else vis_frame.copy()
                vis_resized = cv2.resize(vis_bgr, (thermal_w, thermal_h), interpolation=cv2.INTER_AREA)

                if self.recording_binary:
                    if self.vis_video_writer is None and self.active_vis_file:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        vh, vw = vis_resized.shape[:2]
                        self.vis_video_writer = cv2.VideoWriter(self.active_vis_file, fourcc, 25.0, (vw, vh))
                        print(f"Visible Sync Video Recording STARTED -> {self.active_vis_file} ({vw}x{vh})")
                    if self.vis_video_writer is not None:
                        self.vis_video_writer.write(vis_resized)

                if self.align_mode == 1:
                    vis_bgr = cv2.addWeighted(vis_resized, 0.5, thermal_color_crop, 0.5, 0)
                else:
                    vis_bgr = vis_resized

                mode_label = "Align: RAW OVERLAY" if self.align_mode == 1 else "Align: OFF"
                cv2.putText(
                    vis_bgr, mode_label, (10, int(25 * self.scale_factor)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * self.scale_factor, (0, 255, 255) if self.align_mode == 1 else (180, 180, 180),
                    max(1, int(self.scale_factor)), self.cv_linetype
                )
            else:
                vis_bgr = np.zeros((thermal_h, thermal_w, 3), dtype=np.uint8)

            vis_padded = np.zeros((canvas_h, thermal_w, 3), dtype=np.uint8)
            vis_padded[self.top_bar_h : self.top_bar_h + thermal_h, 0:thermal_w] = vis_bgr
            composite = np.hstack([vis_padded, thermal_canvas])
            return composite, composite

        return thermal_canvas, thermal_canvas

    def _render(self, thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
        env = self.camera.env_params
        temp_c = raw_to_celsius_corrected(thermal, env)
        temp_disp = temp_c.copy() if self.use_celsius else celsius_to_fahrenheit(temp_c)

        cy, cx = thermal.shape[0] // 2, thermal.shape[1] // 2
        frame_stats = {
            "tspot": float(temp_disp[cy, cx]),
            "tmin": float(np.min(temp_disp)),
            "tmax": float(np.max(temp_disp)),
        }
        self._current_frame_stats = frame_stats

        self._log_column_temps(temp_disp)

        # 8-bit Iso-contour frame encoding
        u8_iso = self.encode_8bit_iso_frame(temp_disp)

        # Apply rotation and resolution scaling to match target display / visible stream
        if self.rotation == 90:
            u8_iso = cv2.rotate(u8_iso, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            u8_iso = cv2.rotate(u8_iso, cv2.ROTATE_180)
        elif self.rotation == 270:
            u8_iso = cv2.rotate(u8_iso, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if (u8_iso.shape[1], u8_iso.shape[0]) != (self.target_w, self.target_h):
            u8_iso = cv2.resize(u8_iso, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        # Record pre-rotated 1-to-1 dimension aligned frame
        if self.recording_binary and self.bin_handle:
            self.bin_handle.write(u8_iso.tobytes())

        img = apply_colormap(u8_iso, self.colormap_idx)

        # Apply standard colormap to in-range temperatures (indices 2..253)
        img = apply_colormap(u8_iso, self.colormap_idx)

        # Cold Isotherm: Alternating Blue and White
        img[u8_iso == 0] = (255, 0, 0)     # Blue
        img[u8_iso == 1] = (255, 255, 255) # White

        # Hot Isotherm: Alternating Red and White
        img[u8_iso == 254] = (0, 0, 255)   # Red
        img[u8_iso == 255] = (0, 0, 0) # black

        if self.show_reticule:
            img = draw_reticule_and_points(img, scale_factor=self.scale_factor)

        return self._build_ui_canvas(img, frame_stats)

    def _build_ui_canvas(self, thermal_img: NDArray[np.uint8], frame_stats: dict) -> NDArray[np.uint8]:
        th, tw = thermal_img.shape[:2]
        sf = self.scale_factor
        cw = tw + (self.right_bar_w if self.show_colorbar else 0)
        ch = th + self.top_bar_h + self.bottom_bar_h

        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[self.top_bar_h : self.top_bar_h + th, 0:tw] = thermal_img

        # Render vertical colorbar gradient panel on the right margin
        if self.show_colorbar:
            cbar_panel = render_colorbar_panel(
                height=th,
                width=self.right_bar_w,
                cmap_idx=self.colormap_idx,
                cbar_min=self.cbar_min,
                cbar_max=self.cbar_max,
                unit_str=self.unit_str,
                scale_factor=sf,
            )
            canvas[self.top_bar_h : self.top_bar_h + th, tw:cw] = cbar_panel

        rec_status = " [REC(v)]" if self.recording_video else ""
        bin_status = " [BIN(b)]" if self.recording_binary else ""
        log_status = " [LOG(l)]" if self.logging_enabled else ""

        top_status = f"Min: {frame_stats['tmin']:.1f}{self.unit_str} | Max: {frame_stats['tmax']:.1f}{self.unit_str} | Iso: {self.iso_step:.1f}{self.unit_str}{rec_status}{bin_status}{log_status}"
        cv2.putText(canvas, top_status, (int(8 * sf), int(18 * sf)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype)

        # P8 Center temperature leading the bottom status bar
        status = f"P8: {frame_stats['tspot']:.1f}{self.unit_str} | {self.fps:.1f} FPS | {ColormapID(self.colormap_idx).name} | Range: [{self.cbar_min:.0f}-{self.cbar_max:.0f}{self.unit_str}]"
        cv2.putText(canvas, status, (int(8 * sf), ch - int(10 * sf)), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype)

        return canvas

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self._fps_count / (now - self._fps_time)
            self._fps_count, self._fps_time = 0, now

    def _record_frame_to_video(self, frame: NDArray[np.uint8]) -> None:
        vh, vw = frame.shape[:2]
        if self.video_writer is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"p3_video_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 25.0, (vw, vh))
            self._video_dim = (vw, vh)
            print(f"MP4 Composite Video Recording STARTED -> {self.video_filename} ({vw}x{vh})")

        if self._video_dim is not None and (vw, vh) != self._video_dim:
            frame = cv2.resize(frame, self._video_dim, interpolation=cv2.INTER_AREA)

        self.video_writer.write(frame)


def run_playback(
    bin_path: str,
    vis_path: str | None = None,
    export_mp4: str | None = None,
    fps: float = 25.0,
    cbar_min: float = 50.0,
    cbar_max: float = 90.0,
    use_celsius: bool = False,
    high_res: bool = True,
) -> None:
    """Memory-mapped dataset player for 8-bit binary thermal recordings and visible video."""
    if not os.path.exists(bin_path):
        print(f"Error: Thermal binary file not found: {bin_path}")
        return

    unit_str = "C" if use_celsius else "F"
    sf = 2.0 if high_res else 1.0

    target_w = 640 if high_res else 320
    target_h = 480 if high_res else 240
    top_bar_h = int(36 * sf)
    bottom_bar_h = int(28 * sf)
    right_bar_w = int(85 * sf)

    # Auto-derive matching visible video filename if not explicitly provided
    if vis_path is None:
        dirname, basename = os.path.split(bin_path)
        if "thermal_rec_" in basename:
            candidate = os.path.join(dirname, basename.replace("thermal_rec_", "visible_rec_").replace(".bin", ".mp4"))
            if os.path.exists(candidate):
                vis_path = candidate
                print(f"Auto-detected matching visible video: {vis_path}")
        elif ".bin" in basename:
            candidate = os.path.join(dirname, os.path.splitext(basename)[0] + ".mp4")
            if os.path.exists(candidate):
                vis_path = candidate
                print(f"Auto-detected matching visible video: {vis_path}")

    raw_data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    total_bytes = len(raw_data)

    if total_bytes == 0:
        print(f"Error: Thermal binary file is empty: {bin_path}")
        return

    cap_vis = cv2.VideoCapture(vis_path) if (vis_path and os.path.exists(vis_path)) else None

    # Detect raw thermal frame dimensions from dataset size / visible video frame count
    h_raw, w_raw = 240, 320
    if cap_vis is not None:
        total_vis_frames = int(cap_vis.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_vis_frames > 0:
            calc_frame_bytes = total_bytes // total_vis_frames
            if calc_frame_bytes in (19200, 120 * 160):
                h_raw, w_raw = 120, 160
            elif calc_frame_bytes in (49152, 192 * 256):
                h_raw, w_raw = 192, 256
            elif calc_frame_bytes in (76800, 240 * 320):
                h_raw, w_raw = 240, 320
            elif calc_frame_bytes in (307200, 480 * 640):
                h_raw, w_raw = 480, 640
    else:
        for hr, wr in [(480, 640), (240, 320), (192, 256), (120, 160)]:
            if total_bytes % (hr * wr) == 0:
                h_raw, w_raw = hr, wr
                break

    frame_bytes = h_raw * w_raw
    num_frames = total_bytes // frame_bytes
    if num_frames == 0:
        print(f"Error: No complete thermal frames found in {bin_path}")
        return

    frames = raw_data[: num_frames * frame_bytes].reshape(num_frames, h_raw, w_raw)

    print(f"Loaded {num_frames} frames ({w_raw}x{h_raw}) from {bin_path} (High-Res UI: {high_res})")

    window_name = "Dual Viewer - Playback Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_window_w = (target_w + right_bar_w) if cap_vis is None else (target_w * 2 + right_bar_w)
    total_window_h = target_h + top_bar_h + bottom_bar_h
    cv2.resizeWindow(window_name, total_window_w, total_window_h)

    writer = None
    cmap_idx = ColormapID.IRONBOW
    paused = False
    idx = 0
    delay_ms = int(1000 / fps)

    try:
        while 0 <= idx < num_frames:
            u8_frame = frames[idx]

            # Estimate P8 spot temperature from updated 8-bit index ranges
            p8_val = u8_frame[h_raw // 2, w_raw // 2]
            if 2 <= p8_val <= 253:
                p8_temp = cbar_min + ((p8_val - 2) / 251.0) * (cbar_max - cbar_min)
                p8_str = f"P8: {p8_temp:.1f}{unit_str}"
            elif p8_val in (0, 1):
                p8_str = f"P8: <{cbar_min:.0f}{unit_str}"
            else:
                p8_str = f"P8: >{cbar_max:.0f}{unit_str}"

            color_th = apply_colormap(u8_frame, cmap_idx)
            color_th[u8_frame == 0] = (255, 0, 0)     # Blue
            color_th[u8_frame == 1] = (255, 255, 255) # White
            color_th[u8_frame == 254] = (0, 0, 255)   # Red
            color_th[u8_frame == 255] = (0, 0, 0) # black
            
            color_th = cv2.resize(color_th, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            color_th = draw_reticule_and_points(color_th, scale_factor=sf)

            # Build thermal section with right-side colorbar panel
            cbar_panel = render_colorbar_panel(
                height=target_h,
                width=right_bar_w,
                cmap_idx=cmap_idx,
                cbar_min=cbar_min,
                cbar_max=cbar_max,
                unit_str=unit_str,
                scale_factor=sf,
            )
            thermal_section = np.hstack([color_th, cbar_panel])

            if cap_vis is not None:
                curr_pos = int(cap_vis.get(cv2.CAP_PROP_POS_FRAMES))
                if curr_pos != idx:
                    cap_vis.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, vis_frame = cap_vis.read()
                if ret and vis_frame is not None:
                    vis_resized = cv2.resize(vis_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    video_composite = np.hstack([vis_resized, thermal_section])
                else:
                    video_composite = thermal_section
            else:
                video_composite = thermal_section

            # Wrap in top and bottom UI bar canvas for clean readable text
            vw_total, vh_total = video_composite.shape[1], video_composite.shape[0]
            canvas_h = vh_total + top_bar_h + bottom_bar_h
            canvas = np.zeros((canvas_h, vw_total, 3), dtype=np.uint8)
            canvas[top_bar_h : top_bar_h + vh_total, 0:vw_total] = video_composite

            # Render crisp high-resolution UI top/bottom status text
            status_top = f"Replay: {os.path.basename(bin_path)} | Frame: {idx + 1}/{num_frames} | {'[PAUSED]' if paused else '[PLAYING]'}"
            cv2.putText(canvas, status_top, (int(8 * sf), int(18 * sf)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sf, COLOR_TEXT, max(1, int(sf)), cv2.LINE_AA)

            status_bottom = f"{p8_str} | Colormap: {ColormapID(cmap_idx).name} | Range: [{cbar_min:.0f}-{cbar_max:.0f}{unit_str}]"
            cv2.putText(canvas, status_bottom, (int(8 * sf), canvas_h - int(10 * sf)), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * sf, COLOR_TEXT, max(1, int(sf)), cv2.LINE_AA)

            if export_mp4:
                if writer is None:
                    vh, vw = canvas.shape[:2]
                    writer = cv2.VideoWriter(export_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
                    print(f"Exporting video playback to: {export_mp4} ({vw}x{vh})")
                writer.write(canvas)

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(ColormapID)
            elif key == ord("d"):
                idx = (idx + 1) % num_frames
            elif key == ord("a"):
                idx = (idx - 1) % num_frames

            if not paused:
                idx += 1
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved MP4 export -> {export_mp4}")
        if cap_vis is not None:
            cap_vis.release()
        cv2.destroyAllWindows()


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load settings dictionary from JSON configuration file and normalize keys."""
    alias_map = {
        "webcam": "visible_cam",
        "visible_cam": "visible_cam",
        "visible-cam": "visible_cam",
        "min": "cbar_min",
        "cbar_min": "cbar_min",
        "cbar-min": "cbar_min",
        "temp_min": "cbar_min",
        "temp-min": "cbar_min",
        "max": "cbar_max",
        "cbar_max": "cbar_max",
        "cbar-max": "cbar_max",
        "temp_max": "cbar_max",
        "temp-max": "cbar_max",
        "iso_step": "iso_step",
        "iso-step": "iso_step",
        "high_res": "high_res",
        "high-res": "high_res",
        "celsius": "celsius",
        "model": "model",
        "logfile": "logfile",
        "bin_file": "bin_file",
        "bin-file": "bin_file",
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                normalized = {}
                for k, v in data.items():
                    clean_k = k.lower().strip()
                    target_key = alias_map.get(clean_k, clean_k.replace("-", "_"))
                    normalized[target_key] = v
                print(f"Loaded configuration from: {config_path}")
                return normalized
    except Exception as e:
        print(f"Error reading config file {config_path}: {e}")
    return {}


def main() -> None:
    # Pass 1: Parse --config or fall back to default 'config.json' if present
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "---config", "-c", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_path = pre_args.config
    if not config_path and os.path.exists("config.json"):
        config_path = "config.json"

    config_defaults = {}
    if config_path:
        config_defaults = load_config_file(config_path)

    # Pass 2: Main CLI argument parser
    parser = argparse.ArgumentParser(description="Thermal Master P1/P3 USB Thermal Camera Viewer & Replay Engine")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["p1", "p3"], default="p3", help="Camera model (p1 or p3)")
    parser.add_argument("--min", "--cbar-min", dest="cbar_min", type=float, default=50.0, help="Lower temperature threshold")
    parser.add_argument("--max", "--cbar-max", dest="cbar_max", type=float, default=90.0, help="Upper temperature threshold")
    parser.add_argument("--iso-step", type=float, default=5.0, help="Iso-contour step size in degrees outside bounds")
    parser.add_argument("--celsius", action="store_true", default=False)
    parser.add_argument("--visible-cam", "--webcam", type=str, default=None)
    parser.add_argument("--high-res", action="store_true", default=True, help="Display canvas in high resolution (640x480)")
    parser.add_argument("--low-res", action="store_false", dest="high_res", help="Display canvas in low resolution (320x240)")
    parser.add_argument("--logfile", type=str, default=None, help="Path to output CSV logfile")
    parser.add_argument("--bin-file", type=str, default=None, help="Path to output 8-bit binary thermal dump (.bin)")

    # Playback arguments
    parser.add_argument("--play", type=str, default=None, help="Path to .bin thermal recording to replay")
    parser.add_argument("--play-vis", type=str, default=None, help="Path to visible .mp4 video to sync with thermal playback")
    parser.add_argument("--export-mp4", type=str, default=None, help="Path to save playback composition as MP4 video")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if args.play:
        run_playback(
            bin_path=args.play,
            vis_path=args.play_vis,
            export_mp4=args.export_mp4,
            cbar_min=args.cbar_min,
            cbar_max=args.cbar_max,
            use_celsius=args.celsius,
            high_res=args.high_res,
        )
    else:
        try:
            P3Viewer(
                model=args.model,
                cbar_min=args.cbar_min,
                cbar_max=args.cbar_max,
                iso_step=args.iso_step,
                use_celsius=args.celsius,
                visible_cam=args.visible_cam,
                high_res=args.high_res,
                logfile=args.logfile,
                bin_file=args.bin_file,
            ).run()
        except KeyboardInterrupt:
            print("\nInterrupted")


if __name__ == "__main__":
    main()
