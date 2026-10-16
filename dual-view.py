#!/usr/bin/env python3
"""P3/P1 Thermal Camera Viewer.

Display, colormaps, and image processing (ISP) for the P3/P1 thermal camera.

Controls:
  q - Quit           +/- - Zoom
  r - Rotate 90°     c - Colormap
  s - Shutter/NUC    g - Gain mode
  m - Mirror         h - Help
  space - Screenshot D - Dump raw data
  e - Emissivity     1-9 - Set emissivity (0.1-0.9)
  x - Scale mode     p - Enhanced (CLAHE+DDE)
  a - AGC mode       t - Toggle reticule
  d - Toggle DDE     b - Toggle min/max marker
  f - Toggle Fixed Range Lock
  v - Toggle Video Recording
  L / o - Toggle Spot Temp Logging
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from lockin import LockInController

from p3_camera import (
    GainMode,
    Model,
    P3Camera,
    get_model_config,
    raw_to_celsius,
    raw_to_celsius_corrected,
)


# =============================================================================
# Helper Functions & Configuration Loader
# =============================================================================

def celsius_to_fahrenheit(c: float | NDArray[np.floating]) -> float | NDArray[np.float32]:
    """Convert Celsius to Fahrenheit."""
    return np.float32(c) * 1.8 + 32.0


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) / 1.8




# =============================================================================
# Colormaps & Enums
# =============================================================================

class ColormapID(IntEnum):
    """Colormap IDs."""
    WHITE_HOT = 0
    BLACK_HOT = 1
    RAINBOW = 2
    IRONBOW = 3
    MILITARY = 4
    SEPIA = 5


class ScaleMode(IntEnum):
    """2x upscaling interpolation modes."""
    OFF = 0       # No upscaling
    NEAREST = 1   # Nearest neighbor
    BILINEAR = 2  # Bilinear
    BICUBIC = 3   # Bicubic
    LANCZOS = 4   # Lanczos


class CVLineType(IntEnum):
    """Draw style for cv2 lines and text."""
    FILLED = cv2.FILLED
    LINE_4 = cv2.LINE_4
    LINE_8 = cv2.LINE_8
    LINE_AA = cv2.LINE_AA


class AGCMode(IntEnum):
    """Auto Gain Control modes."""
    FACTORY = 0     # Hardware AGC / IR brightness
    TEMPORAL_1 = 1  # EMA smoothed dynamic AGC
    FIXED_RANGE = 2 # Fixed temperature range


AGC_PERCENTILES = {
    AGCMode.TEMPORAL_1: 1.0,
}


SCALE_INTERP = {
    ScaleMode.NEAREST: cv2.INTER_NEAREST,
    ScaleMode.BILINEAR: cv2.INTER_LINEAR,
    ScaleMode.BICUBIC: cv2.INTER_CUBIC,
    ScaleMode.LANCZOS: cv2.INTER_LANCZOS4,
}


class HotspotMode(IntEnum):
    """Extreme value marker modes."""
    OFF = 0     # Off
    MAX = 1     # Show maximum
    MIN = 2     # Show minimum
    MINMAX = 3  # Show both


COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}

COLOR_SPOT_MAX = (0, 0, 255)
COLOR_SPOT_MIN = (255, 0, 0)
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


def apply_colormap(
    img_u8: NDArray[np.uint8], cmap_id: ColormapID | int
) -> NDArray[np.uint8]:
    lut = get_colormap(cmap_id)
    return lut[img_u8]


def apply_diverging_colormap(data: np.ndarray, cmap_name: str = "bwr") -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    lo = float(np.nanpercentile(data, 0.01))
    hi = float(np.nanpercentile(data, 99.99))
    if hi <= lo:
        hi = lo + 1.0
    norm_data = 2 * (data - lo) / (hi - lo) - 1.0
    data = np.clip(norm_data, -1.0, 1.0)

    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = cm.get_cmap(cmap_name)

    rgba_float = cmap(norm(data))
    rgb_float = rgba_float[..., :3]
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)

    bgr_uint8 = rgb_uint8[..., ::-1]
    return bgr_uint8


# =============================================================================
# Image Processing (ISP)
# =============================================================================

_agc_ema_low: float | None = None
_agc_ema_high: float | None = None


def agc_temporal(
    img: NDArray[np.uint16],
    pct: float = 1.0,
    ema_alpha: float = 0.1,
) -> NDArray[np.uint8]:
    global _agc_ema_low, _agc_ema_high
    img_f = img.astype(np.float32)
    low = float(np.percentile(img_f, pct))
    high = float(np.percentile(img_f, 100.0 - pct))

    if high <= low:
        low = float(np.min(img_f))
        high = float(np.max(img_f))

    if _agc_ema_low is None or _agc_ema_high is None or _agc_ema_high <= _agc_ema_low:
        _agc_ema_low, _agc_ema_high = low, high
    else:
        _agc_ema_low = ema_alpha * low + (1 - ema_alpha) * _agc_ema_low
        _agc_ema_high = ema_alpha * high + (1 - ema_alpha) * _agc_ema_high

    denom = max(1e-5, _agc_ema_high - _agc_ema_low)
    normalized = (img_f - _agc_ema_low) / denom
    return (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def agc_fixed(
    img: NDArray[np.uint16],
    temp_min: float = 18.0,
    temp_max: float = 35.0,
) -> NDArray[np.uint8]:
    raw_min = (temp_min + 273.15) * 64
    raw_max = (temp_max + 273.15) * 64
    if raw_max <= raw_min:
        raw_max = raw_min + 1.0
    normalized = (img.astype(np.float32) - raw_min) / (raw_max - raw_min)
    return (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def dde(
    img_u8: NDArray[np.uint8],
    strength: float = 0.5,
    kernel_size: int = 3,
) -> NDArray[np.uint8]:
    if strength <= 0:
        return img_u8

    ksize = kernel_size | 1
    blurred = cv2.GaussianBlur(img_u8, (ksize, ksize), 0)

    img_f = img_u8.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    enhanced = img_f + strength * (img_f - blurred_f)

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def tnr(
    img: NDArray[np.uint16],
    prev_img: NDArray[np.uint16] | None,
    alpha: float = 0.3,
) -> NDArray[np.uint16]:
    if prev_img is None:
        return img

    result = alpha * img.astype(np.float32) + (1 - alpha) * prev_img.astype(np.float32)
    return result.astype(np.uint16)


# =============================================================================
# Viewer Class
# =============================================================================

class P3Viewer:
    """P3/P1 Thermal Camera Viewer with Padded Interactive UI Layout."""

    def __init__(
        self,
        model: Model | str = Model.P3,
        serial_port: str = "/dev/ttyACM0",
        baud_rate: int = 115200,
        lockin_period: float = 1.0,
        lockin_integration: float = 60.0,
        lockin_invert: bool = False,
        cbar_min: float | None = None,
        cbar_max: float | None = None,
        use_celsius: bool = False,
        visible_cam: str | int | None = None,
        high_res: bool = False,
        logfile: str | None = None,
    ) -> None:
        """Initialize viewer."""
        config = get_model_config(model)
        self.camera = P3Camera(config=config)
        self.model = config.model
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.lockin_period = lockin_period
        self.lockin_integration = lockin_integration
        self.lockin_invert = bool(lockin_invert)
        self.cbar_min = cbar_min
        self.cbar_max = cbar_max
        self._user_cli_bounds: bool = (cbar_min is not None and cbar_max is not None)
        self.fixed_range_enabled: bool = self._user_cli_bounds
        self.use_celsius = use_celsius
        self.unit_str = "C" if use_celsius else "F"
        self.visible_cam_id = visible_cam
        self.cap_visible: cv2.VideoCapture | None = None

        # Resolution & scaling settings
        self.high_res = high_res
        self.target_w = 640 if high_res else 320
        self.target_h = 480 if high_res else 240
        self.scale_factor = 2.0 if high_res else 1.0

        # UI Bar Dimensions
        sf = self.scale_factor
        self.top_bar_h = int(36 * sf)
        self.bottom_bar_h = int(28 * sf)
        self.right_bar_w = int(85 * sf)

        # Logfile setup
        self.logfile = logfile
        self.logging_enabled: bool = logfile is not None
        self._spot_temp_buffer: list[float] = []

        if self.logging_enabled and self.logfile:
            self._init_logfile(self.logfile)

        # Mouse tracking & Interactive UI Button Bounds
        self.mouse_x: int = -1
        self.mouse_y: int = -1
        self.ui_buttons: dict[str, tuple[int, int, int, int]] = {}

        self.rotation: int = 0
        self.colormap_idx: int = ColormapID.RAINBOW
        self.mirror: bool = False
        self.show_help: bool = False
        self.show_reticule: bool = True
        self.show_colorbar: bool = True
        self.hotspot_mode: int = HotspotMode.OFF
        self.zoom: int = 1
        self.fps: float = 0.0
        self.enhanced: bool = True
        self.use_clahe: bool = True
        self.scale_mode: ScaleMode = ScaleMode.BICUBIC
        self.cv_linetype: CVLineType = CVLineType.LINE_AA
        self.agc_mode: AGCMode = AGCMode.FACTORY
        self.dde_strength: float = 0.3
        self.tnr_alpha: float = 0.5
        self._fps_count: int = 0
        self._fps_time: float = time.time()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._last_display: NDArray[np.uint8] | None = None
        self._prev_frame: NDArray[np.uint16] | None = None
        self._ir_brightness: NDArray[np.uint8] | None = None
        self._current_frame_stats: dict = {}

        # Video recording
        self.recording_video: bool = False
        self.video_writer: cv2.VideoWriter | None = None
        self.video_filename: str | None = None
        self._video_dim: tuple[int, int] | None = None

        # Lock-in controller
        self.lockin_controller: LockInController | None = None
        self.lockin_thread: threading.Thread | None = None
        self.lockin_running: bool = False

    def _init_logfile(self, filename: str) -> None:
        """Initialize logfile header."""
        try:
            with open(filename, "a") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"# Thermal Log initialized {ts}\nTimestamp,Min,Max,Median,Unit\n")
            print(f"Spot temperature logging enabled -> {filename}")
        except Exception as e:
            print(f"Warning: Could not open log file {filename}: {e}")

    def _log_spot_temp(self, tspot: float) -> None:
        """Buffer spot temp and write min, max, and median every 25 frames."""
        if not self.logging_enabled or not self.logfile:
            return

        self._spot_temp_buffer.append(tspot)
        if len(self._spot_temp_buffer) >= 25:
            t_min = min(self._spot_temp_buffer)
            t_max = max(self._spot_temp_buffer)
            t_med = float(np.median(self._spot_temp_buffer))
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            log_line = f"{ts},{t_min:.2f},{t_max:.2f},{t_med:.2f},{self.unit_str}\n"
            try:
                with open(self.logfile, "a") as f:
                    f.write(log_line)
            except Exception as e:
                print(f"Error writing to log file {self.logfile}: {e}")

            self._spot_temp_buffer.clear()

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse hover and button click interactions."""
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            for btn_id, (bx1, by1, bx2, by2) in self.ui_buttons.items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if btn_id == "btn_fixed":
                        self._toggle_fixed_lock()
                    elif btn_id == "btn_logging":
                        self._toggle_spot_logging()
                    elif btn_id == "btn_video":
                        self._toggle_video_recording()
                    elif btn_id == "btn_help":
                        self.show_help = not self.show_help
                    break

    def run(self) -> None:
        """Main viewer loop."""
        model_name = self.model.value.upper()
        print(f"{model_name} Thermal Viewer")
        self.camera.connect()
        name, version = self.camera.init()
        print(f"Device: {name}, Firmware: {version}")
        self.camera.start_streaming()

        if self.visible_cam_id is not None:
            cam_idx = (
                int(self.visible_cam_id)
                if str(self.visible_cam_id).isdigit()
                else self.visible_cam_id
            )
            self.cap_visible = cv2.VideoCapture(cam_idx)
            if self.cap_visible.isOpened():
                print(f"Visible camera initialized: {self.visible_cam_id}")
            else:
                print(f"Warning: Could not open visible camera {self.visible_cam_id}")

        print("Press 'h' for help")

        window_name = f"{model_name} Thermal"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)

        dual_active = self.cap_visible is not None and self.cap_visible.isOpened()
        total_w = (self.target_w + self.right_bar_w) if not dual_active else (self.target_w * 2 + self.right_bar_w)
        total_h = self.target_h + self.top_bar_h + self.bottom_bar_h
        cv2.resizeWindow(window_name, total_w, total_h)

        try:
            while True:
                ir_brightness, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue
                self._ir_brightness = ir_brightness

                if self.lockin_controller is not None and self.lockin_running:
                    try:
                        self.lockin_controller.push_frame(
                            thermal.copy(), time.perf_counter()
                        )
                    except Exception:
                        pass

                thermal = tnr(thermal, self._prev_frame, alpha=self.tnr_alpha)
                self._prev_frame = thermal.copy()

                rendered_thermal = self._render(thermal)
                final_display, record_frame = self._process_dual_camera(rendered_thermal)
                self._last_display = final_display

                if self.recording_video:
                    self._record_frame_to_video(record_frame)

                cv2.imshow(window_name, final_display)
                self._update_fps()

                if not self._handle_key(thermal):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.recording_video and self.video_writer is not None:
                self.video_writer.release()
                print(f"Saved video: {self.video_filename}")
            if self.cap_visible is not None:
                self.cap_visible.release()
            self.camera.stop_streaming()
            cv2.destroyAllWindows()

    def _record_frame_to_video(self, frame: NDArray[np.uint8]) -> None:
        """Safely write frame to video with lazy dimension alignment."""
        vh, vw = frame.shape[:2]
        if self.video_writer is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"p3_video_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, 25.0, (vw, vh)
            )
            self._video_dim = (vw, vh)
            print(f"Video recording STARTED: {self.video_filename} ({vw}x{vh})")

        if self._video_dim is not None and (vw, vh) != self._video_dim:
            frame = cv2.resize(frame, self._video_dim, interpolation=cv2.INTER_AREA)

        self.video_writer.write(frame)

    def _process_dual_camera(
        self, thermal_canvas: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Composite visible camera stream on left."""
        if self.cap_visible is not None and self.cap_visible.isOpened():
            ret, vis_frame = self.cap_visible.read()
            canvas_h = thermal_canvas.shape[0]
            thermal_h = max(1, canvas_h - self.top_bar_h - self.bottom_bar_h)

            if ret and vis_frame is not None:
                vh, vw = vis_frame.shape[:2]
                aspect = vw / max(1, vh)
                vis_w = int(thermal_h * aspect)

                gray_vis = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2GRAY)
                gray_vis = cv2.resize(gray_vis, (vis_w, thermal_h), interpolation=cv2.INTER_AREA)
                vis_bgr = cv2.cvtColor(gray_vis, cv2.COLOR_GRAY2BGR)
            else:
                vis_w = self.target_w
                vis_bgr = np.zeros((thermal_h, vis_w, 3), dtype=np.uint8)

            vis_padded = np.zeros((canvas_h, vis_w, 3), dtype=np.uint8)
            vis_padded[self.top_bar_h : self.top_bar_h + thermal_h, 0:vis_w] = vis_bgr

            composite = np.hstack([vis_padded, thermal_canvas])
            return composite, composite

        return thermal_canvas, thermal_canvas

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self._fps_count / (now - self._fps_time)
            self._fps_count = 0
            self._fps_time = now

    def _get_spot_coords(self, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        th, tw = thermal.shape
        cy, cx = th // 2, tw // 2
        if self.mirror:
            cx = tw - 1 - cx
        return cy, cx

    def _coord_to_image(
        self,
        coord: tuple[int, int],
        thermal: NDArray[np.uint16],
        image: NDArray[np.uint8],
    ) -> tuple[int, int]:
        h, w = image.shape[:2]
        if self.rotation in (90, 270):
            h, w = w, h
        th, tw = thermal.shape[:2]
        cy, cx = coord

        cx_d = int(round((cx / tw) * w))
        cy_d = int(round((cy / th) * h))

        if self.mirror:
            cx_d = w - cx_d - 1

        if self.rotation == 90:
            cy_d, cx_d = cx_d, h - cy_d - 1
        elif self.rotation == 180:
            cy_d, cx_d = h - cy_d - 1, w - cx_d - 1
        elif self.rotation == 270:
            cy_d, cx_d = w - cx_d - 1, cy_d

        return cy_d, cx_d

    def _draw_box_marker(
        self,
        coord: tuple[int, int],
        image: NDArray[np.uint8],
        color: tuple[int, int, int] = (255, 255, 255),
        size: int = 14,
    ) -> None:
        sf = self.scale_factor
        sz = int(size * sf // 2)
        cy, cx = coord
        h, w = image.shape[:2]

        cv2.rectangle(
            image,
            (max(0, cx - sz), max(0, cy - sz)),
            (min(w - 1, cx + sz), min(h - 1, cy + sz)),
            color,
            max(1, int(sf)),
            self.cv_linetype,
        )

    def _render(self, thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
        """Render thermal frame and build padded UI canvas."""
        frame_stats = {}
        env = self.camera.env_params
        temp_c = raw_to_celsius_corrected(thermal, env)

        if not self.use_celsius:
            temp_disp = celsius_to_fahrenheit(temp_c)
        else:
            temp_disp = temp_c.copy()

        cy, cx = self._get_spot_coords(thermal)
        tspot = float(temp_disp[cy, cx])
        frame_stats["tspot"] = tspot
        frame_stats["cmax"] = thermal.argmax()
        frame_stats["cmin"] = thermal.argmin()
        frame_stats["tmin"] = float(temp_disp.ravel()[frame_stats["cmin"]])
        frame_stats["tmax"] = float(temp_disp.ravel()[frame_stats["cmax"]])
        self._current_frame_stats = frame_stats

        # Buffer & log spot temperature
        self._log_spot_temp(tspot)

        use_fixed_range = (
            self.fixed_range_enabled
            and self.cbar_min is not None
            and self.cbar_max is not None
            and self.cbar_max > self.cbar_min
        )

        if use_fixed_range:
            t_min = cast(float, self.cbar_min)
            t_max = cast(float, self.cbar_max)

            # Continuous [t_min, t_max] mapping to [0, 255]
            norm_in = np.clip((temp_disp - t_min) / (t_max - t_min), 0.0, 1.0)
            img = (norm_in * 255.0).astype(np.uint8)
        else:
            if self.agc_mode == AGCMode.FACTORY:
                if (
                    self._ir_brightness is not None
                    and np.max(self._ir_brightness) > 0
                ):
                    img = self._ir_brightness.copy()
                else:
                    img = agc_temporal(thermal, pct=1.0)
            elif self.agc_mode == AGCMode.FIXED_RANGE:
                cmin = (
                    self.cbar_min
                    if self.cbar_min is not None
                    else (18.0 if self.use_celsius else 65.0)
                )
                cmax = (
                    self.cbar_max
                    if self.cbar_max is not None
                    else (35.0 if self.use_celsius else 95.0)
                )
                if not self.use_celsius:
                    cmin, cmax = fahrenheit_to_celsius(cmin), fahrenheit_to_celsius(cmax)
                img = agc_fixed(thermal, temp_min=cmin, temp_max=cmax)
            else:
                pct = AGC_PERCENTILES.get(self.agc_mode, 1.0)
                img = agc_temporal(thermal, pct=pct)

        if self.scale_mode != ScaleMode.OFF:
            h, w = img.shape[:2]
            resized: Any = cv2.resize(
                img, (w * 2, h * 2), interpolation=SCALE_INTERP[self.scale_mode]
            )
            img = np.asarray(resized, dtype=np.uint8)

        if not use_fixed_range and self.use_clahe:
            clahe_result: Any = self._clahe.apply(img)
            img = np.asarray(clahe_result, dtype=np.uint8)

        if not use_fixed_range:
            img = dde(img, strength=self.dde_strength)

        frame_stats["range_min"] = int(np.min(img))
        frame_stats["range_max"] = int(np.max(img))

        img = apply_colormap(img, self.colormap_idx)

        # Out-of-range edge detection logic
        if use_fixed_range:
            t_min = cast(float, self.cbar_min)
            t_max = cast(float, self.cbar_max)

            if temp_disp.shape[:2] != img.shape[:2]:
                temp_scaled = cv2.resize(
                    temp_disp,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                temp_scaled = temp_disp

            mask_cold = temp_scaled < t_min
            mask_hot = temp_scaled > t_max

            # Structural Sobel magnitude
            t_range = max(1e-3, float(temp_scaled.max() - temp_scaled.min()))
            temp_norm_u8 = np.clip(
                (temp_scaled - temp_scaled.min()) / t_range * 255.0, 0, 255
            ).astype(np.uint8)

            sobelx = cv2.Sobel(temp_norm_u8, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(temp_norm_u8, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(sobelx, sobely)

            p99 = float(np.percentile(edge_mag, 99.0))
            if p99 < 1e-5:
                p99 = 1.0
            edge_norm = np.clip(edge_mag / p99, 0.0, 1.0)
            edge_u8 = (np.power(edge_norm, 0.5) * 255.0).astype(np.uint8)

            # Cold (T < t_min): Dark background + bright white edges
            bg_cold = (temp_norm_u8.astype(np.float32) * 0.1).astype(np.uint8)
            gray_cold = np.clip(bg_cold.astype(np.int16) + edge_u8.astype(np.int16), 0, 255).astype(np.uint8)
            gray_cold_bgr = cv2.cvtColor(gray_cold, cv2.COLOR_GRAY2BGR)

            # Hot (T > t_max): White background + dark/black edges
            gray_hot = np.clip(255 - edge_u8.astype(np.int16), 0, 255).astype(np.uint8)
            gray_hot_bgr = cv2.cvtColor(gray_hot, cv2.COLOR_GRAY2BGR)

            for c in range(3):
                img[..., c] = np.where(mask_cold, gray_cold_bgr[..., c], img[..., c])
                img[..., c] = np.where(mask_hot, gray_hot_bgr[..., c], img[..., c])

        if self.mirror:
            img = cv2.flip(img, 1)

        if self.rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if (img.shape[1], img.shape[0]) != (self.target_w, self.target_h):
            img = cv2.resize(
                img, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC
            )

        h, w = img.shape[:2]
        thermal_img = cv2.resize(img, (w * self.zoom, h * self.zoom), interpolation=cv2.INTER_LINEAR) if self.zoom > 1 else img

        if self.show_reticule:
            sf = self.scale_factor
            th, tw = thermal_img.shape[:2]
            cx_d, cy_d = tw // 2, th // 2
            r_len = int(10 * sf)
            cv2.line(thermal_img, (cx_d - r_len, cy_d), (cx_d + r_len, cy_d), COLOR_RETICULE, max(1, int(sf)), self.cv_linetype)
            cv2.line(thermal_img, (cx_d, cy_d - r_len), (cx_d, cy_d + r_len), COLOR_RETICULE, max(1, int(sf)), self.cv_linetype)

        if self.hotspot_mode in [HotspotMode.MAX, HotspotMode.MINMAX]:
            cmax_d = self._coord_to_image(divmod(frame_stats["cmax"], thermal.shape[1]), thermal, thermal_img)
            self._draw_box_marker(cmax_d, thermal_img, color=COLOR_SPOT_MAX)

        if self.hotspot_mode in [HotspotMode.MIN, HotspotMode.MINMAX]:
            cmin_d = self._coord_to_image(divmod(frame_stats["cmin"], thermal.shape[1]), thermal, thermal_img)
            self._draw_box_marker(cmin_d, thermal_img, color=COLOR_SPOT_MIN)

        canvas = self._build_ui_canvas(thermal_img, frame_stats)

        if self.lockin_controller is not None:
            in_phase, quad, amplitude, angle = self.lockin_controller.get_latest()
            if in_phase is not None and quad is not None and amplitude is not None and angle is not None:
                try:
                    canvas = self._composite_lockin_panes(canvas, in_phase, quad, amplitude, angle)
                except Exception:
                    pass

        return canvas

    def _build_ui_canvas(
            self, thermal_img: NDArray[np.uint8], frame_stats: dict
    ) -> NDArray[np.uint8]:
        """Place thermal image into canvas with interactive top/bottom/right UI bars."""
        th, tw = thermal_img.shape[:2]
        sf = self.scale_factor

        cw = tw + (self.right_bar_w if self.show_colorbar else 0)
        ch = th + self.top_bar_h + self.bottom_bar_h

        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[self.top_bar_h : self.top_bar_h + th, 0:tw] = thermal_img

        # Top Status Text with explicit shortcut keys
        rec_status = " [REC(v)]" if self.recording_video else ""
        log_status = " [LOG(l)]" if self.logging_enabled else ""
        if self.fixed_range_enabled and self.cbar_min is not None and self.cbar_max is not None:
            top_status = f"Spot: {frame_stats['tspot']:.1f}{self.unit_str} | Min: {frame_stats['tmin']:.1f}{self.unit_str} | Max: {frame_stats['tmax']:.1f}{self.unit_str} | Fixed(f) [{self.cbar_min:.1f}-{self.cbar_max:.1f}{self.unit_str}]:ON{rec_status}{log_status}"
        else:
            top_status = f"Spot: {frame_stats['tspot']:.1f}{self.unit_str} | Min: {frame_stats['tmin']:.1f}{self.unit_str} | Max: {frame_stats['tmax']:.1f}{self.unit_str} | Fixed(f):OFF [AUTO]{rec_status}{log_status}"

        cv2.putText(
            canvas, top_status, (int(8 * sf), int(14 * sf)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype
        )

        # Draw Interactive Top UI Buttons
        self._draw_ui_buttons(canvas)

        # Bottom Bar Status
        cmap_name = ColormapID(self.colormap_idx).name
        gain_name = self.camera.gain_mode.name
        emissivity = self.camera.env_params.emissivity
        scale = self.scale_mode.name if self.scale_mode != ScaleMode.OFF else ""
        status = f"{self.fps:.1f} FPS | {cmap_name} | {gain_name} | e={emissivity:.2f} {scale}"
        cv2.putText(
            canvas, status, (int(8 * sf), ch - int(8 * sf)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype
        )

        if self.show_colorbar:
            self._draw_colorbar(canvas, frame_stats, tw, th)

        if self.show_help:
            self._draw_help(canvas)

        return canvas
    

    def _draw_ui_buttons(self, canvas: NDArray[np.uint8]) -> None:
        """Draw interactive canvas UI buttons with shortcut keys and mouse hover effects."""
        sf = self.scale_factor
        by1 = int(18 * sf)
        by2 = int(32 * sf)

        buttons_def = [
            ("btn_fixed", "Fixed(f)", self.fixed_range_enabled, (0, 165, 255)),
            ("btn_logging", "SpotLog(l)", self.logging_enabled, (0, 200, 0)),
            ("btn_video", "RecVideo(v)", self.recording_video, (0, 0, 220)),
            ("btn_help", "Help(h)", self.show_help, (180, 180, 0)),
        ]

        x_cursor = int(8 * sf)
        self.ui_buttons.clear()

        for btn_id, label, is_active, active_color in buttons_def:
            state_str = "ON" if is_active else "OFF"
            btn_text = f" {label}:{state_str} " if btn_id != "btn_help" else f" {label} "

            text_size, _ = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * sf, max(1, int(sf)))
            bw = text_size[0] + int(6 * sf)
            bx1, bx2 = x_cursor, x_cursor + bw

            self.ui_buttons[btn_id] = (bx1, by1, bx2, by2)

            is_hover = (bx1 <= self.mouse_x <= bx2) and (by1 <= self.mouse_y <= by2)

            if is_hover:
                bg_col = (100, 100, 100) if not is_active else active_color
                border_col = (255, 255, 255)
                text_col = (255, 255, 255)
            else:
                bg_col = (40, 40, 40) if not is_active else (int(active_color[0] * 0.6), int(active_color[1] * 0.6), int(active_color[2] * 0.6))
                border_col = (120, 120, 120)
                text_col = (220, 220, 220)

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_col, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border_col, 1, self.cv_linetype)
            cv2.putText(
                canvas, btn_text, (bx1 + int(2 * sf), by1 + int(10 * sf)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * sf, text_col, max(1, int(sf)), self.cv_linetype
            )

            x_cursor += bw + int(8 * sf)

    def _draw_colorbar(
        self,
        canvas: NDArray[np.uint8],
        frame_stats: dict,
        thermal_w: int,
        thermal_h: int,
    ) -> None:
        """Draw colorbar in right padded bar."""
        sf = self.scale_factor
        cb_width = int(14 * sf)
        ticks = 5

        h_cbar = int(0.8 * thermal_h)
        y_offset = self.top_bar_h + int(0.1 * thermal_h)
        x_start = thermal_w + int(8 * sf)

        use_fixed = (
            self.fixed_range_enabled
            and self.cbar_min is not None
            and self.cbar_max is not None
            and self.cbar_max > self.cbar_min
        )

        if use_fixed:
            cb_min = cast(float, self.cbar_min)
            cb_max = cast(float, self.cbar_max)
            ind_128 = np.linspace(0, 127, h_cbar).astype(np.uint8)
            ind_256 = (ind_128.astype(np.float32) * (255.0 / 127.0)).astype(np.uint8)
            cmap_resamp = get_colormap(self.colormap_idx)[ind_256[::-1]]
        else:
            cb_min = frame_stats["tmin"]
            cb_max = frame_stats["tmax"]
            val_range = max(1, frame_stats["range_max"] - frame_stats["range_min"])
            ind = (np.arange(0.5, h_cbar) / h_cbar) * val_range
            ind = (ind + frame_stats["range_min"] + 0.5).astype(np.uint8)
            cmap_resamp = get_colormap(self.colormap_idx)[ind[::-1]]

        canvas[y_offset : y_offset + h_cbar, x_start : x_start + cb_width] = (
            cmap_resamp.reshape(h_cbar, 1, 3)
        )

        cv2.rectangle(
            canvas, (x_start, y_offset), (x_start + cb_width, y_offset + h_cbar),
            COLOR_TEXT, max(1, int(sf)), self.cv_linetype
        )

        tick_pos = (np.linspace(0, 1, ticks) * h_cbar + y_offset).astype(np.uint16)[::-1]
        tick_labels = [f"{t:.1f}{self.unit_str}" for t in np.linspace(cb_min, cb_max, ticks)]
        tick_notch_len = max(2, int(3 * sf))

        for pos, label in zip(tick_pos, tick_labels):
            cv2.line(
                canvas, (x_start + cb_width, pos), (x_start + cb_width + tick_notch_len, pos),
                COLOR_TEXT, max(1, int(sf)), self.cv_linetype
            )
            cv2.putText(
                canvas, label, (x_start + cb_width + tick_notch_len + int(3 * sf), pos + int(4 * sf)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype
            )

    def _draw_help(self, canvas: NDArray[np.uint8]) -> None:
        """Draw help legend showing keyboard shortcuts."""
        sf = self.scale_factor
        lines = [
            "q-Quit  h-Help  v-Video Rec",
            "f-Fixed range  l-Spot Log",
            "space-Shot  D-Dump raw",
            "+/- Zoom  r-Rotate  m-Mirror",
            "s-Shutter  g-High/Low Gain",
            "c-Colormap  e-Emissivity",
            "1-9 Set emissivity (0.1-0.9)",
            "x-Scale  p-Enhanced  d-DDE",
            "t-Reticule  b-Hotspot markers",
        ]
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (int(5 * sf), self.top_bar_h + int(5 * sf)),
            (int(250 * sf), self.top_bar_h + int(185 * sf)),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        for i, line in enumerate(lines):
            cv2.putText(
                canvas, line,
                (int(10 * sf), self.top_bar_h + int((24 + i * 18) * sf)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * sf, COLOR_TEXT, max(1, int(sf)), self.cv_linetype
            )
            

    def _toggle_video_recording(self) -> None:
        """Toggle video recording state."""
        if not self.recording_video:
            self.recording_video = True
            self.video_writer = None
            self._video_dim = None
        else:
            self.recording_video = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self._video_dim = None
            print(f"Video recording STOPPED and saved: {self.video_filename}")

    def _toggle_spot_logging(self) -> None:
        """Toggle spot temperature logging live."""
        self.logging_enabled = not self.logging_enabled
        if self.logging_enabled:
            if not self.logfile:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.logfile = f"p3_spot_log_{ts}.csv"
            self._init_logfile(self.logfile)
            print(f"Spot logging: STARTED -> {self.logfile}")
        else:
            print("Spot logging: STOPPED")

    def _toggle_fixed_lock(self) -> None:
        """Toggle fixed temperature range lock on and off."""
        self.fixed_range_enabled = not self.fixed_range_enabled
        if self.fixed_range_enabled:
            if not self._user_cli_bounds or self.cbar_min is None or self.cbar_max is None:
                self.cbar_min = self._current_frame_stats.get(
                    "tmin", 18.0 if self.use_celsius else 65.0
                )
                self.cbar_max = self._current_frame_stats.get(
                    "tmax", 35.0 if self.use_celsius else 95.0
                )
            print(f"Fixed range lock: ON ({self.cbar_min:.1f} - {self.cbar_max:.1f} {self.unit_str})")
        else:
            if not self._user_cli_bounds:
                self.cbar_min = None
                self.cbar_max = None
            print("Fixed range lock: OFF (Dynamic AGC)")

    def _handle_key(self, thermal: NDArray[np.uint16]) -> bool:
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            return True

        if key == ord("q"):
            return False

        if key == ord("v"):
            self._toggle_video_recording()
        elif key in (ord("l"), ord("L"), ord("o")):  # 'l' or 'L' toggles logging
            self._toggle_spot_logging()
        elif key == ord("f"):
            self._toggle_fixed_lock()
        elif key == ord("r"):
            self.rotation = (self.rotation + 90) % 360
            print(f"Rotation: {self.rotation} deg")
        elif key == ord("c"):
            self.colormap_idx = (self.colormap_idx + 1) % len(ColormapID)
        elif key == ord("s"):
            self.camera.trigger_shutter()
            print("Shutter triggered")
        elif key == ord("g"):
            new_mode = (
                GainMode.LOW
                if self.camera.gain_mode == GainMode.HIGH
                else GainMode.HIGH
            )
            self.camera.set_gain_mode(new_mode)
            print(f"Gain mode: {new_mode.name}")
        elif key == ord("m"):
            self.mirror = not self.mirror
            print("Mirror:", "ON" if self.mirror else "OFF")
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("e"):
            self._cycle_emissivity()
        elif ord("1") <= key <= ord("9"):
            self._set_emissivity((key - ord("0")) / 10.0)
        elif key == ord("d"):
            self._toggle_dde()
        elif key == ord("D"):
            self._dump(thermal)
        elif key == ord(" "):
            self._screenshot()
        elif key in (ord("+"), ord("=")):
            self.zoom = min(5, self.zoom + 1)
        elif key in (ord("-"), ord("_")):
            self.zoom = max(1, self.zoom - 1)
        elif key == ord("x"):
            self.scale_mode = ScaleMode((self.scale_mode + 1) % len(ScaleMode))
            print(f"Scale: {self.scale_mode.name}")
        elif key == ord("p"):
            self._toggle_enhanced()
        elif key == ord("a"):
            self.agc_mode = AGCMode((self.agc_mode + 1) % len(AGCMode))
            print(f"AGC: {self.agc_mode.name}")
        elif key == ord("t"):
            self.show_reticule = not self.show_reticule
        elif key == ord("b"):
            self.hotspot_mode = HotspotMode((self.hotspot_mode + 1) % len(HotspotMode))
            print(f"Hotspot mode: {self.hotspot_mode.name}")

        return True

    def _toggle_enhanced(self) -> None:
        self.enhanced = not self.enhanced
        self.use_clahe = self.enhanced
        self.dde_strength = 0.3 if self.enhanced else 0.0
        print(f"Enhanced: {'ON' if self.enhanced else 'OFF'}")

    def _toggle_dde(self) -> None:
        self.dde_strength = 0.0 if self.dde_strength > 0 else 0.3
        print(f"DDE: {'ON' if self.dde_strength > 0 else 'OFF'}")

    def _dump(self, thermal: NDArray[np.uint16]) -> None:
        ts = time.strftime("%H%M%S")
        cy, cx = self._get_spot_coords(thermal)
        env = self.camera.env_params
        raw_temp_c = raw_to_celsius(thermal[cy, cx])
        corrected_temp_c = raw_to_celsius_corrected(thermal[cy, cx], env)
        raw_temp = celsius_to_fahrenheit(raw_temp_c) if not self.use_celsius else raw_temp_c
        corrected_temp = celsius_to_fahrenheit(corrected_temp_c) if not self.use_celsius else corrected_temp_c
        print(f"\n--- Dump {ts} ---")
        print(f"Center raw: {thermal[cy, cx]}")
        print(f"Center temp (uncorrected): {raw_temp:.1f}{self.unit_str}")
        print(f"Center temp (e={env.emissivity:.2f}): {corrected_temp:.1f}{self.unit_str}")
        np.save(f"p3_raw_{ts}.npy", thermal)
        print(f"Saved: p3_raw_{ts}.npy\n")

    def _screenshot(self) -> None:
        if self._last_display is None:
            print("No frame to save")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"p3_{ts}.png"
        cv2.imwrite(filename, self._last_display)
        print(f"Saved: {filename}")

    def _set_emissivity(self, value: float) -> None:
        self.camera.env_params.emissivity = value
        print(f"Emissivity: {value:.2f}")

    def _cycle_emissivity(self) -> None:
        values = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10]
        current = self.camera.env_params.emissivity
        idx = 0
        for i, v in enumerate(values):
            if abs(current - v) < 0.01:
                idx = (i + 1) % len(values)
                break
        self._set_emissivity(values[idx])

def load_config_file(config_path: str) -> dict[str, Any]:
    """Load settings dictionary from JSON configuration file and normalize keys."""
    alias_map = {
        # Webcam / Visible Camera Aliases
        "webcam": "visible_cam",
        "visible_cam": "visible_cam",
        "visible-cam": "visible_cam",
        # Temperature Range Aliases
        "min": "min",
        "cbar_min": "min",
        "cbar-min": "min",
        "temp_min": "min",
        "temp-min": "min",
        "max": "max",
        "cbar_max": "max",
        "cbar-max": "max",
        "temp_max": "max",
        "temp-max": "max",
        # System & Serial Port Parameters
        "serial_port": "serial_port",
        "serial-port": "serial_port",
        "baud_rate": "baud_rate",
        "baud-rate": "baud_rate",
        "high_res": "high_res",
        "high-res": "high_res",
        "lockin_period": "period",
        "period": "period",
        "lockin_integration": "integration",
        "integration": "integration",
        "lockin_invert": "invert",
        "invert": "invert",
        "celsius": "celsius",
        "model": "model",
        "logfile": "logfile",
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
            print(f"Warning: Config file {config_path} is not a valid JSON dictionary.")
    except Exception as e:
        print(f"Error reading config file {config_path}: {e}")
    return {}


def main() -> None:
    """Entry point."""

    def _parse_bool(val: str) -> bool:
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {val}")


    # Pass 1: Parse --config or fall back to default 'config.json' if it exists
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
    parser = argparse.ArgumentParser(
        description="Thermal Master P1/P3 USB thermal camera viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "---config", "-c",
        type=str, default=None,
        help="Path to JSON configuration file (defaults to config.json if present)",
    )        

    parser.add_argument(
        "--model", type=str, choices=["p1", "p3"], default="p3",
        help="Camera model (default: p3)",
    )
    parser.add_argument(
        "--serial-port", type=str, default="/dev/ttyACM0",
        help="Serial port for lock-in load controller",
    )
    parser.add_argument("--baud-rate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--period", type=float, default=1.0, help="Lock-in period")
    parser.add_argument("--integration", type=float, default=60.0, help="Lock-in integration")
    parser.add_argument("--invert", nargs="?", const="1", type=_parse_bool, default=False)
    parser.add_argument("--min", "--temp-min", "--cbar-min", type=float, default=None)
    parser.add_argument("--max", "--temp-max", "--cbar-max", type=float, default=None)
    parser.add_argument("--celsius", action="store_true", default=False)
    parser.add_argument("--visible-cam", "--webcam", type=str, default=None)
    parser.add_argument("--high-res", action="store_true", default=False)
    parser.add_argument("--logfile", type=str, default=None)

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    try:
        P3Viewer(
            model=args.model,
            serial_port=args.serial_port,
            baud_rate=args.baud_rate,
            lockin_period=args.period,
            lockin_integration=args.integration,
            lockin_invert=args.invert,
            cbar_min=args.min,
            cbar_max=args.max,
            use_celsius=args.celsius,
            visible_cam=args.visible_cam,
            high_res=args.high_res,
            logfile=args.logfile,
        ).run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")

if __name__ == "__main__":
    main()
