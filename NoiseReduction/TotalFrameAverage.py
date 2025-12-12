"""
temporal_averages.py

Two functions:
 - total_temporal_average(...)
 - recursive_temporal_average(...)

Each returns:
  - np.ndarray : final averaged frame (uint8 or float32) on success

Raises:
  - VideoOpenError, InvalidParameterError, TemporalAverageError, or other exceptions.
"""

from typing import Tuple, Optional
import os
import cv2 as cv
import numpy as np
import time


# ---------------------------
# Exceptions & Helper
# ---------------------------
class TemporalAverageError(Exception):
    pass


class VideoOpenError(TemporalAverageError):
    pass


class InvalidParameterError(TemporalAverageError):
    pass


def _safe_convert_to_single_channel(frame: np.ndarray) -> np.ndarray:
    """Convert frame -> single-channel float32 (grayscale)."""
    if frame is None:
        raise InvalidParameterError("Input frame is None.")

    if frame.ndim == 2:
        return frame.astype(np.float32)

    if frame.ndim == 3:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)

    raise InvalidParameterError(f"Unsupported frame ndim: {frame.ndim}")


# ---------------------------
# Total / Running-average (incremental mean or accumulation)
# ---------------------------
def total_temporal_average(
    video_path: str,
    *,
    is_thermal: bool = False,
    preserve_radiometric: bool = False,
    max_frames: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_dtype: str = "uint8",
    units: str = "raw",
    apply_normalization: bool = True,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    show: bool = False,
    verbose: bool = False,
    alpha: float = 0.02,              # parity only
    running_average_mode: bool = True
) -> np.ndarray:
    """
    Full-sequence temporal averaging (incremental mean).

    Args (types & meaning):

      video_path (str):
          Path to input video file.
          Min/Max: valid file path.
          Units: filesystem path.
          Default: required.
          Best case: Local SSD path.

      is_thermal (bool):
          Whether frames represent radiometric thermal data.
          Min/Max: True/False.
          Default: False.
          Best case: True for calibrated thermal processing.

      preserve_radiometric (bool):
          Keep raw float radiometric values (no normalization).
          Min/Max: True/False.
          Default: False.
          Best case: True when further scientific analysis is needed.

      max_frames (Optional[int]):
          Number of frames to process.
          Min: 1, Max: video length.
          Units: frames.
          Default: None (all frames).
          Best case: small value for testing, None for full accuracy.

      roi (Optional[Tuple[int,int,int,int]]):
          Region of interest as (x,y,w,h).
          Min/Max: must fit inside frame.
          Units: pixels.
          Default: None.
          Best case: cropped stable background.

      output_dtype (str):
          Output data type.
          Options: "uint8", "float32".
          Default: "uint8".
          Best case: "float32" for precision.

      units (str):
          Data unit label.
          Options: "raw", "celsius", "kelvin".
          Default: "raw".
          Best case: match camera specification.

      apply_normalization (bool):
          Normalize output using cv.normalize.
          Default: True.
          Best case: True for display, False for raw analysis.

      norm_alpha (float):
          Normalization lower bound.
          Default: 0.0.
          Units: intensity.

      norm_beta (float):
          Normalization upper bound.
          Default: 255.0.
          Units: intensity.

      norm_type (int):
          cv.normalize method.
          Default: cv.NORM_MINMAX.

      show (bool):
          Display debug windows.
          Default: False.

      verbose (bool):
          Add verbose info to metadata.
          Default: False.

      alpha (float):
          Unused in total averaging (kept for API uniformity).
          Default: 0.02.

      running_average_mode (bool):
          True → incremental mean, False → sum accumulation.
          Default: True.

    Returns:
      avg_frame (np.ndarray | None):
          Final averaged frame (uint8 or float32).
      metadata (dict):
          All processing statistics and error flags.
    """

    """
    Compute temporal average across video using an incremental running mean
    (or accumulation if running_average_mode=False).

    """

    start_time = time.perf_counter()

    # ---- Validate paths & params ----
    if not os.path.exists(video_path):
        raise VideoOpenError(f"Video file not found: {video_path}")

    if output_dtype not in ("uint8", "float32"):
        raise InvalidParameterError("output_dtype must be 'uint8' or 'float32'.")

    if units not in ("raw", "celsius", "kelvin"):
        raise InvalidParameterError("units must be 'raw', 'celsius', or 'kelvin'.")

    if max_frames is not None and max_frames < 1:
        raise InvalidParameterError("max_frames must be >= 1 or None.")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError("Unable to open video.")

    try:
        accumulator: Optional[np.ndarray] = None  # dtype float64 accumulator
        global_min = np.inf
        global_max = -np.inf
        frames = 0

        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]

            cur = _safe_convert_to_single_channel(frame)  # float32

            fmin, fmax = float(cur.min()), float(cur.max())
            global_min = min(global_min, fmin)
            global_max = max(global_max, fmax)

            if accumulator is None:
                accumulator = cur.astype(np.float64)
                frames = 1
            else:
                frames += 1
                if running_average_mode:
                    # incremental mean update
                    accumulator += (cur - accumulator) / frames
                else:
                    # accumulate sum
                    accumulator += cur

            if max_frames and frames >= max_frames:
                break

            # optional display (show current + accumulated/mean)
            if show:
                display_img = accumulator.astype(np.float32)
                disp = np.zeros_like(display_img, dtype=np.uint8)
                cv.normalize(display_img, disp, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                cv.imshow("Total - Current (gray)", cv.convertScaleAbs(cur))
                cv.imshow("Total - Accumulator", disp)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

        if frames == 0 or accumulator is None:
            raise TemporalAverageError("No frames processed.")

        avg_float32 = accumulator.astype(np.float32)

        # Preserve radiometric (raw float) if requested
        if preserve_radiometric:
            if show:
                cv.destroyAllWindows()
            return avg_float32

        # Normalize for display / output if requested
        if apply_normalization:
            avg_norm = cv.normalize(avg_float32, None, norm_alpha, norm_beta, norm_type)
        else:
            avg_norm = avg_float32

        # Convert requested output dtype
        if output_dtype == "uint8":
            avg_out_image = np.clip(avg_norm, 0, 255).astype(np.uint8)
        else:
            avg_out_image = avg_norm.astype(np.float32)

        if show:
            cv.destroyAllWindows()


        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"[total_temporal_average] Time Taken: {total_time:.6f} seconds")

        return avg_out_image

    except Exception:
        try:
            cap.release()
        except Exception:
            pass
        raise


# ---------------------------
# Recursive / Exponential Moving Average
# ---------------------------
def recursive_temporal_average(
    video_path: str,
    *,
    is_thermal: bool = False,
    preserve_radiometric: bool = False,
    max_frames: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_dtype: str = "uint8",
    units: str = "raw",
    apply_normalization: bool = True,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    show: bool = False,
    verbose: bool = False,
    alpha: float = 0.02,  # used by recursive averaging
    running_average_mode: bool = True,  # present for signature parity but ignored here
) -> np.ndarray:
    
    """
    Recursive exponential-moving-average over frames:
        O_new = (1 - alpha) * O_old + alpha * I_k

    Same parameter list as total_temporal_average (alpha used here).

    Args (types & meaning):

      video_path (str):
          Path to input video file.
          Min/Max: valid file path.
          Units: filesystem path.
          Default: required.
          Best case: Local SSD path.

      is_thermal (bool):
          Whether frames represent radiometric thermal data.
          Min/Max: True/False.
          Default: False.
          Best case: True for calibrated thermal processing.

      preserve_radiometric (bool):
          Keep raw float radiometric values (no normalization).
          Min/Max: True/False.
          Default: False.
          Best case: True when further scientific analysis is needed.

      max_frames (Optional[int]):
          Number of frames to process.
          Min: 1, Max: video length.
          Units: frames.
          Default: None (all frames).
          Best case: small value for testing, None for full accuracy.

      roi (Optional[Tuple[int,int,int,int]]):
          Region of interest as (x,y,w,h).
          Min/Max: must fit inside frame.
          Units: pixels.
          Default: None.
          Best case: cropped stable background.

      output_dtype (str):
          Output data type.
          Options: "uint8", "float32".
          Default: "uint8".
          Best case: "float32" for precision.

      units (str):
          Data unit label.
          Options: "raw", "celsius", "kelvin".
          Default: "raw".
          Best case: match camera specification.

      apply_normalization (bool):
          Normalize output using cv.normalize.
          Default: True.
          Best case: True for display, False for raw analysis.

      norm_alpha (float):
          Normalization lower bound.
          Default: 0.0.
          Units: intensity.

      norm_beta (float):
          Normalization upper bound.
          Default: 255.0.
          Units: intensity.

      norm_type (int):
          cv.normalize method.
          Default: cv.NORM_MINMAX.

      show (bool):
          Display debug windows.
          Default: False.

      verbose (bool):
          Add verbose info to metadata.
          Default: False.

      alpha (float):
          Exponential filter weight used in recursive update:
              O_new = (1 - alpha) * O_old + alpha * I_k
          Min/Max: (0.0, 1.0]
          Units: unitless
          Default: 0.02
          Best case: 0.01-0.05 for stable background; larger for faster adaptation.

      running_average_mode (bool):
          Present for API uniformity; ignored by recursive method.
          Default: True.

    Returns:
      avg_frame (np.ndarray | None):
          Final recursively averaged frame (uint8 or float32).
      metadata (dict):
          All processing statistics and error flags.
    """

    start_time = time.perf_counter()

    # ---- Validate paths & params ----
    if not os.path.exists(video_path):
        raise VideoOpenError(f"Video file not found: {video_path}")

    if not (0.0 < alpha <= 1.0):
        raise InvalidParameterError("alpha must be in (0.0, 1.0].")

    if output_dtype not in ("uint8", "float32"):
        raise InvalidParameterError("output_dtype must be 'uint8' or 'float32'.")

    if units not in ("raw", "celsius", "kelvin"):
        raise InvalidParameterError("units must be 'raw', 'celsius', or 'kelvin'.")

    if max_frames is not None and max_frames < 1:
        raise InvalidParameterError("max_frames must be >= 1 or None.")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError("Unable to open video.")

    try:
        # initialize with first frame
        ret, frame_1 = cap.read()
        if not ret:
            cap.release()
            raise VideoOpenError("Video is empty.")

        if roi is not None:
            x, y, w, h = roi
            frame_1 = frame_1[y:y + h, x:x + w]

        I1 = _safe_convert_to_single_channel(frame_1)
        accumulator = I1.astype(np.float64)  # high-precision accumulator
        global_min = float(I1.min())
        global_max = float(I1.max())
        frames = 1

        weight_old = 1.0 - alpha
        weight_new = alpha

        while True:
            ret, frame_k = cap.read()
            if not ret:
                break

            if roi is not None:
                x, y, w, h = roi
                frame_k = frame_k[y:y + h, x:x + w]

            cur = _safe_convert_to_single_channel(frame_k)

            # shape check
            if cur.shape != accumulator.shape:
                break

            # update accumulator using addWeighted
            accumulator = cv.addWeighted(
                accumulator, weight_old,
                cur.astype(np.float64), weight_new,
                0.0
            )

            frames += 1
            # update global min/max
            fmin, fmax = float(cur.min()), float(cur.max())
            global_min = min(global_min, fmin)
            global_max = max(global_max, fmax)

            if max_frames and frames >= max_frames:
                break

            # optional display
            if show:
                disp = np.zeros_like(accumulator, dtype=np.uint8)
                cv.normalize(accumulator, disp, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                cv.imshow("Recursive - Current (gray)", cv.convertScaleAbs(cur))
                cv.imshow("Recursive - Accumulator", disp)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

        avg_float32 = accumulator.astype(np.float32)

        if preserve_radiometric:
            if show:
                cv.destroyAllWindows()
            return avg_float32

        if apply_normalization:
            avg_norm = cv.normalize(avg_float32, None, norm_alpha, norm_beta, norm_type)
        else:
            avg_norm = avg_float32

        if output_dtype == "uint8":
            rec_out_image = np.clip(avg_norm, 0, 255).astype(np.uint8)
        else:
            rec_out_image = avg_norm.astype(np.float32)

        if show:
            cv.destroyAllWindows()


        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"[recursive_temporal_average] Time Taken: {total_time:.6f} seconds")
        return rec_out_image
    


    except Exception:
        try:
            cap.release()
        except Exception:
            pass
        raise
    


if __name__ == "__main__":
    VIDEO = "/home/user1/learning/Testing/Videos/video.mp4"

    # Example 1: total mean (same signature)
    avg_total = total_temporal_average(
        VIDEO,
        is_thermal=False,
        preserve_radiometric=False,
        max_frames=None,
        roi=None,
        output_dtype="uint8",
        units="raw",
        apply_normalization=True,
        norm_alpha=0,
        norm_beta=255,
        norm_type=cv.NORM_MINMAX,
        show=False,
        verbose=True,
        alpha=0.02,               # present but ignored by total
        running_average_mode=True,  # incremental mean
    )

    # Example 2: recursive average (same signature)
    avg_rec = recursive_temporal_average(
        VIDEO,
        is_thermal=False,
        preserve_radiometric=False,
        max_frames=None,
        roi=None,
        output_dtype="uint8",
        units="raw",
        apply_normalization=True,
        norm_alpha=0,
        norm_beta=255,
        norm_type=cv.NORM_MINMAX,
        show=False,
        verbose=True,
        alpha=0.02,               # used by recursive
        running_average_mode=True,  # ignored by recursive
    )

    # display results if available
    if avg_total is not None:
        cv.imshow("Total Frame Average Result", avg_total if avg_total.dtype == np.uint8 else cv.convertScaleAbs(avg_total))
        cv.waitKey(0)
        cv.destroyAllWindows()

    if avg_rec is not None:
        cv.imshow("Recursive Frame Average Result", avg_rec if avg_rec.dtype == np.uint8 else cv.convertScaleAbs(avg_rec))
        cv.waitKey(0)
        cv.destroyAllWindows()