from typing import Optional, Tuple, Dict, Any
import cv2 as cv
import numpy as np
import os


# ---------------------------
# Custom Exceptions
# ---------------------------
class TemporalAverageError(Exception):
    pass


class VideoOpenError(TemporalAverageError):
    pass


class InvalidParameterError(TemporalAverageError):
    pass


# ---------------------------
# Helper
# ---------------------------
def _safe_convert_to_single_channel(frame: np.ndarray) -> np.ndarray:
    """Convert frame → single-channel float32."""
    if frame is None:
        raise InvalidParameterError("Input frame is None.")

    if frame.ndim == 2:
        return frame.astype(np.float32)

    if frame.ndim == 3:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)

    raise InvalidParameterError(f"Unsupported frame ndim: {frame.ndim}")


# ---------------------------
# Temporal Average Function
# ---------------------------
def calculate_temporal_average_frame(
    video_path: str,
    *,
    is_thermal: bool = False,
    preserve_radiometric: bool = False,
    running_average: bool = True,
    max_frames: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_dtype: str = "uint8",
    units: str = "raw",
    apply_normalization: bool = True,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    norm_dst: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:

    # ---- Validate ----
    if not os.path.exists(video_path):
        raise VideoOpenError(f"Video file not found: {video_path}")

    if output_dtype not in ("uint8", "float32"):
        raise InvalidParameterError("output_dtype must be 'uint8' or 'float32'.")

    if units not in ("raw", "celsius", "kelvin"):
        raise InvalidParameterError("units must be 'raw', 'celsius', or 'kelvin'.")

    if max_frames is not None and max_frames < 1:
        raise InvalidParameterError("max_frames must be >=1 or None.")

    # ---- Metadata ----
    meta = {
        "frames_processed": 0,
        "frame_width": None,
        "frame_height": None,
        "input_dtype": None,
        "input_min": None,
        "input_max": None,
        "units": units,
        "preserve_radiometric": preserve_radiometric,
        "error": None,
        "output_dtype": output_dtype,
    }

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError("Unable to open video.")

    try:
        W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        meta["frame_width"] = W
        meta["frame_height"] = H

        running_avg = None
        global_min = np.inf
        global_max = -np.inf

        frames = 0

        # ---- Read Frames ----
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            cur = _safe_convert_to_single_channel(frame)
            meta["input_dtype"] = str(frame.dtype)

            fmin, fmax = float(cur.min()), float(cur.max())
            global_min = min(global_min, fmin)
            global_max = max(global_max, fmax)

            if running_avg is None:
                running_avg = cur.astype(np.float64)
                frames = 1
            else:
                frames += 1
                if running_average:
                    running_avg += (cur - running_avg) / frames
                else:
                    running_avg += cur

            if max_frames and frames >= max_frames:
                break

        cap.release()

        if frames == 0:
            meta["error"] = "No frames processed."
            return None, meta

        meta["frames_processed"] = frames
        meta["input_min"] = global_min
        meta["input_max"] = global_max

        avg = running_avg.astype(np.float32)

        if preserve_radiometric:
            return avg, meta

        # ---- Normalize ----
        if apply_normalization:
            avg = cv.normalize(avg, norm_dst, norm_alpha, norm_beta, norm_type)

        # ---- Convert dtype ----
        if output_dtype == "uint8":
            avg = np.clip(avg, 0, 255).astype(np.uint8)
        else:
            avg = avg.astype(np.float32)

        return avg, meta

    except Exception as e:
        meta["error"] = str(e)
        return None, meta
    

# ---------------------------
# Example + Metadata Logging
# ---------------------------
if __name__ == "__main__":

    VIDEO = "/home/user1/learning/Testing/Videos/video.mp4"

    avg, meta = calculate_temporal_average_frame(
        VIDEO,
        is_thermal=False,
        preserve_radiometric=False,
        output_dtype="uint8",
        apply_normalization=True,
        norm_alpha=0,
        norm_beta=255,
        norm_type=cv.NORM_MINMAX
    )

    print("\n========== TEMPORAL AVERAGING METADATA ==========")
    for key, val in meta.items():
        print(f"{key:20}: {val}")
    print("=================================================\n")

    if avg is not None:
        cv.imshow("Temporal Average Frame", avg)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("ERROR:", meta["error"])
