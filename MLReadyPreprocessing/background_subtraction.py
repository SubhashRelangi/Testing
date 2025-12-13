import cv2
import numpy as np
import time
from typing import Union


def _open_video(video: Union[str, cv2.VideoCapture]) -> cv2.VideoCapture:
    
    """
    Open and validate a video source.

    PARAMETERS
    ----------
    video : str | cv2.VideoCapture
        Description:
            Input video source. Can be a file path or an already opened
            cv2.VideoCapture object.

    RETURNS
    -------
    cap : cv2.VideoCapture
        Opened and validated video capture object.

    EXCEPTIONS
    ----------
    TypeError:
        If input is not a string or cv2.VideoCapture.

    IOError:
        If video source cannot be opened.
    """

    if isinstance(video, cv2.VideoCapture):
        cap = video
    elif isinstance(video, str):
        cap = cv2.VideoCapture(video)
    else:
        raise TypeError("video must be a file path or cv2.VideoCapture")

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    return cap


def _ensure_grayscale_frame(frame: np.ndarray) -> np.ndarray:

    """
    Ensure a frame is single-channel grayscale.

    PARAMETERS
    ----------
    frame : np.ndarray
        Description:
            Input video frame.

        Min & Max:
            - Minimum: 1 × 1 pixels
            - Maximum: system memory bound

        Units:
            Pixel intensity

        Default:
            Required

        Best-case:
            Already single-channel uint8 frame

    RETURNS
    -------
    gray_frame : np.ndarray
        Single-channel grayscale frame.

    EXCEPTIONS
    ----------
    ValueError:
        If frame is empty or invalid.
    """

    
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame.ndim != 2:
        raise ValueError(f"Expected grayscale frame, got shape={frame.shape}")

    return frame


def _validate_frame_dtype(frame: np.ndarray) -> None:
    
    """
    Validate supported frame data types.

    PARAMETERS
    ----------
    frame : np.ndarray
        Description:
            Input grayscale frame.

        Supported dtypes:
            uint8, uint16, float32, float64

        Default:
            Required

        Best-case:
            uint8 for speed, float32 for precision

    RETURNS
    -------
    None

    EXCEPTIONS
    ----------
    TypeError:
        If dtype is unsupported.
    """


    if not (
        frame.dtype == np.uint8
        or frame.dtype == np.uint16
        or np.issubdtype(frame.dtype, np.floating)
    ):
        raise TypeError(
            f"Unsupported dtype {frame.dtype}. "
            "Allowed: uint8, uint16, float32, float64."
        )


def temporal_median_background(
    video: Union[str, cv2.VideoCapture],
    *,
    num_samples: int = 25,
    seed: int | None = None
) -> np.ndarray:
    
    """
    Estimate background using pixel-wise temporal median.

    PARAMETERS
    ----------
    video : str | cv2.VideoCapture
        Description:
            Input video source.

        Units:
            File path or capture handle

        Default:
            Required

        Best-case:
            Stable static camera footage

    num_samples : int
        Description:
            Number of frames randomly sampled across the video.

        Min & Max:
            Min: 1
            Max: total number of frames

        Units:
            Frames

        Default:
            25

        Best-case:
            30–100 for stable background estimation

    seed : int | None
        Description:
            Random seed for reproducible frame sampling.

        Min & Max:
            Any integer or None

        Units:
            Random seed

        Default:
            None

        Best-case:
            Fixed value for debugging

    RETURNS
    -------
    background : np.ndarray
        Estimated background frame (uint8).

    EXCEPTIONS
    ----------
    ValueError, RuntimeError, IOError
    """

    start_time = time.time()

    try:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        if seed is not None:
            np.random.seed(seed)

        cap = _open_video(video)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError("Video contains no frames")

        frame_ids = np.random.randint(0, frame_count, size=num_samples)

        frames: list[np.ndarray] = []

        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret:
                continue

            gray = _ensure_grayscale_frame(frame)
            _validate_frame_dtype(gray)

            frames.append(gray.astype(np.float32))

        if len(frames) == 0:
            raise RuntimeError("No valid frames collected")

        background = np.median(frames, axis=0)
        background = np.clip(background, 0, 255).astype(np.uint8)

        end_time = time.time()
        print(
            f"{'temporal_median_background time:':<36}"
            f"{end_time - start_time:.4f} seconds"
        )

        return background

    except Exception as ex:
        print(f"[temporal_median_background ERROR] {ex}")
        raise


def subtract_background(
    frame: np.ndarray,
    background: np.ndarray,
    *,
    threshold: int = 30
) -> np.ndarray:
    
    """
    Subtract background and produce a binary foreground mask.

    PARAMETERS
    ----------
    frame : np.ndarray
        Description:
            Current grayscale frame.

        Min & Max:
            Same shape as background

        Units:
            Pixel intensity

        Default:
            Required

        Best-case:
            uint8 grayscale frame

    background : np.ndarray
        Description:
            Background reference image.

        Units:
            Pixel intensity

        Default:
            Required

    threshold : int
        Description:
            Absolute difference threshold.

        Min & Max:
            0–255

        Units:
            Pixel intensity

        Default:
            30

        Best-case:
            20–40 for thermal scenes

    RETURNS
    -------
    foreground_mask : np.ndarray
        Binary foreground mask (uint8).

    EXCEPTIONS
    ----------
    ValueError:
        Shape mismatch or invalid input.
    """

    frame = _ensure_grayscale_frame(frame)
    _validate_frame_dtype(frame)

    if background is None or background.size == 0:
        raise ValueError("Invalid background")

    if frame.shape != background.shape:
        raise ValueError("Frame and background shape mismatch")

    diff = cv2.absdiff(frame, background)
    _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return fg


def spatial_background_gaussian(
    image: Union[str, np.ndarray],
    *,
    ksize: int = 51,
    sigma: float = 0.0
) -> np.ndarray:
    
    """
    Estimate background using spatial Gaussian low-pass filtering.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Description:
            Input grayscale image.

        Units:
            Pixel intensity

        Default:
            Required

        Best-case:
            Static background image

    ksize : int
        Description:
            Gaussian kernel size.

        Min & Max:
            Min: 3
            Max: odd positive integer

        Units:
            Pixels

        Default:
            51

        Best-case:
            31–101 for background smoothing

    sigma : float
        Description:
            Gaussian standard deviation.

        Min & Max:
            ≥ 0

        Units:
            Pixels

        Default:
            0.0 (auto)

        Best-case:
            Auto (0.0)

    RETURNS
    -------
    residual : np.ndarray
        Foreground residual image (uint8).

    EXCEPTIONS
    ----------
    ValueError, IOError
    """

    start_time = time.time()

    try:
        if image is None:
            raise ValueError("Input image is None")

        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("Cannot read image")
        else:
            img = np.asarray(image)

        img = _ensure_grayscale_frame(img)
        _validate_frame_dtype(img)

        if ksize <= 0:
            raise ValueError("ksize must be > 0")

        if ksize % 2 == 0:
            ksize += 1

        background = cv2.GaussianBlur(
            img,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma
        )

        residual = cv2.absdiff(img, background)

        end_time = time.time()
        print(
            f"{'spatial_background_gaussian time:':<36}"
            f"{end_time - start_time:.4f} seconds"
        )

        return residual

    except Exception as ex:
        print(f"[spatial_background_gaussian ERROR] {ex}")
        raise


if __name__ == "__main__":

    # Spatial background
    residual = spatial_background_gaussian("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/image.png")
    cv2.imshow("Residual", residual)

    # Temporal background
    video_path = "/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/resized_video_640x360.avi"

    bg = temporal_median_background(video_path, num_samples=25, seed=42)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = subtract_background(gray, bg, threshold=30)

        cv2.imshow("Background", bg)
        cv2.imshow("Foreground Mask", fg)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
