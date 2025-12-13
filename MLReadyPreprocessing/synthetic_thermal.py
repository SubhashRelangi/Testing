import cv2
import numpy as np
import time
from typing import Union

def synthetic_thermal(
    image: Union[str, np.ndarray],
    *,
    # --- Thermal generation (proxy model) ---
    temp_min: float = 20.0,
    temp_max: float = 80.0,
    thermal_smooth_sigma: float = 7.0,

    # --- Sensor effects ---
    diffusion_sigma: float = 2.5,
    drift_max: float = 3.0,
    noise_std: float = 4.0,
    hot_pixel_count: int = 200,
    gain_mean: float = 1.0,
    gain_std: float = 0.015,
    offset_mean: float = 0.0,
    offset_std: float = 1.5,

    # --- Optional augmentation ---
    rotate_deg: float | None = None,
    flip_code: int | None = None,
    scale: float | None = None,
) -> np.ndarray:
    
    """
    Generate a synthetic thermal image using a structured thermal imaging pipeline.

    This function simulates the formation of a thermal image by sequentially modeling:
        1. Thermal content generation (proxy model, non-radiometric)
        2. Thermal sensor artifacts
        3. Optional geometric data augmentation

    IMPORTANT:
    ----------
    This function DOES NOT produce radiometrically accurate thermal images.
    The generated output is suitable for:
        - Algorithm development
        - Robustness testing
        - Data augmentation
        - Deep learning training (non-temperature tasks)

    It MUST NOT be used for:
        - Absolute temperature measurement
        - Medical thermography
        - Sensor calibration or certification
        - Safety-critical thermal analysis

    PIPELINE
    --------
    Base image / scene
        → Thermal generation (proxy temperature field)
        → Synthetic thermal frame (new thermal content)
        → Sensor artifacts simulation
        → Optional augmentation
        → Output image

    THERMAL GENERATION (Proxy Model)
    --------------------------------
    - The input image is normalized and mapped to a synthetic temperature range
      [temp_min, temp_max].
    - Gaussian smoothing simulates thermal diffusion within materials.
    - The resulting temperature field is mapped to an 8-bit sensor intensity range.

    NOTE:
    -----
    This stage approximates thermal behavior but does NOT model:
        - Emissivity
        - Atmospheric transmission
        - Reflected apparent temperature
        - Planck-based radiance

    SENSOR EFFECTS SIMULATED
    -----------------------
    - Thermal diffusion blur (optical + detector spread)
    - Horizontal thermal drift (temporal or spatial bias)
    - Additive Gaussian sensor noise
    - Random hot pixels
    - Pixel-wise gain and offset non-uniformity (NUC-like behavior)

    OPTIONAL AUGMENTATION
    --------------------
    - Rotation (degrees)
    - Flip (OpenCV flip codes)
    - Scaling (isotropic)

    These operations affect geometry only and do NOT change the underlying
    synthetic thermal content.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Base image or file path. Must be single-channel or convertible to grayscale.

    temp_min : float
        Minimum synthetic temperature (proxy).
        Units: arbitrary (°C-like scale)
        Default: 20.0

    temp_max : float
        Maximum synthetic temperature (proxy).
        Units: arbitrary (°C-like scale)
        Default: 80.0

    thermal_smooth_sigma : float
        Gaussian sigma used to smooth the temperature field.
        Units: pixels
        Typical range: 5.0 – 10.0

    diffusion_sigma : float
        Additional Gaussian blur to simulate sensor diffusion.
        Units: pixels
        Typical range: 1.5 – 4.0

    drift_max : float
        Maximum horizontal drift added across the image.
        Units: intensity levels
        Typical range: 1.0 – 5.0

    noise_std : float
        Standard deviation of additive Gaussian noise.
        Units: intensity levels
        Typical range: 2.0 – 6.0

    hot_pixel_count : int
        Number of randomly injected hot pixels.
        Recommended: < 0.1% of total pixels

    gain_mean : float
        Mean pixel gain (NUC).
        Default: 1.0

    gain_std : float
        Gain variation standard deviation.
        Typical range: 0.01 – 0.03

    offset_mean : float
        Mean offset value.
        Default: 0.0

    offset_std : float
        Offset variation standard deviation.
        Typical range: 0.5 – 2.0

    rotate_deg : float | None
        Optional rotation angle in degrees.
        If None, rotation is skipped.

    flip_code : int | None
        OpenCV flip code:
            0  → vertical
            1  → horizontal
           -1  → both
        If None, no flip is applied.

    scale : float | None
        Optional isotropic scaling factor.
        If None, scaling is skipped.

    RETURNS
    -------
    synthetic_thermal : np.ndarray
        Synthetic thermal image.
        - dtype: uint8
        - shape: (H, W)
        - single-channel grayscale

    """

    start_time = time.time()

    # --------------------------------------------------
    # 1. Load & validate image
    # --------------------------------------------------
    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, str):
        base = cv2.imread(image, cv2.IMREAD_COLOR)
        if base is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        base = np.asarray(image)

    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("Expected BGR image (H×W×3)")

    h, w, _ = base.shape

    # --------------------------------------------------
    # 2. BGR → LUMINANCE (NO BLUR)
    # --------------------------------------------------
    ycrcb = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[..., 0].astype(np.float32)

    lum_norm = cv2.normalize(
        luminance, None, 0.0, 1.0, cv2.NORM_MINMAX
    )

    # --------------------------------------------------
    # 3. Temperature field (NO BLUR)
    # --------------------------------------------------
    temp_field = temp_min + lum_norm * (temp_max - temp_min)

    # Radiometric nonlinearity (important)
    radiance = (temp_field + 273.15) ** 4

    thermal_frame = cv2.normalize(
        radiance, None, 0.0, 255.0, cv2.NORM_MINMAX
    )

    # --------------------------------------------------
    # 4. SENSOR EFFECTS
    # --------------------------------------------------

    # ONE optical blur only (small!)
    thermal_frame = cv2.GaussianBlur(
        thermal_frame, (0, 0), 1.2   # <= 1.5
    )


    # Horizontal drift
    drift = np.linspace(0, drift_max, w, dtype=np.float32)
    thermal_frame += drift

    # Sensor noise
    noise = np.random.normal(0.0, noise_std, (h, w)).astype(np.float32)
    thermal_frame += noise

    # Hot pixels
    if hot_pixel_count > 0:
        ys = np.random.randint(0, h, hot_pixel_count)
        xs = np.random.randint(0, w, hot_pixel_count)
        thermal_frame[ys, xs] = 255.0

    # Gain / offset non-uniformity
    gain = np.random.normal(gain_mean, gain_std, (h, w)).astype(np.float32)
    offset = np.random.normal(offset_mean, offset_std, (h, w)).astype(np.float32)
    thermal_frame = thermal_frame * gain + offset

    # --------------------------------------------------
    # 5. Optional augmentation
    # --------------------------------------------------
    if rotate_deg is not None:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
        thermal_frame = cv2.warpAffine(
            thermal_frame, M, (w, h), flags=cv2.INTER_LINEAR
        )

    if flip_code is not None:
        thermal_frame = cv2.flip(thermal_frame, flip_code)

    if scale is not None and scale > 0:
        thermal_frame = cv2.resize(
            thermal_frame, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------
    # 6. Clip & convert
    # --------------------------------------------------
    thermal_frame = np.clip(thermal_frame, 0, 255).astype(np.uint8)

    print(
        f"{'synthetic_thermal execution time:':<36}"
        f"{time.time() - start_time:.4f} s"
    )

    return thermal_frame

if __name__ == "__main__":

    synthetic = synthetic_thermal("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/image.png")

    original = cv2.imread("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/image.png")

    # stackedresult = np.hstack([original, synthetic])

    # cv2.imshow("Original | synthetic", stackedresult)


    thermal_color = cv2.applyColorMap(
        synthetic,
        cv2.COLORMAP_INFERNO  
    )

    cv2.imshow("Thermal (visualized)", thermal_color)

    cv2.imshow("Original Image", original)
    cv2.imshow("Synthetic Image", synthetic)
    cv2.imwrite("/home/user1/learning/Testing/MLReadyPreprocessing/Outputs/synthetic.png", synthetic)
    cv2.imwrite("/home/user1/learning/Testing/MLReadyPreprocessing/Outputs/thermal_Color.png", thermal_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

