from typing import Any, Dict, Optional, Tuple
import cv2 as cv
import numpy as np
import time


# =====================================================
# 1. FREQUENCY MASK GENERATOR (MODULAR & FLEXIBLE)
# =====================================================
def create_frequency_mask(
    D: np.ndarray,
    mode: str = "highpass",
    **params: Any
) -> np.ndarray:
    """
    Creates a frequency-domain mask for FFT-based filtering of thermal or
    grayscale images. Supports low-pass, high-pass, band-pass, Gaussian, and
    Butterworth filters.

    -------------------------------------------------------------------------
    PARAMETER SPECIFICATIONS
    -------------------------------------------------------------------------

    D : np.ndarray  
        Description: Distance matrix from FFT center (Euclidean distances).  
        Min & Max: Must be ≥ 1×1; values range from 0 → sqrt(H² + W²).  
        Units: Pixels (in the *frequency* domain).  
        Default: Required.  
        Best case: Generated via meshgrid centered at (H/2, W/2).

    mode : str  
        Description: Filter type determining mask shape.  
        Allowed Values:
            "lowpass", "highpass", "bandpass",
            "gaussian_lp", "gaussian_hp",
            "butterworth_lp", "butterworth_hp"
        Units: String identifier.  
        Default: "highpass".  
        Best case:
            - "highpass" for general edge enhancement  
            - "gaussian_hp" for smooth HF boost on thermal frames  
            - "butterworth_hp" for sharper rolloff  

    **params : dict  
        Description: Mode-specific parameters controlling mask behavior.  
        Units: Mixed (pixel radii, sigma, order).  
        Default: Depends on mode.  

        -------------------------
        radius : int or float  
        -------------------------
            Description: Cutoff frequency for LP/HP/BW filters.  
            Min & Max: 1 → min(H, W).  
            Units: Pixels (FFT radius).  
            Default: 50.  
            Best case: 20–60 for thermal edge enhancement.

        -------------------------
        inner : int  
        -------------------------
            Description: Inner radius for band-pass filters.  
            Min & Max: 1 → outer−1.  
            Units: Pixels.  
            Default: 20.  
            Best case: Narrow bands: 10–30.

        -------------------------
        outer : int  
        -------------------------
            Description: Outer radius for band-pass filters.  
            Min & Max: inner+1 → min(H, W).  
            Units: Pixels.  
            Default: 60.  
            Best case: 40–80 for thermal mid-frequency extraction.

        -------------------------
        sigma : float  
        -------------------------
            Description: Spread parameter in Gaussian filters.  
            Min & Max: 0.1 → 200.0  
            Units: Pixels (frequency-domain).  
            Default: 20.0  
            Best case:
                LP: 15–40 (smooth blurring)  
                HP: 10–30 (smooth sharpening)

        -------------------------
        order : int  
        -------------------------
            Description: Filter steepness for Butterworth filters.  
            Min & Max: 1 → 10  
            Units: Exponent.  
            Default: 2  
            Best case: 2–4 for stable HF emphasis in thermal images.

    -------------------------------------------------------------------------
    PROCESS SUMMARY:
      - Creates a binary or soft mask depending on mode.
      - Masks are used in frequency domain to amplify or suppress components.
      - Gaussian & Butterworth masks provide smooth transitions (no ringing).
    -------------------------------------------------------------------------

    RETURNS
    -------
    mask : np.ndarray (float32, H×W)
        Frequency mask with values 0.0–1.0 suitable for FFT filtering.
    """

    h, w = D.shape
    mask = np.zeros((h, w), np.float32)

    if mode == "lowpass":
        R: int = int(params.get("radius", 50))
        mask[D <= R] = 1.0

    elif mode == "highpass":
        R: int = int(params.get("radius", 50))
        mask[D > R] = 1.0

    elif mode == "bandpass":
        r1: int = int(params.get("inner", 20))
        r2: int = int(params.get("outer", 60))
        mask[(D >= r1) & (D <= r2)] = 1.0

    elif mode == "gaussian_lp":
        sigma: float = float(params.get("sigma", 20.0))
        mask = np.exp(-(D**2) / (2.0 * sigma**2))

    elif mode == "gaussian_hp":
        sigma: float = float(params.get("sigma", 20.0))
        mask = 1.0 - np.exp(-(D**2) / (2.0 * sigma**2))

    elif mode == "butterworth_lp":
        R: float = float(params.get("radius", 50.0))
        order: int = int(params.get("order", 2))
        mask = 1.0 / (1.0 + (D / R) ** (2.0 * order))

    elif mode == "butterworth_hp":
        R: float = float(params.get("radius", 50.0))
        order: int = int(params.get("order", 2))
        mask = 1.0 - (1.0 / (1.0 + (D / R) ** (2.0 * order)))

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask.astype(np.float32)


# =====================================================
# 2. ENHANCEMENT FUNCTION (COMPLETE IMPROVED VERSION)
# =====================================================
def HighPassEdge(
    image: np.ndarray,
    *,
    filter_mode: str = "highpass",
    radius: int = 50,
    gain: float = 1.5,
    sigma: float = 20.0,
    order: int = 2,
    fft_flag: int = cv.DFT_COMPLEX_OUTPUT,
    norm_dst: Optional[np.ndarray] = None,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
) -> np.ndarray:
    
    """
    Fourier-based high-frequency enhancement for thermal or grayscale images.
    Produces a contrast-boosted, edge-enhanced uint8 output.

    -------------------------------------------------------------------------
    PARAMETER SPECIFICATIONS
    -------------------------------------------------------------------------

    image : np.ndarray  
        Description: Input single-channel grayscale or thermal image.  
        Min & Max: ≥ 1×1 pixels, must be 2D.  
        Units: Pixel intensity (uint8/uint16/float32/float64).  
        Default: Required.  
        Best case: Clean uint8 thermal frame with good signal/noise ratio.

    filter_mode : str  
        Description: Frequency mask type (e.g., "highpass", "gaussian_hp", "butterworth_hp").  
        Min & Max: Any supported mask name defined by create_frequency_mask().  
        Units: String identifier.  
        Default: "highpass".  
        Best case: "highpass" for general sharpening; "gaussian_hp" for smooth HF rolloff.

    radius : int  
        Description: High-pass cutoff radius determining which frequencies pass.  
        Min & Max: 1 → min(image_width, image_height).  
        Units: Pixels (frequency-domain distance).  
        Default: 50.  
        Best case: 20–60 for thermal edge enhancement; too large removes structure.

    gain : float  
        Description: Magnification factor for high-frequency components.  
        Min & Max: 1.0 → 10.0 (practically).  
        Units: Scalar multiplier.  
        Default: 1.5.  
        Best case: 1.5–3.0 for thermal edges without overshoot.

    sigma : float  
        Description: Spread parameter for Gaussian frequency masks.  
        Min & Max: 0.1 → 200.0  
        Units: Pixels (in frequency domain).  
        Default: 20.0.  
        Best case: 10–30 for thermal images (smooth attenuation).

    order : int  
        Description: Order of the Butterworth filter (controls steepness).  
        Min & Max: 1 → 10.  
        Units: Exponent.  
        Default: 2.  
        Best case: 2–4 for stable, clean HF enhancement.

    fft_flag : int  
        Description: OpenCV cv.dft flag controlling complex output layout.  
        Min & Max: Any valid OpenCV DFT flag.  
        Units: OpenCV enum (cv.DFT_*).  
        Default: cv.DFT_COMPLEX_OUTPUT.  
        Best case: cv.DFT_COMPLEX_OUTPUT (provides 2-channel complex result).

    norm_dst : Optional[np.ndarray]  
        Description: Optional destination buffer for cv.normalize().  
        Min & Max: Must match output shape or be None.  
        Units: ndarray or None.  
        Default: None.  
        Best case: None (OpenCV allocates internally).

    norm_alpha : float  
        Description: Minimum intensity after normalization.  
        Min & Max: 0.0 → 255.0  
        Units: Pixel intensity.  
        Default: 0.0  
        Best case: 0.0 for full-range visualization.

    norm_beta : float  
        Description: Maximum intensity after normalization.  
        Min & Max: 1.0 → 255.0  
        Units: Pixel intensity.  
        Default: 255.0  
        Best case: 255.0 for maximum contrast; lower values compress dynamic range.

    norm_type : int  
        Description: Normalization method (cv.NORM_MINMAX, cv.NORM_INF, etc.).  
        Min & Max: Any valid cv.normalize type.  
        Units: OpenCV enum.  
        Default: cv.NORM_MINMAX.  
        Best case: cv.NORM_MINMAX for thermal frequency results.

    -------------------------------------------------------------------------
    PROCESS SUMMARY:
      - Converts to float32
      - Performs FFT → shifts spectrum
      - Builds HF mask → applies gain → recombines with LF
      - Inverse FFT
      - Magnitude computation
      - Normalization to uint8 output (0–255)
    -------------------------------------------------------------------------

    RETURNS
    -------
    final_img : np.ndarray  
        uint8 image (H×W) with enhanced edges/sharpness.
    """
       
    start_time = time.time()
    # validation
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim != 2:
        raise ValueError("Input must be single-channel (grayscale).")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError("Unsupported image dtype. Use uint8/uint16/float32/float64.")
    if gain < 1.0:
        raise ValueError("gain must be >= 1.0")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    try:
        # convert to float32 for FFT
        float_image: np.ndarray = np.float32(image)

        # DFT
        F: np.ndarray = cv.dft(float_image, flags=fft_flag)
        F_shift: np.ndarray = np.fft.fftshift(F)

        # distance matrix
        h, w = float_image.shape
        cx, cy = w // 2, h // 2
        u = np.arange(w)
        v = np.arange(h)
        U, V = np.meshgrid(u, v)
        D = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)

        # mask
        mask = create_frequency_mask(
            D,
            mode=filter_mode,
            radius=radius,
            sigma=sigma,
            order=order,
            inner=max(1, radius // 2),
            outer=radius,
        )
        mask2 = cv.merge([mask, mask])

        # apply and enhance
        HF = F_shift * mask2
        HF_enhanced = HF * float(gain)
        LF = F_shift * (1.0 - mask2)
        merged = LF + HF_enhanced

        # inverse
        merged_shift = np.fft.fftshift(merged)
        complex_image = cv.idft(merged_shift)
        magnitude_image = cv.magnitude(complex_image[:, :, 0], complex_image[:, :, 1])

        # normalize
        out_image = cv.normalize(magnitude_image, norm_dst, norm_alpha, norm_beta, norm_type)

        end_time = time.time() 
        total_time = end_time - start_time 
        print(f"[HighPassEdge] Time Taken: {total_time:.6f} seconds")
        
        return out_image.astype(np.uint8)

    except Exception as exc:
        raise RuntimeError(f"HighPassEdge failed: {exc}")

       



# =====================================================
# 3. EXECUTION EXAMPLE
# =====================================================
if __name__ == "__main__":
    src_path = "StructureModule/Inputs/Input.jpg"
    image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read {src_path}")

    enhanced = HighPassEdge(
        image,
        filter_mode="gaussian_hp",   # str
        radius=40,                   # int
        gain=2.0,                    # float
        sigma=25.0,                  # float
        order=3,                     # int
        fft_flag=cv.DFT_COMPLEX_OUTPUT,  # int (OpenCV constant)
        norm_alpha=0.0,              # float
        norm_beta=255.0,             # float
        norm_type=cv.NORM_MINMAX     # int (OpenCV constant)
    )

    cv.imshow("Original", image)
    cv.imshow("Enhanced", enhanced)
    result = np.hstack([image, enhanced])
    cv.imwrite("/home/user1/learning/Testing/StructureModule/Outputs/HighPassEdgeEnhancement.png", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
