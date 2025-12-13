import cv2
import numpy as np
import time
import random
from typing import Union


def _load_thermal_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load a thermal image from path or ndarray and perform basic validation.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {image}")
    else:
        img = np.asarray(image)

    if img.size == 0:
        raise ValueError("Input image is empty.")

    return img


def _ensure_single_channel(img: np.ndarray) -> np.ndarray:
    """
    Enforce single-channel thermal image.
    """
    if img.ndim > 2:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={img.shape}")

    return img


def _validate_thermal_dtype(img: np.ndarray) -> None:
    """
    Validate supported thermal image dtypes.
    """
    if not (
        img.dtype == np.uint8
        or img.dtype == np.uint16
        or np.issubdtype(img.dtype, np.floating)
    ):
        raise TypeError(
            f"Unsupported dtype {img.dtype}. "
            "Allowed: uint8, uint16, float32, float64."
        )



def thermal_flip(
    image: Union[str, np.ndarray],
    *,
    flip_code: int = 1
) -> np.ndarray:
    
    """
    Flip a thermal (single-channel) image using OpenCV flip codes.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Description:
            Input thermal image or file path.

        Min / Max:
            - Minimum size: 1 × 1 pixels
            - Maximum size: system memory bound

        Units:
            Pixel intensity

        Supported dtypes:
            uint8, uint16, float32, float64

        Default:
            Required (no default)

        Best-case:
            uint8 / uint16 single-channel thermal image

    flip_code : int
        Description:
            Specifies the flip direction.

        Allowed values:
            1   → Horizontal flip (left ↔ right)
            0   → Vertical flip (top ↔ bottom)
           -1   → Both axes (180° rotation)

        Units:
            Discrete code (OpenCV convention)

        Default:
            1

        Best-case:
            1 (horizontal) for ML augmentation

    RETURNS
    -------
    out_image : np.ndarray
        Flipped thermal image (same shape and dtype as input).

    EXCEPTIONS
    ----------
    ValueError:
        - Invalid flip_code
        - Input image is None / empty
        - Image is not 2D

    FileNotFoundError:
        - Image path cannot be read

    TypeError:
        - Unsupported dtype
    """

    start_time = time.time()

    try:
        if flip_code not in (-1, 0, 1):
            raise ValueError("flip_code must be -1, 0, or 1")

        img = _load_thermal_image(image)
        img = _ensure_single_channel(img)
        _validate_thermal_dtype(img)

        out_image = cv2.flip(img, flip_code)

        end_time = time.time()
        print(f"{'thermal_flip execution time:':<36}{end_time - start_time:.4f} seconds")

        return out_image

    except Exception as ex:
        print(f"[thermal_flip ERROR] {ex}")
        raise

def thermal_rotate(
    image: Union[str, np.ndarray],
    *,
    angle: float,
    scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101
) -> np.ndarray:
    
    """
    Rotate a thermal (single-channel) image.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    angle : float
        Description:
            Rotation angle in degrees.
            Positive values rotate counter-clockwise.

        Min / Max:
            No hard limit (practically -360 to +360)

        Units:
            Degrees

        Default:
            Required (no default)

        Best-case:
            Small angles (±15°) for augmentation

    scale : float
        Description:
            Isotropic scaling factor applied during rotation.

        Min / Max:
            > 0

        Units:
            Unitless scale factor

        Default:
            1.0

        Best-case:
            1.0 (no scaling)

    interpolation : int
        Description:
            Interpolation method used by OpenCV during rotation.

        Allowed values:
            cv2.INTER_NEAREST
            cv2.INTER_LINEAR
            cv2.INTER_CUBIC
            cv2.INTER_AREA

        Units:
            OpenCV enum

        Default:
            cv2.INTER_LINEAR

        Best-case:
            cv2.INTER_LINEAR (balanced for thermal images)

    border_mode : int
        Description:
            Pixel extrapolation method for borders.

        Allowed values:
            cv2.BORDER_CONSTANT
            cv2.BORDER_REPLICATE
            cv2.BORDER_REFLECT
            cv2.BORDER_REFLECT_101

        Units:
            OpenCV enum

        Default:
            cv2.BORDER_REFLECT_101

        Best-case:
            cv2.BORDER_REFLECT_101 (avoids artificial cold borders)

    RETURNS
    -------
    out_image : np.ndarray
        Rotated thermal image (same shape and dtype as input).
    """

    start_time = time.time()

    try:
        if scale <= 0:
            raise ValueError("scale must be > 0")

        img = _load_thermal_image(image)
        img = _ensure_single_channel(img)
        _validate_thermal_dtype(img)

        h, w = img.shape
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, float(angle), float(scale))

        out_image = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=interpolation,
            borderMode=border_mode
        )

        end_time = time.time()
        print(f"{'thermal_rotate execution time:':<36}{end_time - start_time:.4f} seconds")

        return out_image

    except Exception as ex:
        print(f"[thermal_rotate ERROR] {ex}")
        raise

def add_gaussian_noise(
    image: Union[str, np.ndarray],
    *,
    mean: float = 0.0,
    std: float = 25.0
) -> np.ndarray:
    
    """
    Add Gaussian noise to a thermal (single-channel) image.

    Parameters
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    mean : float
        Mean of the Gaussian noise.
        Units: pixel intensity
        Default: 0.0

    std : float
        Standard deviation of the Gaussian noise.
        Units: pixel intensity
        Default: 25.0

    Returns
    -------
    out_image : np.ndarray
        Noisy thermal image (same shape and dtype as input).
    """

    start_time = time.time()

    try:
        img = _load_thermal_image(image)
        img = _ensure_single_channel(img)
        _validate_thermal_dtype(img)

        orig_dtype = img.dtype

        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        img_f = img.astype(np.float32)
        noisy_f = img_f + noise

        if orig_dtype == np.uint8:
            noisy_f = np.clip(noisy_f, 0, 255)
            out_image = noisy_f.astype(np.uint8)
        elif orig_dtype == np.uint16:
            noisy_f = np.clip(noisy_f, 0, 65535)
            out_image = noisy_f.astype(np.uint16)
        else:
            min_val, max_val = img_f.min(), img_f.max()
            out_image = np.clip(noisy_f, min_val, max_val).astype(orig_dtype)

        end_time = time.time()
        print(f"{'add_gaussian_noise execution time:':<36}{end_time - start_time:.4f} seconds")

        return out_image

    except Exception as ex:
        print(f"[add_gaussian_noise ERROR] {ex}")
        raise

def add_salt_pepper_noise(
    image: Union[str, np.ndarray],
    *,
    density: float
) -> np.ndarray:
    
    """
    Add salt-and-pepper noise to a thermal (single-channel) image.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    density : float
        Fraction of pixels to corrupt with noise.

        Min / Max:
            0.0 ≤ density ≤ 1.0

        Units:
            Ratio (unitless)

        Best-case:
            0.001 – 0.01 for augmentation

    RETURNS
    -------
    noisy_image : np.ndarray
        Thermal image with salt & pepper noise applied.
    """

    start_time = time.time()

    try:
        if not (0.0 <= density <= 1.0):
            raise ValueError("density must be in range [0, 1]")

        img = _load_thermal_image(image)
        img = _ensure_single_channel(img)
        _validate_thermal_dtype(img)

        noisy_image = img.copy()
        h, w = img.shape
        num_pixels = int(h * w * density)

        if img.dtype == np.uint8:
            salt, pepper = 255, 0
        elif img.dtype == np.uint16:
            salt, pepper = 65535, 0
        else:
            salt, pepper = img.max(), img.min()

        for _ in range(num_pixels):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            noisy_image[y, x] = salt if random.random() < 0.5 else pepper

        end_time = time.time()
        print(f"{'add_salt_pepper_noise execution time:':<36}{end_time - start_time:.4f} seconds")

        return noisy_image

    except Exception as ex:
        print(f"[add_salt_pepper_noise ERROR] {ex}")
        raise


if __name__ == "__main__":

    # fliped = thermal_flip("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/input.jpg", flip_code=0)

    # originalImage = cv2.imread("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/input.jpg", cv2.IMREAD_GRAYSCALE)
    
    # outputStacked = np.hstack([originalImage, fliped])

    # cv2.imshow("Output Flip", outputStacked)
    # cv2.imwrite("MLReadyPreprocessing/Outputs/Flip_image.png", outputStacked)

    noise = add_salt_pepper_noise("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/input.jpg", density=0.3)

    originalImage = cv2.imread("/home/user1/learning/Testing/MLReadyPreprocessing/Inputs/input.jpg", cv2.IMREAD_GRAYSCALE)
    
    outputStacked = np.hstack([originalImage, noise])

    cv2.imshow("Output noise", outputStacked)
    cv2.imwrite("MLReadyPreprocessing/Outputs/add_salt_pepper_noise_image.png", outputStacked)

    cv2.waitKey(0)
    cv2.destroyAllWindows()