import cv2
import numpy as np

def thermal_bilateral_filter(
    src: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0
) -> np.ndarray:
    """
    Apply bilateral filtering on a THERMAL single-channel image.

    Parameters:
        src (ndarray): Thermal image (must be single-channel). *
        diameter (int): Neighborhood diameter (default=5)
        sigma_color (float): Intensity-domain sigma (default=25.0)
        sigma_space (float): Spatial-domain sigma (default=25.0)

    Returns:
        ndarray: Bilaterally filtered thermal image.

    Raises:
        ValueError: If invalid arguments or multi-channel image provided.
    """

    # --- Input check ---
    if src is None:
        raise ValueError("Source image is NONE")

    if not isinstance(src, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")

    # --- Strict thermal rule: single-channel only ---
    if src.ndim == 2:
        pass
    elif src.ndim == 3 and src.shape[2] == 1:
        src = src[..., 0]  # squeeze single channel
    else:
        raise ValueError(
            f"Bilateral filter requires SINGLE-CHANNEL thermal image. Got shape={src.shape}"
        )

    # --- Parameter checks ---
    if diameter <= 0:
        raise ValueError("diameter must be a positive integer")

    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("sigma_color and sigma_space must be > 0")

    # --- Bilateral Filtering ---
    outimg = cv2.bilateralFilter(src, diameter, sigma_color, sigma_space)
    return outimg



if __name__ == "__main__":
    # Example usage
    img = cv2.imread("/home/user1/learning/Testing/NoiseReduction/Inputs/SingleChannel.png", cv2.IMREAD_UNCHANGED)

    result = thermal_bilateral_filter(img)

    cv2.imshow("Thermal Bilateral Filter", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
