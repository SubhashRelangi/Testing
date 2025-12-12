import cv2
import numpy as np

def thermal_median_blur(src: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply median blur on a THERMAL single-channel image.

    Parameters:
        src (ndarray): Thermal image (must be single-channel). *
        ksize (int): Positive, odd kernel size. *

    Returns:
        ndarray: Blurred thermal image.

    Raises:
        ValueError: If invalid arguments or multi-channel image received.
    """

    # ------ Basic validation ------
    if src is None:
        raise ValueError("Source image is NONE")

    if not isinstance(src, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")

    # ------ Strict thermal rule ------
    if src.ndim == 2:
        pass  # OK
    elif src.ndim == 3 and src.shape[2] == 1:
        src = src[..., 0]  # squeeze
    else:
        raise ValueError("Thermal median filter requires SINGLE-CHANNEL image")

    # ------ ksize validation ------
    if ksize <= 1 or (ksize % 2) == 0:
        raise ValueError("ksize must be a positive ODD number (>1)")

    # ------ Apply median blur ------
    out_image = cv2.medianBlur(src, ksize)

    return out_image


if __name__ == "__main__":
    src_path = "/home/user1/learning/Testing/NoiseReduction/Inputs/SingleChannel.png"

    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

    ksize = 3
    res = thermal_median_blur(img, ksize)

    cv2.putText(res, f"ksize:{ksize}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 1)

    cv2.imshow("Thermal Median Blur", res)
    cv2.imwrite("outputs/median_thermal.png", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
