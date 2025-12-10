import cv2 as cv
import numpy as np

def calculate_temporal_average_frame(video_path: str, is_thermal: bool = False) -> np.ndarray | None:
    """
    Calculates the temporal average of all frames in a video sequence.

    This technique is used to create a static background image by averaging 
    pixel intensities over time, effectively reducing random noise and 
    blurring out moving objects.

    Args:
        video_path (str): The file path to the input video.
        is_thermal (bool): Set to True if the input video is a single-channel 
                           thermal sequence. Skips BGR-to-Grayscale conversion. 
                           Defaults to False (assuming standard color video).

    Returns:
        np.ndarray or None: A single 2D NumPy array representing the 
                            averaged frame, scaled and converted to 8-bit (uint8), 
                            or None if the video cannot be processed.
    """
    # 1. Video Capture and Robust Error Handling
    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        print(f"ERROR: Could not open video file at '{video_path}'")
        return None
    
    # 2. Get Video Properties (Dimensions and Frame Count)
    try:
        # Get properties using the dedicated constants
        W = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        H = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    except Exception as e:
        print(f"ERROR: Failed to retrieve video properties: {e}")
        video.release()
        return None

    print(f"Processing Video: H={H}, W={W}, Total Frames={frame_count}")

    # 3. Initialize Accumulator
    # Use float64 to prevent integer overflow when summing many frames.
    # The shape is (H, W) because we ensure the frame is single-channel before summing.
    sum_frames = np.zeros((H, W), dtype=np.float64)
    frames_processed = 0

    # 4. Accumulation Loop
    while True:
        ret, frame = video.read()
        if not ret:
            # Breaks if frame read fails (end of file or read error)
            break
        
        # Determine Frame Format based on 'is_thermal' flag
        if is_thermal:
            # For thermal (already single-channel or need specific handling)
            if frame.ndim == 3:
                # If OpenCV reads it as BGR (3D), assume all channels are identical 
                # and extract one channel for simplicity and correct dimensionality.
                current_frame = frame[:, :, 0]
            else:
                current_frame = frame
        else:
            # Standard procedure: Convert BGR frame to Grayscale (single channel)
            current_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Accumulate: Convert current_frame to float64, then ADD it to the sum
        sum_frames += current_frame.astype(np.float64)
        
        frames_processed += 1
        
        # Safety break if frame count is zero or unreliable
        if frame_count > 0 and frames_processed >= frame_count:
             break


    # Release the video object immediately after processing
    video.release()

    # 5. Final Calculation and Post-Processing
    if frames_processed == 0:
        print("WARNING: No frames were successfully processed.")
        return None
        
    # Calculate the average
    average_frame = sum_frames / frames_processed
    
    # Normalize and Convert for Display: 
    # Normalization (NORM_MINMAX) scales the float range (0 to ~255) 
    # to the full 0-255 8-bit integer range for optimal visualization.
    avg_norm = cv.normalize(
        average_frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    
    print(f"Successfully processed {frames_processed} frames. Returning averaged frame.")
    return avg_norm

# --- Main Execution Block (Industrial Standard) ---
if __name__ == '__main__':
    VIDEO_FILE = 'Videos/video.mp4'
    
    # 1. Calculate Average Frame
    # Pass 'is_thermal=True' if using a thermal video.
    averaged_image = calculate_temporal_average_frame(VIDEO_FILE, is_thermal=False)
    
    # 2. Display Result
    if averaged_image is not None:
        cv.imshow("Temporal Frame Average (Industrial Standard)", averaged_image)
        cv.waitKey(0)
        
    cv.destroyAllWindows()