import cv2 as cv
import numpy as np

def lucas_kanade_method(video_path):
    """
    Implements the Lucas-Kanade Sparse Optical Flow method to track features
    across video frames and displays the result with motion vectors drawn.
    """
    # Read the video 
    cap = cv.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at '{video_path}'")
        return

    # --- Parameters ---
    # Parameters for ShiTomasi corner detection (Initial Feature Detection)
    feature_params = dict(
        maxCorners=100,  # Maximum number of features to track
        qualityLevel=0.3, # Minimum acceptable quality of an image corner
        minDistance=7,    # Minimum Euclidean distance between features
        blockSize=7
    )
 
    # Parameters for Lucas Kanade optical flow (Tracking Algorithm)
    lk_params = dict(
        winSize=(15, 15), # Size of the search window at each pyramid level
        maxLevel=2,       # Number of pyramid levels
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03), # Termination criteria
    )
 
    # Create random colors for drawing the tracks (100 rows, 3 columns for BGR)
    color = np.random.randint(0, 255, (100, 3))
 
    # Take the first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame.")
        cap.release()
        return

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    
    # p0 holds the coordinates of the initial features/corners
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # --- Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        # Calculate Optical Flow (The core of Lucas-Kanade)
        # p1: Next features coordinates
        # st: Status array (1 if flow found, 0 otherwise)
        # err: Error between images
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
 
        # Select good points (where st == 1)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
 
            # Draw the tracks (lines connecting old and new positions)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Draw the line (track)
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                
                # Draw the new point (feature location)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
 
            # Combine the original frame with the mask containing the tracks
            img = cv.add(frame, mask)
 
            # Display the result
            cv.imshow('Lucas-Kanade Optical Flow', img)
            
            k = cv.waitKey(30) & 0xFF
            if k == 27:  # Press 'ESC' to exit
                break
 
            # Update the previous frame and features for the next iteration
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
 
    # Cleanup
    cap.release()
    cv.destroyAllWindows()

# --- Execution ---
# Note: Ensure 'Videos/video.mp4' is a valid path to your video file
lucas_kanade_method("Videos/Motion.webm ")