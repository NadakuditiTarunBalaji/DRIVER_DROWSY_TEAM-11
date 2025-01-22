
import cv2
import numpy as np
import mediapipe as mp
import csv
import time

"""
cv2 is the OpenCV library for computer vision tasks.
numpy is a library for numerical operations.
mediapipe is a library for machine learning solutions, particularly for face and hand tracking.
csv is a module for handling CSV files.
time is a module for time-related functions.
"""

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
"""
static_image_mode=False: Process video frames instead of static images.
max_num_faces=1: Detect only one face.
refine_landmarks=True: Refine the landmarks for better accuracy.
"""
# Landmark indices for eyes and mouth
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 81, 78, 13, 312, 308, 291, 402, 318, 324, 14, 88]

"""
LEFT_EYE_IDX contains the indices of landmarks for the left eye.
RIGHT_EYE_IDX contains the indices of landmarks for the right eye.
MOUTH_IDX contains the indices of landmarks for the mouth
"""
# Function to calculate Eye Aspect Ratio (EAR)
def EAR(eye_landmarks):
    """ EAR that takes a single argument eye_landmarks, 
    which takes a list or array of coordinates representing the landmarks of an eye."""
    point1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    point2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    """calculates the Euclidean distance between the landmarks at indices 1,5 and 2,4. 
    This distance represents the vertical distance between two points on the eye."""
    distance = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    """ This distance represents the horizontal distance between two points on the eye."""
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    """The EAR is a measure used to determine if the eye is open or closed."""
    return ear_aspect_ratio

# Function to calculate Mouth Aspect Ratio (MAR)
def MAR(mouth_landmarks):
    """ takes a single argument mouth_landmarks,
     which is expected to be a list or array of coordinates representing the landmarks of the mouth."""
    point = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
    """ calculates the Euclidean distance (norm) between the landmarks at indices 0 and 6.
    This distance represents the horizontal distance between two points on the mouth."""
    point1 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    point2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
    """ calculates the vertical distance between the landmarks at indices 2,10 and 4,8."""
    Ypoint = (point1 + point2) / 2.0
    #calculates the average vertical distance between the two points
    mouth_aspect_ratio = Ypoint / point
    #calculates the ratio of the vertical distance to the horizontal distance
    return mouth_aspect_ratio

# Open webcam and collect EAR, MAR with labels (0: Not Drowsy, 1: Drowsy)
def collect_data():
    # Open webcam
    webcamera = cv2.VideoCapture(0)
    # Create an empty list to store the collected data
    data = []
    # Instructions for data collection
    print("Press 'q' to quit.")
    print("Press 'd' when you feel drowsy and 'n' when you are not drowsy to label the data.")
    # Loop to collect data
    while True:
        # Read frame from webcam
        ret, frame = webcamera.read()
        # Break the loop if the frame is not read
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame using Mediapipe Face Mesh
        result = face_mesh.process(rgb_frame)
# Check if face landmarks are detected
        if result.multi_face_landmarks:
            # Loop through each detected face
            for face_landmarks in result.multi_face_landmarks:
                # Extract landmarks as numpy array
                landmarks = np.array([[p.x, p.y] for p in face_landmarks.landmark])

                # Extract left and right eye landmarks
                left_eye = landmarks[LEFT_EYE_IDX]
                right_eye = landmarks[RIGHT_EYE_IDX]
                
                # Calculate EAR for eyes
                leftEAR = EAR(left_eye)
                rightEAR = EAR(right_eye)
                # Average EAR of both eyes
                ear = (leftEAR + rightEAR) / 2.0
                
                # Extract mouth landmarks and calculate MAR
                mouth = landmarks[MOUTH_IDX]
                mouEAR = MAR(mouth)

                # Show EAR and MAR on the frame
                #green color for not drowsy
                #red color for drowsy
                """(function) def putText(
                        img: MatLike,
                        text: str,
                        org: Point,
                        fontFace: int,
                        fontScale: float,
                        color: Scalar,
                        thickness: int = ...,
                        lineType: int = ...,
                        bottomLeftOrigin: bool = ..."""
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mouEAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for key presses once per frame
        # Press 'd' for Drowsy, 'n' for Not Drowsy, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            label = 1  # Drowsy
            print(f"Collected data: EAR={ear}, MAR={mouEAR}, Label=Drowsy")
            data.append([ear, mouEAR, label])  # Collect EAR, MAR, and label
        elif key == ord("n"):
            label = 0  # Not Drowsy
            print(f"Collected data: EAR={ear}, MAR={mouEAR}, Label=Not Drowsy")
            data.append([ear, mouEAR, label])  # Collect EAR, MAR, and label
        elif key == ord("q"):
            break
# Show the frame with EAR, MAR, and labels
        cv2.imshow("Frame", frame)
# Release the webcam and close all windows
    webcamera.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Check if data has been collected before writing to CSV
    if data:
        print(f"Saving {len(data)} rows of data to drowsiness_data.csv")
        # Save collected data to a CSV file
        with open("daata.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["EAR", "MAR", "Label"])  # Column names
            writer.writerows(data)
    else:
        print("No data collected!")

collect_data()