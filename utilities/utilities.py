import cv2
import os

# Define the model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")  # Ensure this folder exists in your project

# Define the correct file paths for the model
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Check if the model files exist
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
    raise FileNotFoundError("ERROR: Model files not found. Ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' are inside the 'models/' folder.")


def start_camera():
    cap = cv2.VideoCapture(1)  # 0 is usually the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("iPhone Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def detect_face():
    """Detects faces in real-time and draws an oval around them."""

    # Load the deep learning-based face detector model (DNN)
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        h, w = frame.shape[:2]  # Get frame height and width

        # Preprocess the image for deep learning model
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

        # Run the model to get face detections
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # Only consider high-confidence detections
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")

                # Compute face width & height
                face_width = x2 - x
                face_height = y2 - y

                # Compute ellipse parameters for oval shape
                center = (x + face_width // 2, y + face_height // 2)
                axes = (face_width // 2, int(face_height * 0.6))  # 60% of face height for better fit
                angle = 0

                # Draw the oval (ellipse)
                cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 0), 3)

        cv2.imshow("Face Detection with Ovals", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()