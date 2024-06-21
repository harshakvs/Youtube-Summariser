import cv2

# Load the pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # path to the deploy file
    "res10_300x300_ssd_iter_140000.caffemodel"  # path to the pre-trained model
)

# Open the video file
video_path = 'D:\Youtube-Summariser\Youtube-Summariser\\video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

best_face_frame = None
best_face_area = 0

# Iterate through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Resize the frame to 300x300 (required input size for the model)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the model
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Iterate through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Check if the confidence is above a certain threshold
        if confidence > 0.8:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (x, y, w, h) = box.astype(int)

            face_area = w * h

            # Update best face if a larger frontal face is found
            if face_area > best_face_area:
                best_face_frame = frame[y:y+h, x:x+w]
                best_face_area = face_area

    # Check if the best face is found
    if best_face_frame is not None:
        # Save the frame with the best face as a PNG image
        cv2.imwrite('Best_Face_Frame_Capture.png', best_face_frame)
        print("Frame with the best frontal face saved as Best_Face_Frame_Capture.png.")
        break

# Release the video capture object
cap.release()
