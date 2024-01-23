import tensorflow as tf
import cv2
import numpy as np
# Load the trained model (Assuming you have already trained and saved the model)
model = tf.keras.models.load_model('Fire_detection.h5')

# Define a function for preprocessing a frame
def preprocess_frame(frame):
    # Resize the frame to the model's input size (224x224)
    frame = cv2.resize(frame, (224, 224))
    # Convert the pixel values to the range [0, 1]
    frame = frame / 255.0
    return frame

# Define the input video file
video_path = 'fire1.mov'

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save the processed video
output_path = 'path/to/your/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'vid')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Expand dimensions to match the model input shape
    processed_frame = np.expand_dims(processed_frame, axis=0)

    # Make predictions using the model
    predictions = model.predict(processed_frame)
    predicted_label = np.argmax(predictions[0])

    # Post-process the predicted label as needed
    # (e.g., overlay label on the frame, etc.)
    if predicted_label==0:
        print("fire detected")
        break
    print(predicted_label)
    # Write the processed frame to the output video
    out.write(frame)

    # Display the processed frame (optional, for real-time processing)
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
