from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
# Load the trained model
model = load_model('violence_detection_model_best.h5', compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Function to predict the class of each frame
def predict_frame_class(frame, model):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    prediction = model.predict(frame_resized)
    
    # Predict the class (0 = Violence, 1 = Non-Violence)
    if np.argmax(prediction) == 0:
        return "Violence"
    else:
        return "Non-Violence"

# Function to display video with predictions
def display_video_with_classification(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the class for the current frame
        frame_class = predict_frame_class(frame, model)

        # Add text to the frame indicating the classification
        cv2.putText(frame, f"Class: {frame_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame with the classification
        cv2.imshow("Video with Classification", frame)

        # Exit video display on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage: Predict and display for a new video
video_path = 'dataset/train/non violence/2gkuUVq1_0.avi'  # Replace with the actual video path
display_video_with_classification(video_path, model)
