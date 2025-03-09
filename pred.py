import os
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np

def load_violence_detection_model(model_path):
    """
    Load the pre-trained violence detection model with error handling.
    
    Args:
        model_path (str): Path to the .h5 model file
    
    Returns:
        model: Compiled Keras model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_frame_class(frame, model):
    """
    Predict the class of a single frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        model (keras.Model): Trained violence detection model
    
    Returns:
        str: Predicted class (Violence/Non-Violence)
    """
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    prediction = model.predict(frame_resized)
    
    return "Violence" if np.argmax(prediction) == 0 else "Non-Violence"

def display_video_with_classification(video_path, model, delay=25):
    """
    Display video with real-time violence classification.
    
    Args:
        video_path (str): Path to the input video file
        model (keras.Model): Trained violence detection model
        delay (int, optional): Delay between frames in milliseconds. Defaults to 25.
    """
    # Validate input video path
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the class for the current frame
        frame_class = predict_frame_class(frame, model)

        # Add text to the frame indicating the classification
        cv2.putText(frame, 
                    f"Class: {frame_class}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255) if frame_class == "Violence" else (0, 255, 0), 
                    2)
        
        # Display the frame with the classification
        cv2.imshow("Video Violence Classification", frame)

        # Exit video display on 'q' key press
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to parse arguments and run video classification.
    """
    parser = argparse.ArgumentParser(description='Violence Detection in Videos')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='violence_detection_model_best.h5', 
                        help='Path to trained model file')
    parser.add_argument('--delay', type=int, default=25, 
                        help='Delay between frames in milliseconds')
    
    args = parser.parse_args()

    # Load the model
    model = load_violence_detection_model(args.model)
    
    if model is not None:
        # Display video with classification
        display_video_with_classification(args.video, model, args.delay)
    else:
        print("Model loading failed. Cannot proceed with classification.")

if __name__ == '__main__':
    main()