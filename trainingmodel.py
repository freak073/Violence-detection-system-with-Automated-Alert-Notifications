import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to extract frames from video
def extract_frames_from_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = fps * frame_rate  # Extract frames every `frame_rate` seconds
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Load frames from all videos in a folder and assign labels
def load_data_from_folder(folder_path, label, frame_rate=1):
    frames = []
    labels = []
    
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        
        if video_path.endswith('.avi'):  # Check for .avi files
            video_frames = extract_frames_from_video(video_path, frame_rate)
            frames.extend(video_frames)
            labels.extend([label] * len(video_frames))
    
    return np.array(frames), np.array(labels)

# Load frames and labels from both "violence" and "non-violence" folders
violence_folder = 'dataset/train/violence'
non_violence_folder = 'dataset/train/non violence'

violence_frames, violence_labels = load_data_from_folder(violence_folder, 0)  # Label "0" for violence
non_violence_frames, non_violence_labels = load_data_from_folder(non_violence_folder, 1)  # Label "1" for non-violence

# Combine frames and labels from both classes
all_frames = np.concatenate([violence_frames, non_violence_frames], axis=0)
all_labels = np.concatenate([violence_labels, non_violence_labels], axis=0)

# Preprocess frames (resize and normalize)
def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        # Resize to 224x224 as VGG16 expects this input size
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        preprocessed_frames.append(frame_resized)
    return np.array(preprocessed_frames)

all_frames = preprocess_frames(all_frames)

# Load VGG16 pre-trained model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)  # 2 classes: "Violence" (0) and "Non-Violence" (1)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with categorical cross-entropy loss
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_frames, all_labels, test_size=0.2, random_state=42)

# Define ModelCheckpoint callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('violence_detection_model_best2.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[checkpoint])



# Plot training & validation accuracy and loss
def plot_training_history(history):
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the training and validation accuracy and loss
plot_training_history(history)