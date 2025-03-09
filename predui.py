import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from PIL import Image, ImageTk
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time
import os
import pyttsx3
import tensorflow as tf

engine = pyttsx3.init()

# Load the trained model
model = load_model('violence_detection_model_best.h5', compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Email configuration
EMAIL_ADDRESS = "varunkpvkp2003@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "oqeenudjtgvsrqld"  # Replace with your password
RECIPIENT_EMAIL = "rohankp24@gmail.com"  # Replace with recipient email

# Initialize variables
violence_counter = 0
violence_detected = False

# Function to send an email with the image
def send_email(image_path):
    print("email sent")
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Violence Detected"

        with open(image_path, 'rb') as img_file:
            img_data = MIMEImage(img_file.read(), name=os.path.basename(image_path))
        msg.attach(img_data)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to save frame
def save_frame(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"violence_detected{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved: {filename}")
    return filename 

# Function to trigger the buzzer
def trigger_buzzer():
    print("Buzzer triggered!")
    engine.say("violence detected")
    engine.runAndWait()
    engine.say("violence detected")
    engine.runAndWait()

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
    global violence_counter, violence_detected
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict the class for the current frame
        frame_class = predict_frame_class(frame, model)

        if frame_class == "Violence":
            violence_counter += 1
        else:
            violence_counter = 0

        # Trigger action if violence detected for 4 seconds (assuming 30 FPS)
        if violence_counter >= 10 and not violence_detected:
            violence_detected = True
            saved_frame_path = save_frame(frame)
            send_email(saved_frame_path)
            trigger_buzzer()

        # Add text to the frame indicating the classification
        cv2.putText(frame, f"Class: {frame_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with the classification
        cv2.imshow("Video with Classification", frame)

        # Exit video display on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to browse for a video file
def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi *.mp4 *.mov")])
    if video_path:
        messagebox.showinfo("Selected File", f"Selected Video: {video_path}")

# Function to start detection
def start_detection():
    if not video_path:
        messagebox.showerror("Error", "Please select a video file first.")
    else:
        display_video_with_classification(video_path, model)

# Create the main Tkinter window
root = tk.Tk()
root.title("Violence Detection using Deep Learning")
root.geometry("800x600")

# Add background image
bg_image = Image.open("Security Cameras Installation Los Angeles.jpg")  # Replace with your background image file path
bg_image = bg_image.resize((800, 600), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(root, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Add header
header = tk.Label(root, text="Violence Detection using Deep Learning", font=("Helvetica", 24, "bold"), bg="black", fg="white")
header.pack(pady=20)

# Add browse button
browse_button = tk.Button(root, text="Browse Video File", font=("Helvetica", 16), command=browse_video, bg="blue", fg="white")
browse_button.pack(pady=20)

# Add detect button
detect_button = tk.Button(root, text="Detect", font=("Helvetica", 16), command=start_detection, bg="green", fg="white")
detect_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
