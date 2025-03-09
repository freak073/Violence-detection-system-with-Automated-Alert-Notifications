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
import telepot
import threading
from datetime import datetime

# ---------------------- Configuration ---------------------- #
ALERT_INTERVAL = 5  # seconds between successive alerts during continuous violence
MAX_EMAIL_ALERTS = 1       # maximum email alerts per continuous violence episode
MAX_TELEGRAM_ALERTS = 1    # maximum Telegram alerts per continuous violence episode

# ---------------------- Utility Functions ---------------------- #

def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_frame(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"violence_detected_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# ---------------------- Alert Functions ---------------------- #

def send_email(image_path):
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
        update_status("Email alert sent.")
    except Exception as e:
        print(f"Error sending email: {e}")
        update_status("Error sending email.")

def send_telegram_alert(image_path):
    try:
        bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
        alert_message = (
            f"\U0001F6A8 VIOLENCE ALERT \U0001F6A8\n"
            f"📍 LOCATION: Cam_1\n"
            f"🕒 TIME: {get_time()}"
        )
        bot.sendMessage(TELEGRAM_CHANNEL_ID, alert_message)
        with open(image_path, 'rb') as photo:
            bot.sendPhoto(TELEGRAM_CHANNEL_ID, photo, caption="\U0001F6A8 Violence Detected! \U0001F6A8")
        print("Telegram alert sent successfully.")
        update_status("Telegram alert sent.")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")
        update_status("Error sending Telegram alert.")

def trigger_buzzer():
    try:
        print("Buzzer triggered!")
        engine.say("Violence detected")
        engine.runAndWait()
        update_status("Buzzer triggered.")
    except Exception as e:
        print(f"Error triggering buzzer: {e}")
        update_status("Error triggering buzzer.")

def alert_actions(frame):
    """Trigger alert actions conditionally:
       - Email and Telegram alerts are sent only if their respective counters are below max.
       - The buzzer is triggered every time.
    """
    global email_alert_count, telegram_alert_count
    saved_frame_path = save_frame(frame)
    threads = []

    if email_alert_count < MAX_EMAIL_ALERTS:
        threads.append(threading.Thread(target=send_email, args=(saved_frame_path,), daemon=True))
        email_alert_count += 1
    else:
        print("Email alert limit reached for this episode.")

    if telegram_alert_count < MAX_TELEGRAM_ALERTS:
        threads.append(threading.Thread(target=send_telegram_alert, args=(saved_frame_path,), daemon=True))
        telegram_alert_count += 1
    else:
        print("Telegram alert limit reached for this episode.")

    # Always trigger the buzzer alert.
    threads.append(threading.Thread(target=trigger_buzzer, daemon=True))

    for thread in threads:
        thread.start()
# ---------------------- Model & Prediction ---------------------- #
def predict_frame_class(frame, model):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    prediction = model.predict(frame_resized)
    # Assuming that index 0 corresponds to Violence and index 1 to Non-Violence.
    return "Violence" if np.argmax(prediction) == 0 else "Non-Violence"

# ---------------------- Video Processing ---------------------- #
# Global variables to track the violence detection and alert counts.
violence_counter = 0
last_alert_time = None  # Timestamp (in seconds) of the last alert
email_alert_count = 0
telegram_alert_count = 0

def display_video_with_classification(video_path, model):
    global violence_counter, last_alert_time, email_alert_count, telegram_alert_count
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_class = predict_frame_class(frame, model)
        if frame_class == "Violence":
            violence_counter += 1
        else:
            # Reset counters when non-violence is detected.
            violence_counter = 0
            last_alert_time = None
            email_alert_count = 0
            telegram_alert_count = 0

        # If violence is detected for at least 10 consecutive frames,
        # trigger alert actions at a set interval.
        if violence_counter >= 10:
            current_time = time.time()
            if last_alert_time is None or (current_time - last_alert_time) >= ALERT_INTERVAL:
                last_alert_time = current_time
                alert_actions(frame)

        cv2.putText(frame, f"Class: {frame_class}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Video with Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Email & Telegram Configuration ---------------------- #

EMAIL_ADDRESS = "varunkpvkp2003@gmail.com"
EMAIL_PASSWORD = "oqeenudjtgvsrqld"
RECIPIENT_EMAIL = "rohankp24@gmail.com"

TELEGRAM_BOT_TOKEN = '7554916364:AAEkJVKbZCO748BRBajQngsMjFk4d7mnMS4'
TELEGRAM_CHANNEL_ID = '-1002287104880'

# ---------------------- Global Variables ---------------------- #

video_path = None
engine = pyttsx3.init()

# Load the trained model.
model = load_model('violence_detection_model_best.h5', compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------- GUI Setup ---------------------- #

root = tk.Tk()
root.title("Violence Detection using Deep Learning")
root.geometry("800x600")

# A thread-safe status update function that uses root.after.
def update_status(message):
    root.after(0, lambda: status_label.config(text="Status: " + message))

# Try to load and set a background image.
try:
    bg_image = Image.open("Security Cameras Installation Los Angeles.jpg")
    bg_image = bg_image.resize((800, 600), Image.ANTIALIAS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    background_label = tk.Label(root, image=bg_photo)
    background_label.place(relwidth=1, relheight=1)
except Exception as e:
    print(f"Error loading background image: {e}")

header = tk.Label(root, text="Violence Detection system", font=("Helvetica", 24, "bold"), bg="black", fg="white")
header.pack(pady=20)

def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi *.mp4 *.mov")])
    if video_path:
        messagebox.showinfo("Selected File", f"Selected Video: {video_path}")
        update_status("Video file selected.")

browse_button = tk.Button(root, text="Browse Video File", font=("Helvetica", 16), command=browse_video, bg="blue", fg="white")
browse_button.pack(pady=20)

def start_detection():
    global video_path, violence_counter, last_alert_time, email_alert_count, telegram_alert_count
    if not video_path:
        messagebox.showerror("Error", "Please select a video file first.")
        update_status("No video file selected.")
    else:
        # Reset the global variables for a new detection run.
        violence_counter = 0
        last_alert_time = None
        email_alert_count = 0
        telegram_alert_count = 0
        update_status("Starting detection...")
        threading.Thread(target=display_video_with_classification, args=(video_path, model), daemon=True).start()

detect_button = tk.Button(root, text="Detect", font=("Helvetica", 16), command=start_detection, bg="green", fg="white")
detect_button.pack(pady=20)

# Status label to show when alerts are sent.
status_label = tk.Label(root, text="Status: Idle", font=("Helvetica", 14), bg="black", fg="white")
status_label.pack(pady=20)

root.mainloop()
