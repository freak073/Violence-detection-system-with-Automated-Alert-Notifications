import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from PIL import Image, ImageTk
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import os
import pyttsx3
import telepot
from datetime import datetime

def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load the trained model
model = load_model('violence_detection_model_best.h5', compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

EMAIL_ADDRESS = "varunkpvkp2003@gmail.com"
EMAIL_PASSWORD = "oqeenudjtgvsrqld"
RECIPIENT_EMAIL = "rohankp24@gmail.com"
TELEGRAM_BOT_TOKEN = '7554916364:AAEkJVKbZCO748BRBajQngsMjFk4d7mnMS4'
TELEGRAM_CHANNEL_ID = '-1002287104880'

violence_counter = 0
violence_detected = False

# Email Alert Function
def send_email(image_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "⚠️ Violence Alert - Immediate Attention Required ⚠️"
        with open(image_path, 'rb') as img_file:
            img_data = MIMEImage(img_file.read(), name=os.path.basename(image_path))
        msg.attach(img_data)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Telegram Alert Function
def send_telegram_alert(image_path):
    try:
        bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
        bot.sendMessage(TELEGRAM_CHANNEL_ID, f"\U0001F6A8 VIOLENCE ALERT \U0001F6A8\n📍 LOCATION: Cam_1\n🕒 TIME: {get_time()}")
        with open(image_path, 'rb') as photo:
            bot.sendPhoto(TELEGRAM_CHANNEL_ID, photo, caption="\U0001F6A8 Violence Detected! Immediate Action Required! \U0001F6A8")
        print("✅ Telegram alert sent successfully.")
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")

# Save Detected Frame
def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"violence_detected_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# Trigger Alarm
def trigger_buzzer():
    print("🚨 Buzzer triggered!")
    engine.say("Violence detected. Immediate action required!")
    engine.runAndWait()

# Predict Frame Class
def predict_frame_class(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    prediction = model.predict(frame_resized)
    confidence = prediction[0][0]  # Probability of violence
    return ("Violence", confidence) if confidence > 0.5 else ("Non-Violence", 1 - confidence)

# Process Video in a Separate Thread
def process_video():
    global violence_counter, violence_detected
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_class, confidence = predict_frame_class(frame)
        if frame_class == "Violence":
            violence_counter += 1
        else:
            violence_counter = 0
        if violence_counter >= 10 and not violence_detected:
            violence_detected = True
            saved_frame_path = save_frame(frame)
            send_email(saved_frame_path)
            send_telegram_alert(saved_frame_path)
            trigger_buzzer()
        cv2.putText(frame, f"Class: {frame_class} ({confidence:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Violence Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    if not video_path:
        messagebox.showerror("Error", "Please select a video file first.")
    else:
        threading.Thread(target=process_video, daemon=True).start()

def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi *.mp4 *.mov")])
    if video_path:
        messagebox.showinfo("Selected File", f"Selected Video: {video_path}")

# GUI Setup
root = tk.Tk()
root.title("Violence Detection using Deep Learning")
root.geometry("800x600")
root.configure(bg='black')

header = tk.Label(root, text="Violence Detection System", font=("Helvetica", 24, "bold"), bg="black", fg="white")
header.pack(pady=20)

browse_button = ttk.Button(root, text="Browse Video File", command=browse_video)
browse_button.pack(pady=20)

detect_button = ttk.Button(root, text="Start Detection", command=start_detection)
detect_button.pack(pady=20)

root.mainloop()
