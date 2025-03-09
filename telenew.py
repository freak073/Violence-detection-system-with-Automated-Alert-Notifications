import os
import cv2
import time
import threading
import numpy as np
import pyttsx3
import telepot  # Telegram bot library
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Folder to store uploaded videos
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables for alerts and frame storage
latest_prediction = "Non-Violence"
buzzer_triggered = False
telegram_alert_triggered = False
last_frame = None
consecutive_violence_count = 0
VIOLENCE_THRESHOLD = 4   # Number of consecutive detections needed to trigger alert

# Load the trained model
model = load_model('violence_detection_model_best.h5', compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
 
# Telegram bot details
TELEGRAM_BOT_TOKEN = '7554916364:AAEkJVKbZCO748BRBajQngsMjFk4d7mnMS4'
TELEGRAM_CHANNEL_ID = '-4508985837'  # Corrected Channel ID

def predict_frame_class(frame, model):
    """Resize, normalize, and predict the class for a given frame."""
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    prediction = model.predict(frame_resized)
    return "Violence" if np.argmax(prediction) == 0 else "Non-Violence"

def trigger_buzzer():
    """Trigger the buzzer using pyttsx3 to announce 'violence detected'."""
    global buzzer_triggered
    print("Buzzer triggered!")
    engine = pyttsx3.init()
    engine.say("Violence detected")
    engine.runAndWait()
    engine.say("Violence detected")
    engine.runAndWait()
    time.sleep(5)
    buzzer_triggered = False

def get_time():
    """Return the current time as a formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def send_telegram_alert(image_path):
    """Send a Telegram message and photo alert when violence is detected."""
    global telegram_alert_triggered
    print("Sending Telegram alert...")
    try:
        bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
        bot.sendMessage(TELEGRAM_CHANNEL_ID, f"🚨 VIOLENCE ALERT 🚨\n📍 LOCATION: Cam_1\n🕒 TIME: {get_time()}")
        with open(image_path, 'rb') as photo:
            bot.sendPhoto(TELEGRAM_CHANNEL_ID, photo, caption="🚨 Violence Detected! 🚨")
        print("Telegram alert sent successfully.")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")
    time.sleep(60)  # Cooldown period before next alert
    telegram_alert_triggered = False

def generate_frames(video_path):
    """Process video frame-by-frame, predict violence, and stream."""
    global latest_prediction, last_frame
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        frame_class = predict_frame_class(frame, model)
        latest_prediction = frame_class

        # Overlay prediction text on the frame
        cv2.putText(frame, f"Class: {frame_class}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    video_filename = request.args.get('video_filename', default=None, type=str)
    return render_template('index1.html', video_filename=video_filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    if 'video' not in request.files:
        flash('No video part in the request.')
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        video_filename = file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)
        return redirect(url_for('index', video_filename=video_filename))
    else:
        flash('Unsupported file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS))
        return redirect(url_for('index'))

@app.route('/video_feed/<video_filename>')
def video_feed(video_filename):
    """Stream video frames."""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Get the latest prediction and trigger alerts if needed."""
    global latest_prediction, buzzer_triggered, telegram_alert_triggered, last_frame, consecutive_violence_count

    if latest_prediction == "Violence":
        consecutive_violence_count += 1
        print(f"Consecutive violence count: {consecutive_violence_count}")
        if consecutive_violence_count >= VIOLENCE_THRESHOLD:
            if not buzzer_triggered:
                buzzer_triggered = True
                threading.Thread(target=trigger_buzzer).start()
            if not telegram_alert_triggered and last_frame is not None:
                telegram_alert_triggered = True
                image_path = "alert.jpg"
                cv2.imwrite(image_path, last_frame)
                threading.Thread(target=send_telegram_alert, args=(image_path,)).start()
    else:
        consecutive_violence_count = 0

    return jsonify({"prediction": latest_prediction})

if __name__ == '__main__':
    app.run(debug=True)
