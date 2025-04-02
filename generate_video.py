from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from flask_cors import CORS
import cv2.face as cvface
import pygame  # Alarm sound
from reportlab.pdfgen import canvas
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image


app = Flask(__name__)
CORS(app)

# Load YOLOv3 model for weapon detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
weapon_classes = ["gun", "knife", "sword"]

# Alarm 
alarm_sound = "alarm.mp3"
alarm_active = False
pygame.mixer.init()

def play_alarm():
    """Plays the alarm sound."""
    pygame.mixer.music.load(alarm_sound)
    pygame.mixer.music.play()

# Email settings
OWNER_EMAIL = "n.sharam9561@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = "n.sharma9561@gmail.com"
EMAIL_PASSWORD = "tccq wdzl hlnn qymg"

# Shared sentiment text
current_text = ""

# Load known authorized faces
authorized_faces = {}
recognizer = cvface.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_authorized_faces():
    global authorized_faces, recognizer
    face_dir = "authorized_faces/"
    images, labels = [], []
    label_map = {}
    label_id = 0

    for filename in os.listdir(face_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(face_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label_id)
            label_map[label_id] = name
            label_id += 1

    if images and labels:
        recognizer.train(images, np.array(labels))

    authorized_faces = label_map

load_authorized_faces()

@app.route('/')
def entry_page():
    return render_template('entry_page.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/start_monitoring')
def start_monitoring():
    return redirect(url_for('dashboard'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_stream():
    global alarm_active, current_text
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        authorized_person = False

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)
            if label in authorized_faces and confidence < 70:
                authorized_person = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Authorized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        weapon_detected = False
        detected_weapon = ""
        weapon_confidence = 0

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3 and classes[class_id] in weapon_classes:
                    weapon_detected = True
                    detected_weapon = classes[class_id]
                    weapon_confidence = confidence
                    print(f"Detected weapon: {detected_weapon} with confidence: {weapon_confidence}")

        if weapon_detected and not authorized_person and not alarm_active:
            print("Weapon detected and unauthorized person!")
            alarm_active = True
            play_alarm()
            threading.Thread(target=send_notification, args=(current_text, frame.copy())).start()
            threading.Thread(target=generate_pdf_report, args=(current_text, frame.copy(), weapon_detected, weapon_confidence, authorized_person, current_text)).start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    global alarm_active
    alarm_active = False
    pygame.mixer.music.stop()
    return "Alarm stopped successfully"

@app.route('/update_sentiment', methods=['POST'])
def update_sentiment():
    global current_text
    current_text = request.form.get('text', "Everything seems fine")
    return f"Sentiment text updated to: {current_text}"

def send_notification(threat_text, frame):
    try:
        image_path = "threat.jpg"
        cv2.imwrite(image_path, frame)

        msg = MIMEMultipart()
        msg['Subject'] = "Security Alert: Weapon Detected"
        msg['From'] = EMAIL_USERNAME
        msg['To'] = OWNER_EMAIL

        body = MIMEText("Intruder with weapon detected, call security.")
        msg.attach(body)

        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, OWNER_EMAIL, msg.as_string())
        os.remove(image_path)
    except Exception as e:
        print(f"Failed to send email with image: {e}")

def generate_pdf_report(threat_text, frame, weapon_detected, confidence, authorized_person, sentiment_text):
    """Generates a professional PDF report with a tabular format and an intruder image."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f"intruder_{timestamp}.jpg"
    pdf_path = f"intrusion_report_{timestamp}.pdf"

    # Save the intruder image
    cv2.imwrite(image_path, frame)

    # Create the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("<b><font size=16 color='red'>🚨 Security Alert Report</font></b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Timestamp
    timestamp_text = Paragraph(f"<b>📅 Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
    elements.append(timestamp_text)
    elements.append(Spacer(1, 12))

    # Table Data
    data = [
        ["🔍 Threat Detected", threat_text],
        ["🔫 Weapon Detected", "Yes" if weapon_detected else "No"],
        ["📈 Weapon Confidence", f"{confidence:.2f}"],
        ["🛑 Authorized Person", "Yes" if authorized_person else "No"],
        ["📊 Sentiment Analysis", sentiment_text],
    ]

    # Create Table
    table = Table(data, colWidths=[180, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Add Image if Available
    if os.path.exists(image_path):
        elements.append(Paragraph("<b>📸 Captured Intruder Image:</b>", styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Image(image_path, width=250, height=200))
    else:
        elements.append(Paragraph("⚠ No image available!", styles["Normal"]))

    # Build the PDF
    doc.build(elements)
    print(f"✅ PDF Report saved: {pdf_path}")

    # Try opening the file automatically (Windows only)
    try:
        os.startfile(pdf_path)
    except Exception:
        print("❗ Unable to open PDF automatically. Open manually.")
if __name__ == '__main__':
    app.run(debug=True)
