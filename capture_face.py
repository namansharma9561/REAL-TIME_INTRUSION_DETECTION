import cv2
import os

name = input("Enter the authorized person's name: ").strip()
if not name:
    print("⚠️ Name cannot be empty! Please enter a valid name.")
    exit(1)

print(f"✅ Capturing face data for: {name}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

save_dir = "authorized_faces/"
os.makedirs(save_dir, exist_ok=True)

count = 0
while count < 5:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image. Check your camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("⚠️ No face detected. Try again.")
        continue

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, face)
        print(f"✅ Saved: {img_path}")
        count += 1

cap.release()
print("✅ Face data collection completed!")
