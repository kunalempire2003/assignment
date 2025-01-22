import streamlit as st
import cv2
from deepface import DeepFace

# Load pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app title with styled header
st.set_page_config(page_title='People Counting & Emotion Detection', page_icon='📓', layout='wide')
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,200..1000;1,200..1000&family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap');
        .stApp {
            background-color: #f0f2f6;
            font-family: Roboto, sans-serif;
        }
        .title-text {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }
        .info-box {
            background-color: #d1e7dd;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: #155724;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">Assignment</p>', unsafe_allow_html=True)

# Sidebar for additional info
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
st.sidebar.markdown("Developed by Kunal")
st.sidebar.info("Adjust the confidence level to fine-tune the face detection accuracy.")

# Information box
st.markdown('<div class="info-box">Live emotion detection using webcam feed. Stay still for better accuracy.</div>', unsafe_allow_html=True)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Streamlit display for video frames
frame_window = st.image([], use_container_width=True)


while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error accessing webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop face from frame
        face_region = frame[y:y + h, x:x + w]

        # Analyze emotion using DeepFace
        result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display face count and emotion text
    cv2.putText(frame, f'People Count: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update Streamlit image with the new frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

cap.release()
