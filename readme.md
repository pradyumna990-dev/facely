# Facely AI - Face Recognition Attendance System

Facely AI is a Python-based application for automated student attendance using **face recognition** with blink verification to prevent spoofing.  
It uses a webcam to detect and recognize registered faces in real time, logs attendance to CSV files, and provides a GUI for easy interaction.

---

## Features

- **Face Registration**: Encode faces from student images.
- **Blink Detection**: Ensures live presence, preventing photo-based spoofing.
- **Attendance Logging**: Saves daily attendance with timestamp and status.
- **GUI Interface**: Built with Tkinter for starting/stopping sessions and viewing logs.
- **Face Position Alerts**: Warns if a face is too close, far, or at the frame border.
- **Multiple Safeguards**: Uses dlib landmarks and EAR (Eye Aspect Ratio) for blink validation.

---

## Project Structure

├── encode_faces.py # Script to register faces from images
├── face.py # Main attendance system with GUI
├── images/ # Student images (named as Roll.jpg)
├── students.csv # Student details (Roll, Name, Branch, Sem)
├── encodings.dat # Saved face encodings
├── attendance/ # Attendance CSV logs
└── shape_predictor_68_face_landmarks.dat # dlib facial landmark model

---

## Installation

### 1. Clone the Repository

```bash
git https://github.com/pradyumna990-dev/facely.git
cd facely
```

### Install dependencis

pip install face_recognition opencv-python pandas numpy dlib Pillow scipy

run above script on your terminal

### Download Shape Predictor

Download shape_predictor_68_face_landmarks.dat from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Extract it into your project directory.
