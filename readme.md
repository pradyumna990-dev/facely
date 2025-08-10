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

## Uses

1. Prepare Student data

- Create a folder named images/
- Add student images named as Roll.jpg (e.g., 101.jpg)
- Create students.csv with columns:
  Roll,Name,Branch,Sem
  101,John Doe,CSE,5
  102,Jane Smith,IT,3

## Run Python face.py to start application

From the GUI:

- Register Faces: Runs face encoding
- Start Attendance: Begins webcam session.
- Stop Session: Ends attendance logging.

## Attendance is saved in attendance/ as:

Roll,Name,Branch,Sem,Timestamp,Status
101,John Doe,CSE,5,09:30:15,Present

## Requirements

- Python 3.7+
- Webcam
- dlib shape predictor model
- Required Python packages (see Installation)

## Notes

- Ensure good lighting and frontal face view for better accuracy.
- Blink detection requires visible eyes.
- The system assumes one face per student image in images/ folder.

## License

This project is open-source. You can modify and distribute it under your own terms.
