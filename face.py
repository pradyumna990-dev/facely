import face_recognition
import cv2
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import dlib
from scipy.spatial import distance as dist
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk

# --- CONFIGURATION ---
ENCODING_FILE_PATH = 'encodings.dat'
STUDENT_DETAILS_CSV = 'students.csv'
IMAGES_DIR = 'images'
ATTENDANCE_DIR = 'attendance'
FONT = cv2.FONT_HERSHEY_DUPLEX

# Path to dlib's pre-trained facial landmark predictor
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
# Check if the file exists
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"[ERROR] '{SHAPE_PREDICTOR_PATH}' not found. Please download it from:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("And extract it to your project directory. Exiting.")
    exit()

# --- BLINK DETECTION CONSTANTS ---
# Eye Aspect Ratio (EAR) constants for blinking detection
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 2

# dlib eye landmark indices
LEFT_EYE_START = 36
LEFT_EYE_END = 42
RIGHT_EYE_START = 42
RIGHT_EYE_END = 48

# --- BORDER AND DISTANCE DETECTION CONSTANTS ---
# Face bounding box distance from the edge of the frame
BORDER_THRESHOLD = 50
# Percentage of frame area that a face should occupy (min/max)
FACE_AREA_MIN_THRESHOLD = 0.03
FACE_AREA_MAX_THRESHOLD = 0.65

# --- HELPER FUNCTIONS ---
def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for blinking detection."""
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    if C == 0: return 0 # Avoid division by zero
    ear = (A + B) / (2.0 * C)
    return ear

def get_landmark_array(landmarks, start, end):
    """Converts dlib landmarks to a NumPy array for easy manipulation."""
    return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in range(start, end)])

def is_face_unusually_sized(box, frame_width, frame_height):
    """Checks if the face is too close, too far, or at the border of the frame."""
    top, right, bottom, left = box
    face_area = (right - left) * (bottom - top)
    frame_area = frame_width * frame_height
    
    # Check for border proximity
    if (top < BORDER_THRESHOLD or bottom > frame_height - BORDER_THRESHOLD or
        left < BORDER_THRESHOLD or right > frame_width - BORDER_THRESHOLD):
        return "at_border"

    # Check for unusual size
    if frame_area > 0:
        face_area_ratio = face_area / frame_area
        if face_area_ratio < FACE_AREA_MIN_THRESHOLD:
            return "too_far"
        elif face_area_ratio > FACE_AREA_MAX_THRESHOLD:
            return "too_close"
            
    return None
    
def mark_attendance(roll_no, student_details_df, status, log_widget):
    """Marks attendance for a student and saves it to a daily CSV file."""
    try:
        student = student_details_df[student_details_df['Roll'] == roll_no].iloc[0]
        name = student['Name']
        branch = student['Branch']
        sem = student['Sem']
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if not os.path.exists(ATTENDANCE_DIR):
            os.makedirs(ATTENDANCE_DIR)
            
        attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance-{date_str}.csv")
        
        write_header = not os.path.exists(attendance_file)

        with open(attendance_file, 'a', newline='') as f:
            if write_header:
                f.write("Roll,Name,Branch,Sem,Timestamp,Status\n")
            
            f.write(f"{roll_no},{name},{branch},{sem},{time_str},{status}\n")
        
        log_message = f"{name} present\n"
        print(log_message, end='')
        if log_widget:
            log_widget.config(state=tk.NORMAL)
            log_widget.insert(tk.END, log_message)
            log_widget.see(tk.END)
            log_widget.config(state=tk.DISABLED)
        return log_message

    except IndexError:
        log_message = f"[WARNING] Roll number {roll_no} found but not in '{STUDENT_DETAILS_CSV}'.\n"
        print(log_message, end='')
        if log_widget:
            log_widget.config(state=tk.NORMAL)
            log_widget.insert(tk.END, log_message)
            log_widget.see(tk.END)
            log_widget.config(state=tk.DISABLED)
    except Exception as e:
        log_message = f"[ERROR] An error occurred while marking attendance: {e}\n"
        print(log_message, end='')
        if log_widget:
            log_widget.config(state=tk.NORMAL)
            log_widget.insert(tk.END, log_message)
            log_widget.see(tk.END)
            log_widget.config(state=tk.DISABLED)

class AttendanceApp:
    def __init__(self, window, window_title="Facely AI - Attendance System"):
        self.window = window
        self.window.title(window_title)

        self.video_capture = None
        self.running = False
        
        self.blink_counter = {}

        self.setup_ui()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.window)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_frame = tk.LabelFrame(main_frame, text="Live Camera Feed", padx=5, pady=5)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.btn_register = tk.Button(controls_frame, text="Register Faces", command=self.register_faces, width=20)
        self.btn_register.pack(pady=5)
        self.btn_start_cam = tk.Button(controls_frame, text="Start Attendance", command=self.start_attendance, width=20)
        self.btn_start_cam.pack(pady=5)
        self.btn_stop = tk.Button(controls_frame, text="Stop Session", command=self.stop_attendance, state=tk.DISABLED, width=20)
        self.btn_stop.pack(pady=5)
        
        log_label_frame = tk.LabelFrame(controls_frame, text="Status & Attendance Log", padx=5, pady=5)
        log_label_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_label_frame, wrap=tk.WORD, width=40, height=20, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Called when the user closes the window."""
        self.stop_attendance()
        self.window.destroy()

    def load_resources(self):
        """Loads face encodings and dlib predictor, returns False if a file is missing."""
        try:
            with open(ENCODING_FILE_PATH, "rb") as f:
                data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_ids = data["ids"]
            
            self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
            self.student_details_df = pd.read_csv(STUDENT_DETAILS_CSV, dtype={'Roll': str})
            
            if self.student_details_df.empty:
                messagebox.showerror("Error", f"The student details file '{STUDENT_DETAILS_CSV}' is empty.")
                return False
            
            return True
            
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"A required file was not found: {e.filename}. Please run 'Register Faces' first.")
            return False
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", f"No data or columns found in '{STUDENT_DETAILS_CSV}'.")
            return False
        except RuntimeError:
            messagebox.showerror("Error", f"Failed to load '{SHAPE_PREDICTOR_PATH}'. Check file path.")
            return False

    def start_attendance(self):
        """Starts the attendance system using the webcam."""
        if self.running:
            messagebox.showinfo("Info", "Attendance session is already running.")
            return

        if not self.load_resources():
            return
        
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam stream.")
            return
        
        self.start_session()
        
    def start_session(self):
        """Common initialization for both webcam and video sessions."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, f"[INFO] Attendance session started. Please blink to confirm your presence.\n")
        self.log_text.config(state=tk.DISABLED)

        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_register.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        self.running = True
        self.process_video_stream()
        
    def stop_attendance(self):
        """Stops the attendance session."""
        if not self.running:
            return
        
        self.running = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
        
        self.btn_start_cam.config(state=tk.NORMAL)
        self.btn_register.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, "\n[INFO] Attendance session stopped.\n")
        self.log_text.config(state=tk.DISABLED)

    def process_video_stream(self):
        """Handles the main logic for face recognition, attendance, and GUI updates."""
        if not self.running:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, "\n[INFO] Webcam stream ended. Stopping session.\n")
            self.log_text.config(state=tk.DISABLED)
            self.stop_attendance()
            return
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # We process the frame at a smaller size for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_locations_scaled = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

        if not hasattr(self, 'already_marked_rolls'): self.already_marked_rolls = set()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations_scaled, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            if True in matches:
                first_match_index = matches.index(True)
                roll_no = self.known_face_ids[first_match_index]
                
                try:
                    student_name = self.student_details_df[self.student_details_df['Roll'] == roll_no]['Name'].iloc[0]
                except IndexError:
                    student_name = "Unregistered"
                
                if roll_no not in self.already_marked_rolls:
                    unusual_size_status = is_face_unusually_sized((top, right, bottom, left), frame_width, frame_height)
                    
                    if unusual_size_status:
                        if unusual_size_status == "at_border":
                            name = f"{student_name} (Move closer to center)"
                            self.log_text.config(state=tk.NORMAL)
                            self.log_text.insert(tk.END, f"[ALERT] {student_name}'s face is too close to the border.\n")
                            self.log_text.config(state=tk.DISABLED)
                        elif unusual_size_status == "too_far":
                            name = f"{student_name} (Move closer)"
                            self.log_text.config(state=tk.NORMAL)
                            self.log_text.insert(tk.END, f"[ALERT] {student_name}'s face is too far away.\n")
                            self.log_text.config(state=tk.DISABLED)
                        elif unusual_size_status == "too_close":
                            name = f"{student_name} (Move back)"
                            self.log_text.config(state=tk.NORMAL)
                            self.log_text.insert(tk.END, f"[ALERT] {student_name}'s face is too close.\n")
                            self.log_text.config(state=tk.DISABLED)
                    else:
                        face_rect = dlib.rectangle(left, top, right, bottom)
                        landmarks = self.predictor(frame, face_rect)

                        # --- Simplified Blink Detection ---
                        left_eye = get_landmark_array(landmarks, LEFT_EYE_START, LEFT_EYE_END)
                        right_eye = get_landmark_array(landmarks, RIGHT_EYE_START, RIGHT_EYE_END)
                        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                        
                        if ear < EYE_AR_THRESHOLD:
                            self.blink_counter[roll_no] = self.blink_counter.get(roll_no, 0) + 1
                        else:
                            if self.blink_counter.get(roll_no, 0) >= EYE_AR_CONSEC_FRAMES:
                                mark_attendance(roll_no, self.student_details_df, "Present", self.log_text)
                                self.already_marked_rolls.add(roll_no)
                                name = f"{student_name} (Present)"
                                color = (0, 255, 0)
                            self.blink_counter[roll_no] = 0

                        if name != f"{student_name} (Present)":
                             name = f"{student_name} (Please blink)"
                else:
                    name = f"{student_name} (Present)"
                    color = (0, 255, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), FONT, 1.0, (0, 0, 0), 1)
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.window.after(1, self.process_video_stream)

    def register_faces(self):
        """Scans the images directory, registers faces, and saves encodings."""
        print("[INFO] Starting face registration process...")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, "[INFO] Starting face registration process...\n")
        self.log_text.config(state=tk.DISABLED)

        try:
            df = pd.read_csv(STUDENT_DETAILS_CSV, dtype={'Roll': str})
            if df.empty:
                messagebox.showerror("Error", f"The file '{STUDENT_DETAILS_CSV}' is empty.")
                return
        except FileNotFoundError:
            messagebox.showerror("Error", f"'{STUDENT_DETAILS_CSV}' not found. Please create it.")
            return
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", f"No data or columns found in '{STUDENT_DETAILS_CSV}'.")
            return

        known_face_encodings = []
        known_face_ids = []

        for row in df.itertuples(index=False):
            roll_no = row.Roll
            image_path = os.path.join(IMAGES_DIR, f"{roll_no}.jpg")

            if not os.path.exists(image_path):
                log_message = f"[WARNING] Image not found for Roll {roll_no}. Skipping.\n"
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, log_message)
                self.log_text.config(state=tk.DISABLED)
                continue

            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_ids.append(str(roll_no))
                log_message = f"[SUCCESS] Encoded face for Roll: {roll_no}\n"
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, log_message)
                self.log_text.config(state=tk.DISABLED)
            else:
                log_message = f"[WARNING] No face found in '{image_path}'. Skipping.\n"
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, log_message)
                self.log_text.config(state=tk.DISABLED)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n[INFO] Saving {len(known_face_ids)} encoded faces to '{ENCODING_FILE_PATH}'...\n")
        self.log_text.config(state=tk.DISABLED)
        data = {"encodings": known_face_encodings, "ids": known_face_ids}
        with open(ENCODING_FILE_PATH, "wb") as f:
            f.write(pickle.dumps(data))
        
        messagebox.showinfo("Success", f"Face registration process completed successfully. {len(known_face_ids)} faces registered.")
        
def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
