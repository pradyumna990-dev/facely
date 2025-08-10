import face_recognition
import pickle
import os
import pandas as pd

# --- CONFIGURATION ---
IMAGES_DIR = 'images'
STUDENT_DETAILS_CSV = 'students.csv'
ENCODING_FILE_PATH = 'encodings.dat'
# ---------------------

print("[INFO] Starting face encoding process...")

# Load student details from CSV
try:
    df = pd.read_csv(STUDENT_DETAILS_CSV)
except FileNotFoundError:
    print(f"[ERROR] '{STUDENT_DETAILS_CSV}' not found.")
    print("Please create it with columns: Roll,Name,Branch,Sem")
    exit()

known_face_encodings = []
known_face_ids = []  # Using Roll number as the unique ID

# Loop over the rows of the student details CSV
for index, row in df.iterrows():
    roll_no = row['Roll']
    image_path = os.path.join(IMAGES_DIR, f"{roll_no}.jpg")  # Assumes image is named as Roll.jpg

    if not os.path.exists(image_path):
        print(f"[WARNING] Image not found for Roll {roll_no} at '{image_path}'. Skipping.")
        continue

    # Load the image file
    image = face_recognition.load_image_file(image_path)
    
    # Find face encodings. We assume one face per image for registration.
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        # Add the first found encoding and corresponding roll number
        known_face_encodings.append(face_encodings[0])
        known_face_ids.append(str(roll_no))
        print(f"[INFO] Encoded face for Roll: {roll_no}")
    else:
        print(f"[WARNING] No face found in '{image_path}' for Roll {roll_no}. Skipping.")

# Save the encodings and IDs to a file for the main app to use
print(f"[INFO] Saving encodings to '{ENCODING_FILE_PATH}'...")
data = {"encodings": known_face_encodings, "ids": known_face_ids}
with open(ENCODING_FILE_PATH, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Face encoding process completed successfully.")