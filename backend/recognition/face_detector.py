import face_recognition
import numpy as np
import cv2

def detect_faces_in_image_file(image_path: str) -> list:
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image) 

        detected_faces_info = []
        for (top, right, bottom, left) in face_locations:
             detected_faces_info.append({
                 "box": [left, top, right - left, bottom - top], 
                 "location": (top, right, bottom, left)          
             })
        return detected_faces_info
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return []
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return []

def detect_faces_in_frame(frame: np.ndarray) -> list:
     try:
         
         face_locations = face_recognition.face_locations(frame) 

         detected_faces_info = []
         for (top, right, bottom, left) in face_locations:
              detected_faces_info.append({
                  "box": [left, top, right - left, bottom - top], 
                  "location": (top, right, bottom, left)          
              })
         return detected_faces_info
     except Exception as e:
         print(f"Error detecting faces in frame: {e}") # Might be too noisy for videos
         return []