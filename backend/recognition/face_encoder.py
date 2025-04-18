import face_recognition
import numpy as np
import cv2

def get_face_encoding_from_image_file(image_path: str, known_face_location: tuple | None = None) -> np.ndarray | None:
    try:
        image = face_recognition.load_image_file(image_path)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # face_recognition loads as RGB already

        face_locations = []
        if known_face_location:
             face_locations = [known_face_location] 
        else:
             face_locations = face_recognition.face_locations(image)
             if not face_locations:
                 print(f"No face found in {image_path} for encoding.")
                 return None
             # If multiple faces detected without specific location, encode the first one
             if len(face_locations) > 1:
                  print(f"Warning: Multiple faces detected in {image_path}. Encoding the first one.")
             face_locations = [face_locations[0]] # Use the first detected face


        # Get face encodings for the specified (or first detected) location(s)
        encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        if encodings:
            return encodings[0] 
        else:
            print(f"Could not generate encoding for face in {image_path}.")
            return None

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error getting face encoding from {image_path}: {e}")
        return None

def get_face_encodings_from_frame(frame: np.ndarray, face_locations: list[tuple]) -> list[np.ndarray]:

     if not face_locations:
         return []

     try:
         # face_recognition takes frame (numpy array) and locations
         encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
         return encodings
     except Exception as e:
         print(f"Error getting face encodings from frame: {e}")
         return []