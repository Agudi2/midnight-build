import face_recognition
import numpy as np
from config import RECOGNITION_TOLERANCE

def find_best_match_index(unknown_encoding: np.ndarray, known_encodings: list[np.ndarray]) -> int | None:

    if not known_encodings or unknown_encoding is None:
        return None

    try:
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        if not face_distances.size > 0:
             return None

        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < RECOGNITION_TOLERANCE:
            return int(best_match_index) 
        else:
            return None # No match found within tolerance

    except Exception as e:
        print(f"Error finding best match: {e}")
        return None