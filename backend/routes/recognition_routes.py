from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse 
from typing import Annotated
import os
import shutil
import cv2
import numpy as np
import uuid 
import time 

from config import TEMP_FOLDER, FUGITIVES_PHOTO_FOLDER, VIDEO_FRAME_INTERVAL
from database.mongo import get_all_fugitives
from recognition.face_detector import detect_faces_in_frame 
from recognition.face_encoder import get_face_encodings_from_frame
from recognition.face_matcher import find_best_match_index

router = APIRouter()

_known_face_encodings = []
_known_face_info = [] # List of dicts (name, age, gender, photo_path) corresponding to encodings

def load_known_faces():
    """Loads fugitive data from DB into the in-memory cache."""
    global _known_face_encodings, _known_face_info
    print("Loading known faces from database into memory...")
    try:
        fugitives = get_all_fugitives() 
        _known_face_encodings = [f['embedding'] for f in fugitives]
        # Store relevant info for display
        _known_face_info = [{
            "_id": str(f["_id"]), 
            "name": f['name'],
            "age": f['age'],
            "gender": f['gender'],
            "photo_filename": f['photo_path'] 
            } for f in fugitives]
        print(f"Loaded {len(_known_face_encodings)} known faces.")
    except Exception as e:
        print(f"Error loading known faces from DB: {e}")

@router.post("/recognize/")
async def recognize_in_media(
    file: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks # Used for cleanup
):

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    # Ensure known faces are loaded (simple check)
    if not _known_face_encodings:
         load_known_faces()
         if not _known_face_encodings:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No known faces loaded. Please add fugitives first or check backend logs.")

    os.makedirs(TEMP_FOLDER, exist_ok=True)

    unique_temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_FOLDER, unique_temp_filename)

    try:
        with open(temp_file_path, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        print(f"Error saving temp file {file.filename} to {temp_file_path}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error saving uploaded file.")

    mime_type = file.content_type or 'application/octet-stream' 
    print(f"Received file: {file.filename}, MIME type: {mime_type}")

    # Process File 
    processing_results = {
        "message": "Processing complete",
        "type": "unknown",
        "results": [] 
    }

    try:
        if mime_type.startswith('image/'):
            processing_results["type"] = "image"
            # Process Image
            image = cv2.imread(temp_file_path)
            if image is None:
                 raise Exception(f"Could not read image file: {temp_file_path}")

            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces
            detected_faces = detect_faces_in_frame(rgb_image) # Returns list with "box" and "location"

            if not detected_faces:
                processing_results["message"] = "No faces detected in image."
                pass 

            # Get encodings for detected faces
            face_locations = [d["location"] for d in detected_faces]
            face_encodings = get_face_encodings_from_frame(rgb_image, face_locations)

            # Match detected faces against known faces
            identified_faces_info = [] 

            for i, unknown_encoding in enumerate(face_encodings):
                best_match_index = find_best_match_index(unknown_encoding, _known_face_encodings)

                # Store results for this face
                face_result = {
                    "box": detected_faces[i]["box"], 
                    "match": False,
                    "info": None 
                }

                if best_match_index is not None:
                    # Found a match!
                    matched_fugitive_info = _known_face_info[best_match_index]
                    face_result["match"] = True
                    face_result["info"] = matched_fugitive_info
                    print(f"Image: Match found for face {i} -> {matched_fugitive_info['name']}")

                identified_faces_info.append(face_result)

            processing_results["results"] = identified_faces_info

            # Annotate the image with results 
            image_to_annotate = image.copy() # Work on a copy

            for face_data in identified_faces_info:
                x, y, w, h = face_data["box"]
                # Draw bounding box
                color = (0, 255, 0) if face_data["match"] else (0, 0, 255) # Green for match, Red for no match
                thickness = 2
                cv2.rectangle(image_to_annotate, (x, y), (x+w, y+h), color, thickness)

                # Add text (name and info if matched)
                label = "Unknown"
                if face_data["match"]:
                    info = face_data["info"]
                    label = f"{info['name']} ({info['age']}, {info['gender']})"
        

                # Put text slightly above the box
                text_y_offset = 15
                cv2.putText(image_to_annotate, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)


            # Save the annotated image temporarily
            annotated_filename = f"annotated_{os.path.splitext(unique_temp_filename)[0]}.jpg" # Force JPG output
            annotated_file_path = os.path.join(TEMP_FOLDER, annotated_filename)
            cv2.imwrite(annotated_file_path, image_to_annotate)

            # Add cleanup task for the annotated image file
            background_tasks.add_task(lambda: os.path.exists(annotated_file_path) and os.remove(annotated_file_path))

            # Return the path to the annotated image and the results JSON
            processing_results["annotated_image_url"] = f"/api/temp/{annotated_filename}" # URL accessible from frontend


        elif mime_type.startswith('video/'):
            processing_results["type"] = "video"
            # Process Video (Simplified: Process sampled frames)

            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                raise Exception(f"Error opening video file: {temp_file_path}")

            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            processing_results["video_info"] = {
                "frame_count": frame_count,
                "frame_rate": frame_rate,
                "width": frame_width,
                "height": frame_height
            }
            processing_results["results_per_frame"] = [] # Detailed results for processed frames
            processing_results["annotated_frame_urls"] = [] # Paths to saved annotated frames

            print(f"Video Info: {frame_count} frames, {frame_rate:.2f} FPS, {frame_width}x{frame_height}")

            processed_frame_paths = [] # Keep track of saved annotated frames for cleanup/return

            # Process frames at intervals
            frame_interval = max(1, int(VIDEO_FRAME_INTERVAL)) # Ensure interval is at least 1
            frames_indices_to_process = range(0, frame_count, frame_interval)
            print(f"Processing approximately every {frame_interval} frames.")


            for i in frames_indices_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    print(f"Warning: Could not read frame {i}. Stopping video processing loop.")
                    break # End of video or read error

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                detected_faces = detect_faces_in_frame(rgb_frame) # Returns list with "box" and "location"

                frame_results = {
                    "frame_index": i,
                    "timestamp_ms": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "faces": []
                }

                if not detected_faces:
                     # Append frame results even if no faces found
                     processing_results["results_per_frame"].append(frame_results)
                     # Optionally save original frame if needed for display later
                     continue # Skip encoding/matching for this frame

                # Get encodings
                face_locations = [d["location"] for d in detected_faces]
                face_encodings = get_face_encodings_from_frame(rgb_frame, face_locations)

                # Match faces
                identified_faces_info_this_frame = []
                for j, unknown_encoding in enumerate(face_encodings):
                    best_match_index = find_best_match_index(unknown_encoding, _known_face_encodings)

                    face_result = {
                         "box": detected_faces[j]["box"],
                         "match": False,
                         "info": None
                    }

                    if best_match_index is not None:
                        matched_fugitive_info = _known_face_info[best_match_index]
                        face_result["match"] = True
                        face_result["info"] = matched_fugitive_info

                    identified_faces_info_this_frame.append(face_result)

                frame_results["faces"] = identified_faces_info_this_frame
                processing_results["results_per_frame"].append(frame_results)

                frame_to_annotate = frame.copy() # Work on a copy

                for face_data in identified_faces_info_this_frame:
                     x, y, w, h = face_data["box"]
                     color = (0, 255, 0) if face_data["match"] else (0, 0, 255)
                     thickness = 2
                     cv2.rectangle(frame_to_annotate, (x, y), (x+w, y+h), color, thickness)

                     label = "Unknown"
                     if face_data["match"]:
                         info = face_data["info"]
                         label = f"{info['name']}" # Keep label short for video frames

                     # Put text
                     cv2.putText(frame_to_annotate, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

                # Save the annotated frame temporarily as an image file
                annotated_frame_filename = f"frame_{i}_{os.path.splitext(unique_temp_filename)[0]}.jpg"
                annotated_frame_path = os.path.join(TEMP_FOLDER, annotated_frame_filename)
                cv2.imwrite(annotated_frame_path, frame_to_annotate)
                processed_frame_paths.append(annotated_frame_path)


            cap.release() # Release the video file handle

            # Add cleanup tasks for all saved annotated frame files
            for frame_path in processed_frame_paths:
                 background_tasks.add_task(lambda p=frame_path: os.path.exists(p) and os.remove(p))

            # Return URLs for the annotated frames and the structured results
            processing_results["annotated_frame_urls"] = [f"/api/temp/{os.path.basename(p)}" for p in processed_frame_paths]

        else:
            # Unsupported file type based on MIME
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {mime_type}. Please upload an image or video.")

    except Exception as e:
        print(f"Error during file processing: {e}")
        # Ensure the initial temp file is also cleaned up on error
        if os.path.exists(temp_file_path):
             os.remove(temp_file_path)
        # Clean up any annotated frames that might have been saved before the error
        if 'processed_frame_paths' in locals(): # Check if list was created
            for frame_path in processed_frame_paths:
                 if os.path.exists(frame_path):
                      os.remove(frame_path)

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing media file: {e}")

    finally:
        # Ensure the initial temp file is removed after processing (success or failure caught above)
        if os.path.exists(temp_file_path):
             os.remove(temp_file_path)


    # Return the final results JSON
    return JSONResponse(content=processing_results)


@router.get("/temp/{filename}")
async def serve_temp_file(filename: str):
    """Serves temporary annotated image or frame files."""
    # Basic sanitization: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

    file_path = os.path.join(TEMP_FOLDER, filename)

    # Check if the file exists and is actually within the TEMP_FOLDER
    if os.path.exists(file_path) and os.path.abspath(file_path).startswith(os.path.abspath(TEMP_FOLDER)):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

# Helper endpoint to serve fugitive photos if the frontend needs them (e.g., for display)
@router.get("/fugitives/photos/{filename}")
async def serve_fugitive_photo(filename: str):
     """Serves fugitive photos."""
     # Basic sanitization: prevent directory traversal
     if '..' in filename or '/' in filename or '\\' in filename:
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

     file_path = os.path.join(FUGITIVES_PHOTO_FOLDER, filename)

     # Check if the file exists and is actually within the FUGITIVES_PHOTO_FOLDER
     if os.path.exists(file_path) and os.path.abspath(file_path).startswith(os.path.abspath(FUGITIVES_PHOTO_FOLDER)):
         return FileResponse(file_path)
     else:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found")