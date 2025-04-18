from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Annotated
import os
import shutil
import uuid 

from config import FUGITIVES_PHOTO_FOLDER
from database.mongo import insert_fugitive, get_all_fugitives
from recognition.face_encoder import get_face_encoding_from_image_file
from recognition.face_detector import detect_faces_in_image_file

router = APIRouter()

@router.post("/fugitives/")
async def add_fugitive(
    name: Annotated[str, Form(...)], 
    age: Annotated[int, Form(...)],
    gender: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)]
):
    """
    Adds a new fugitive to the database. Requires name, age, gender, and a photo file.
    The photo must contain exactly one detectable face.
    """
    # Validate input and file
    if not file.filename:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only JPG, JPEG, PNG allowed.")

    os.makedirs(FUGITIVES_PHOTO_FOLDER, exist_ok=True)

    # Generate a unique filename to prevent conflicts
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = os.path.join(FUGITIVES_PHOTO_FOLDER, unique_filename)

    # Save the uploaded photo temporarily to process it
    temp_processing_path = file_location 
    try:
        with open(temp_processing_path, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        print(f"Error saving file {original_filename} to {temp_processing_path}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error saving uploaded file.")

    # Process the photo: Detect face and get encoding
    try:
        face_locations = detect_faces_in_image_file(temp_processing_path)

        if not face_locations:
            # Clean up the saved file if no face is found
            os.remove(temp_processing_path)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No face detected in the uploaded photo.")

        # Ensure exactly one face is detected for a fugitive photo
        if len(face_locations) > 1:
             os.remove(temp_processing_path) # Clean up the saved file
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Multiple faces detected. Please upload a photo with only one person.")

        # Get encoding for the single detected face
        encoding = get_face_encoding_from_image_file(temp_processing_path, known_face_location=face_locations[0]['location'])

        if encoding is None:
             # Clean up the saved file
             os.remove(temp_processing_path)
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not generate face encoding from the photo.")

    except Exception as e:
        # Catch any other processing errors and clean up
        if os.path.exists(temp_processing_path):
             os.remove(temp_processing_path)
        print(f"Error processing photo {original_filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing photo: {e}")


    # Save data to MongoDB
    try:
        
        photo_storage_path = os.path.relpath(file_location, FUGITIVES_PHOTO_FOLDER) 

        fugitive_id = insert_fugitive(name, age, gender, unique_filename, encoding) 

        # Return success response
        return JSONResponse(content={
            "message": "Fugitive added successfully",
            "fugitive_id": str(fugitive_id), 
            "name": name,
            "photo_filename": unique_filename
            },
            status_code=status.HTTP_201_CREATED
        )

    except Exception as e:

        if os.path.exists(temp_processing_path):
            os.remove(temp_processing_path)
        print(f"Database error adding fugitive {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error saving fugitive to database.")


@router.get("/fugitives/")
async def get_fugitives_list():
    """
    Retrieves a list of all fugitives in the database (excluding embeddings).
    """
    try:
        fugitives = get_all_fugitives()
        
        fugitives_list = []
        for fugitive in fugitives:
            fugitives_list.append({
                "_id": str(fugitive["_id"]), 
                "name": fugitive["name"],
                "age": fugitive["age"],
                "gender": fugitive["gender"],
                "photo_filename": fugitive["photo_path"]
            })
        return fugitives_list
    except Exception as e:
        print(f"Error fetching fugitives list: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving fugitives list.")
