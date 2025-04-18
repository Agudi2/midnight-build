from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo.errors
import numpy as np
from bson.objectid import ObjectId
from config import MONGO_URI, MONGO_DB_NAME, FUGITIVES_COLLECTION

# Global client and database objects
client = None
db = None
fugitives_collection = None

def connect_db():
    """Connects to the MongoDB database."""
    global client, db, fugitives_collection
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)

        client.admin.command('ping')

        print("MongoDB connection successful!")
        db = client[MONGO_DB_NAME]
        fugitives_collection = db[FUGITIVES_COLLECTION]
        print(f"Connected to database: '{MONGO_DB_NAME}', collection: '{FUGITIVES_COLLECTION}'")

    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB connection error: Could not connect to {MONGO_URI}. Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        raise

def close_db():
    """Closes the MongoDB connection."""
    global client
    if client:
        client.close()
        print("MongoDB connection closed.")

def insert_fugitive(name: str, age: int, gender: str, photo_path: str, embedding: np.ndarray):
    """Inserts fugitive data into the database."""
    if fugitives_collection is None:
        raise ConnectionError("MongoDB not connected. Call connect_db() first.")

    embedding_list = embedding.tolist()

    fugitive_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "photo_path": photo_path, 
        "embedding": embedding_list
    }

    try:
        result = fugitives_collection.insert_one(fugitive_data)
        print(f"Inserted fugitive: {name} with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error inserting fugitive {name} into DB: {e}")
        raise

def get_all_fugitives():
     """Retrieves all fugitive data, converting embedding back to numpy array."""
     if fugitives_collection is None:
        raise ConnectionError("MongoDB not connected. Call connect_db() first.")

     fugitives = []
     try:
         # Retrieve all documents
         cursor = fugitives_collection.find({})
         for doc in cursor:
             doc['embedding'] = np.array(doc['embedding'])
             fugitives.append(doc)
         return fugitives
     except Exception as e:
         print(f"Error fetching all fugitives from DB: {e}")
         raise