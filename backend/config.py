import os

# Database Configuration
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Midnight")  
MONGO_URI = f"mongodb+srv://user1:{DB_PASSWORD}@cluster0.3c1j4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = "facial_recognition_db"
FUGITIVES_COLLECTION = "fugitives"

# Uploads Configuration 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FUGITIVES_PHOTO_FOLDER = os.path.join(UPLOAD_FOLDER, "fugitives")
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")

os.makedirs(FUGITIVES_PHOTO_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Recognition Configuration 
RECOGNITION_TOLERANCE = 0.6 

# Video processing configuration
VIDEO_FRAME_INTERVAL = 30 