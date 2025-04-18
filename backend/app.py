from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager 

# Import database connection functions and route routers
from database.mongo import connect_db, close_db
from routes import fugitive_routes, recognition_routes
# Import the function to load known faces into memory on startup
from routes.recognition_routes import load_known_faces

# Lifespan Management (Startup/Shutdown) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    print("Backend starting up...")
    try:
        # Connect to the database
        connect_db()
        # Load known faces into memory cache
        load_known_faces()
        print("Startup tasks complete.")
        yield # Application runs
    except Exception as e:
        print(f"Error during startup: {e}")

    print("Backend shutting down...")
    # Close the database connection on shutdown
    close_db()
    print("Shutdown tasks complete.")


# FastAPI Application Initialization 
app = FastAPI(
    title="Facial Recognition Backend",
    description="API for managing fugitives and performing facial recognition on photos/videos.",
    version="1.0.0",
    lifespan=lifespan 
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"], 
    allow_headers=["*"],
)

# Include Route Routers 

app.include_router(fugitive_routes.router, prefix="/api", tags=["Fugitives"])
app.include_router(recognition_routes.router, prefix="/api", tags=["Recognition"])

# Root or Status Endpoint
@app.get("/api/status", summary="Check Backend Status")
async def get_status():
    """Returns a simple status message to indicate the backend is running."""
    # You could add checks here to see if the DB is connected, etc.
    return {"status": "Backend is running", "database_connected": database.mongo.client is not None}

# Global Exception Handler 
@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc):
    print(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred."},
    )
