# app/main.py
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
import re
from dotenv import load_dotenv
import redis
import json

from app.youtube_insights import (
    extract_video_id,
    process_video,
    get_video_status,
    get_video_insights,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_ERROR
)

# Load environment variables
load_dotenv()

# Get configuration from environment variables
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

app = FastAPI(
    title="DocRAT API",
    description="API for processing YouTube videos and extracting meeting insights",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.sariphi.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection
try:
    redis_client = redis.from_url(REDIS_URL)
    # Test connection
    redis_client.ping()
    print(f"Successfully connected to Redis at {REDIS_URL}")
except Exception as e:
    print(f"Warning: Could not connect to Redis: {str(e)}")
    print("The application will work but without caching capabilities.")
    redis_client = None

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Create a simple index.html if it doesn't exist
if not os.path.exists("static/index.html"):
    with open("static/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>DocRAT API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .endpoint {
            background-color: #f5f5f5;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        code {
            background-color: #eee;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>DocRAT API</h1>
    <p>Welcome to the DocRAT API - a tool for extracting insights from YouTube videos.</p>
    
    <h2>Endpoints:</h2>
    <div class="endpoint">
        <h3>POST /videos/process</h3>
        <p>Submit a YouTube video URL for processing.</p>
        <p>Example request body:</p>
        <pre><code>{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "chunk_size_minutes": 10
}</code></pre>
    </div>
    
    <div class="endpoint">
        <h3>GET /videos/{video_id}/status</h3>
        <p>Check the processing status of a video.</p>
    </div>
    
    <div class="endpoint">
        <h3>GET /videos/{video_id}/insights</h3>
        <p>Get the insights extracted from a processed video.</p>
    </div>
</body>
</html>
        """)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class VideoRequest(BaseModel):
    url: str
    chunk_size_minutes: int = 10
    
    @validator("url")
    def validate_youtube_url(cls, v):
        if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$', v):
            raise ValueError("Invalid YouTube URL")
        return v

class VideoStatus(BaseModel):
    video_id: str
    status: str
    message: Optional[str] = None

class TopicInsight(BaseModel):
    chunk_id: str
    start: str  # Format: "HH:MM:SS"
    end: str    # Format: "HH:MM:SS"
    topics: List[str]
    decisions: List[str]
    concerns: List[str]
    actions: List[str]
    summary: str

class VideoInsights(BaseModel):
    video_id: str
    video_info: Dict[str, Any]
    insights: List[TopicInsight]

# Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root():
    # Return static/index.html if it exists, otherwise return a JSON response
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return {"message": "Welcome to DocRAT API", "status": "operational"}

@app.post("/videos/process", response_model=VideoStatus)
async def process_video_endpoint(video_request: VideoRequest, background_tasks: BackgroundTasks):
    video_id = extract_video_id(video_request.url)
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Check if already being processed
    if redis_client:
        current_status = get_video_status(video_id, redis_client)
        if current_status == STATUS_PROCESSING:
            return VideoStatus(
                video_id=video_id, 
                status=STATUS_PROCESSING,
                message="Video is already being processed"
            )
        
        # If already completed, just return the status
        if current_status == STATUS_COMPLETED:
            return VideoStatus(
                video_id=video_id, 
                status=STATUS_COMPLETED,
                message="Video has already been processed"
            )
    
    # Mark as pending
    if redis_client:
        redis_client.setex(f"status:{video_id}", 3600, STATUS_PENDING)
    
    # Process the video in the background
    background_tasks.add_task(
        process_video, 
        video_id=video_id,
        chunk_size_minutes=video_request.chunk_size_minutes,
        redis_client=redis_client
    )
    
    return VideoStatus(
        video_id=video_id,
        status=STATUS_PENDING,
        message="Video processing started"
    )

@app.get("/videos/{video_id}/status", response_model=VideoStatus)
async def video_status_endpoint(video_id: str):
    status = get_video_status(video_id, redis_client)
    
    message = None
    if redis_client and status == STATUS_ERROR and redis_client.exists(f"error:{video_id}"):
        message = redis_client.get(f"error:{video_id}").decode("utf-8")
    
    return VideoStatus(
        video_id=video_id, 
        status=status,
        message=message
    )

@app.get("/videos/{video_id}/insights")
async def video_insights_endpoint(video_id: str):
    insights = get_video_insights(video_id, redis_client)
    
    if not insights:
        # Check status to provide better error messages
        status = get_video_status(video_id, redis_client)
        
        if status == STATUS_PENDING:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Video is queued for processing"
            )
        elif status == STATUS_PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Video is still being processed"
            )
        elif status == STATUS_ERROR:
            if redis_client and redis_client.exists(f"error:{video_id}"):
                error_message = redis_client.get(f"error:{video_id}").decode("utf-8")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error processing video: {error_message}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error processing video"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insights not found or processing incomplete"
            )
    
    return insights

# Add this at the end of the file for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=DEBUG)

