# app/main.py
import os
import logging
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
import sys
from datetime import datetime

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

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f"Attempting to connect to Redis at {REDIS_URL}")
    redis_client = redis.from_url(REDIS_URL)
    # Test connection
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_URL}")
except Exception as e:
    logger.error(f"Warning: Could not connect to Redis: {str(e)}")
    logger.warning("The application will work but without caching capabilities.")
    redis_client = None

# Log environment info for troubleshooting
logger.info(f"Python version: {sys.version}")
logger.info(f"Running with DEBUG={DEBUG}, PORT={PORT}")
if "OPENAI_API_KEY" in os.environ:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    masked_key = "AVAILABLE" if api_key else "NOT SET"
    logger.info(f"OPENAI_API_KEY: {masked_key}")
else:
    logger.error("OPENAI_API_KEY environment variable is not set. Whisper fallback will not work.")
    
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
        logger.error(f"Invalid YouTube URL: {video_request.url}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    logger.info(f"Processing request for video ID: {video_id} with chunk size: {video_request.chunk_size_minutes}")
    
    # Test direct YouTube transcript availability
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info(f"YouTube transcript test successful - found {len(transcript)} segments")
    except Exception as e:
        logger.warning(f"YouTube transcript test failed: {str(e)}")
    
    # Check Redis connection
    if redis_client:
        try:
            redis_client.ping()
            logger.info("Redis connection test successful")
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
    
    # Clear any existing error or status for this video to start fresh
    if redis_client:
        try:
            for key in [f"error:{video_id}", f"status:{video_id}"]:
                if redis_client.exists(key):
                    redis_client.delete(key)
                    logger.info(f"Endpoint deleted existing {key} key to start fresh")
        except Exception as e:
            logger.error(f"Error clearing Redis keys: {str(e)}")
    
    # Set initial pending status
    if redis_client:
        try:
            redis_client.setex(f"status:{video_id}", 3600, STATUS_PENDING)
            logger.info(f"Endpoint set status:{video_id} to {STATUS_PENDING}")
        except Exception as e:
            logger.error(f"Error setting pending status: {str(e)}")
    
    # Check if already being processed
    if redis_client:
        try:
            if redis_client.exists(f"processing:{video_id}"):
                logger.info(f"Video {video_id} is already being processed")
                return VideoStatus(
                    video_id=video_id, 
                    status=STATUS_PROCESSING,
                    message="Video is already being processed"
                )
            
            # If already completed, just return the status
            if redis_client.exists(f"processed_video:{video_id}"):
                logger.info(f"Video {video_id} has already been processed")
                try:
                    redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
                    logger.info(f"Endpoint set status:{video_id} to {STATUS_COMPLETED} (already processed)")
                except Exception as e:
                    logger.error(f"Error setting completed status: {str(e)}")
                return VideoStatus(
                    video_id=video_id, 
                    status=STATUS_COMPLETED,
                    message="Video has already been processed"
                )
        except Exception as e:
            logger.error(f"Error checking video processing state: {str(e)}")
    
    # Process the video in the background
    try:
        background_tasks.add_task(
            process_video,
            video_id=video_id,
            chunk_size_minutes=video_request.chunk_size_minutes,
            redis_client=redis_client
        )
        logger.info(f"Started background processing task for video {video_id}")
    except Exception as e:
        logger.error(f"Error starting background task: {str(e)}")
        if redis_client:
            try:
                redis_client.setex(f"error:{video_id}", 86400, f"Failed to start processing: {str(e)}")
                redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
            except Exception as redis_e:
                logger.error(f"Error setting error status: {str(redis_e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
    
    return VideoStatus(
        video_id=video_id,
        status=STATUS_PENDING,
        message="Video processing started"
    )

@app.get("/videos/{video_id}/status", response_model=VideoStatus)
async def video_status_endpoint(video_id: str):
    status = get_video_status(video_id, redis_client)
    logger.info(f"Status check for video {video_id}: {status}")
    
    message = None
    if redis_client:
        if status == STATUS_ERROR and redis_client.exists(f"error:{video_id}"):
            message = redis_client.get(f"error:{video_id}").decode("utf-8")
            logger.info(f"Error message for video {video_id}: {message}")
        elif status == STATUS_COMPLETED:
            message = "Video processing completed successfully"
        elif status == STATUS_PROCESSING:
            message = "Video is being processed"
        elif status == STATUS_PENDING:
            message = "Video is queued for processing"
    
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

@app.get("/test/youtube/{video_id}", response_model=dict)
async def test_youtube_endpoint(video_id: str):
    """Endpoint to test YouTube API directly."""
    logger.info(f"Testing YouTube API for video ID: {video_id}")
    
    try:
        # Test direct access to YouTube
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            logger.info(f"Attempting to access YouTube: {watch_url}")
            
            import urllib.request
            headers = {
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            }
            req = urllib.request.Request(watch_url, headers=headers)
            
            with urllib.request.urlopen(req) as response:
                logger.info(f"Successfully accessed YouTube. Status: {response.status}")
                youtube_access = True
        except Exception as e:
            logger.error(f"Error accessing YouTube: {str(e)}")
            youtube_access = False
        
        # Test YouTube Transcript API
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            logger.info(f"Attempting to get transcript for {video_id}")
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            logger.info(f"Successfully retrieved transcript with {len(transcript)} segments")
            transcript_api = True
            transcript_sample = transcript[0] if transcript else None
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}")
            transcript_api = False
            transcript_sample = None
        
        # Create a mock transcript for fallback testing
        mock_transcript = [
            {"start": 0.0, "duration": 10.0, "text": "This is a mock transcript for testing purposes."}
        ]
        
        # Cache the test transcript
        if redis_client:
            logger.info(f"Caching test transcript for {video_id}")
            redis_client.setex(
                f"transcript:{video_id}", 
                604800,  # 1 week in seconds
                json.dumps(mock_transcript)
            )
            redis_client.setex(
                f"status:{video_id}", 
                604800,
                STATUS_COMPLETED
            )
            logger.info(f"Successfully cached test data for {video_id}")
        
        return {
            "video_id": video_id,
            "youtube_access": youtube_access,
            "transcript_api": transcript_api,
            "redis_client": redis_client is not None,
            "transcript_sample": transcript_sample,
            "test_status": "Testing completed and mock transcript cached",
            "next_step": f"Try accessing /videos/{video_id}/status or /videos/{video_id}/insights"
        }
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return {
            "video_id": video_id,
            "error": str(e),
            "youtube_access": False,
            "transcript_api": False,
            "redis_client": redis_client is not None,
            "test_status": "Error during testing"
        }

@app.post("/test/mock/{video_id}", response_model=dict)
async def create_mock_transcript(video_id: str, background_tasks: BackgroundTasks):
    """Create a mock transcript and insights for a video ID."""
    logger.info(f"Creating mock data for video ID: {video_id}")
    
    # Create a mock transcript
    mock_transcript = [
        {"start": 0.0, "duration": 10.0, "text": "This is a mock transcript for testing purposes."},
        {"start": 10.0, "duration": 10.0, "text": "The YouTube transcript API may be blocked in the Render environment."},
        {"start": 20.0, "duration": 10.0, "text": "This fallback ensures you can still test the processing pipeline."}
    ]
    
    # Create mock video info
    mock_video_info = {
        "id": video_id,
        "title": f"Mock Video {video_id}",
        "duration": "00:00:30",
        "uploader": "Test User",
        "upload_date": datetime.now().strftime("%Y-%m-%d"),
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    }
    
    # Create mock insights
    mock_insights = {
        "video_id": video_id,
        "video_info": mock_video_info,
        "insights": [
            {
                "start": "00:00:00",
                "end": "00:00:30",
                "chunk_id": "00:00:00-00:00:30",
                "topics": ["Testing", "Mock Data", "API Development"],
                "decisions": ["Use mock data for testing"],
                "concerns": ["YouTube API access in cloud environment"],
                "actions": ["Implement robust fallbacks"],
                "summary": "This is a mock summary for testing the DocRAT API without relying on external services."
            }
        ]
    }
    
    # Store the mock data in Redis
    if redis_client:
        try:
            # Save transcript
            redis_client.setex(
                f"transcript:{video_id}", 
                604800,  # 1 week
                json.dumps(mock_transcript)
            )
            
            # Save processed video data
            redis_client.setex(
                f"processed_video:{video_id}", 
                604800,  # 1 week
                json.dumps(mock_insights)
            )
            
            # Set status to completed
            redis_client.setex(
                f"status:{video_id}", 
                604800,  # 1 week
                STATUS_COMPLETED
            )
            
            # Clear any error flags
            if redis_client.exists(f"error:{video_id}"):
                redis_client.delete(f"error:{video_id}")
                
            # Clear processing flag
            if redis_client.exists(f"processing:{video_id}"):
                redis_client.delete(f"processing:{video_id}")
                
            logger.info(f"Successfully created mock data for video {video_id}")
            
            return {
                "video_id": video_id,
                "status": "success",
                "message": "Mock transcript and insights created",
                "next_steps": [
                    f"GET /videos/{video_id}/status",
                    f"GET /videos/{video_id}/insights"
                ]
            }
        except Exception as e:
            logger.error(f"Error creating mock data: {str(e)}")
            return {
                "video_id": video_id,
                "status": "error",
                "message": f"Failed to create mock data: {str(e)}"
            }
    else:
        return {
            "video_id": video_id,
            "status": "error",
            "message": "Redis client not available"
        }

# Add this at the end of the file for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=DEBUG)

