# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

# Import our YouTube insights module
try:
    from app.youtube_insights import extract_video_id, process_video, get_video_status, get_video_insights
except ImportError:
    # Define mock functions for testing/deployment
    def extract_video_id(url): return url.split("v=")[-1] if "v=" in url else None
    def process_video(video_id, chunk_size_minutes=10): pass
    def get_video_status(video_id): return "pending"
    def get_video_insights(video_id): return {"message": "Not implemented yet"}

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

class VideoRequest(BaseModel):
    url: str
    chunk_size_minutes: int = 10

@app.get("/")
def read_root():
    return {"message": "Welcome to DocRAT API", "status": "operational"}

@app.post("/videos/process")
async def process_video_endpoint(video_request: VideoRequest, background_tasks: BackgroundTasks):
    video_id = extract_video_id(video_request.url)
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Process the video in the background
    background_tasks.add_task(
        process_video, 
        video_id=video_id,
        chunk_size_minutes=video_request.chunk_size_minutes
    )
    
    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video processing started"
    }

@app.get("/videos/{video_id}/status")
async def video_status(video_id: str):
    status = get_video_status(video_id)
    return {"video_id": video_id, "status": status}

@app.get("/videos/{video_id}/insights")
async def video_insights(video_id: str):
    insights = get_video_insights(video_id)
    if not insights:
        raise HTTPException(status_code=404, detail="Insights not found or processing incomplete")
    return insights

