# app/youtube_insights.py
import os
import openai
from urllib.parse import urlparse, parse_qs

# Set OpenAI API key from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    # Handle youtu.be format
    if "youtu.be" in url:
        path = urlparse(url).path
        return path.strip("/")
    
    # Handle youtube.com format
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        if "/watch" in parsed_url.path:
            return parse_qs(parsed_url.query).get("v", [None])[0]
    
    return None

def process_video(video_id, chunk_size_minutes=10):
    """Process a video and extract insights."""
    # Implementation would go here
    pass

def get_video_status(video_id):
    """Get the processing status of a video."""
    # Implementation would go here
    return "pending"

def get_video_insights(video_id):
    """Get the generated insights for a processed video."""
    # Implementation would go here
    return {"message": "Not implemented yet"}
