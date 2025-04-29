# app/youtube_insights.py
import os
import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Status constants
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

def extract_video_id(url: str) -> Optional[str]:
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
        elif "/live" in parsed_url.path:
            # For live streams like youtube.com/live/XQRui39pGSM
            parts = parsed_url.path.split("/")
            if len(parts) > 2:
                return parts[2]
    
    return None

def format_time(seconds: int) -> str:
    """Format seconds into HH:MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def get_transcript(video_id: str, redis_client=None) -> List[Dict[str, Any]]:
    """Get transcript for a YouTube video with caching."""
    # Check cache first
    if redis_client and redis_client.exists(f"transcript:{video_id}"):
        transcript_json = redis_client.get(f"transcript:{video_id}")
        return json.loads(transcript_json)
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Cache the transcript with TTL of 1 week (604800 seconds)
        if redis_client:
            redis_client.setex(
                f"transcript:{video_id}", 
                604800, 
                json.dumps(transcript)
            )
        
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        # Handle the case where transcript is not available
        if redis_client:
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                f"Transcript not available: {str(e)}"
            )
        return []
    except Exception as e:
        if redis_client:
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                f"Error fetching transcript: {str(e)}"
            )
        return []

def get_video_info(video_id: str, redis_client=None) -> Dict[str, Any]:
    """Get basic video information with caching."""
    # Check cache first
    if redis_client and redis_client.exists(f"video_info:{video_id}"):
        video_info_json = redis_client.get(f"video_info:{video_id}")
        return json.loads(video_info_json)
    
    try:
        # For a full implementation, you would use youtube_dl or yt-dlp here
        # For this MVP, we'll use a simple mock that returns basic information
        video_info = {
            "id": video_id,
            "title": f"Video {video_id}",
            "duration": "Unknown",  # Will be calculated from transcript
            "uploader": "YouTube Creator",
            "upload_date": datetime.now().strftime("%Y-%m-%d"),
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        }
        
        # Cache the video info with TTL of 1 week (604800 seconds)
        if redis_client:
            redis_client.setex(
                f"video_info:{video_id}", 
                604800, 
                json.dumps(video_info)
            )
        
        return video_info
    except Exception as e:
        # In case of error, return minimal data
        error_info = {
            "id": video_id,
            "title": f"Video {video_id}",
            "error": str(e)
        }
        
        if redis_client:
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                f"Error fetching video info: {str(e)}"
            )
        
        return error_info

def chunk_transcript(
    transcript: List[Dict[str, Any]], 
    chunk_size_minutes: int
) -> List[Tuple[float, float, str]]:
    """Split transcript into chunks of specified minutes."""
    chunks = []
    chunk_size_seconds = chunk_size_minutes * 60
    
    # Calculate total duration from transcript
    if transcript:
        last_item = transcript[-1]
        total_duration = last_item['start'] + last_item.get('duration', 0)
        
        # Determine chunk boundaries
        for chunk_start in range(0, int(total_duration), chunk_size_seconds):
            chunk_end = min(chunk_start + chunk_size_seconds, total_duration)
            
            # Filter transcript items that fall within this chunk
            chunk_text = ""
            for item in transcript:
                if chunk_start <= item['start'] < chunk_end:
                    chunk_text += item['text'] + " "
            
            if chunk_text:
                chunks.append((chunk_start, chunk_end, chunk_text.strip()))
    
    return chunks

async def extract_insights(
    video_title: str,
    chunk_start: float, 
    chunk_end: float, 
    transcript_text: str,
    redis_client=None,
    video_id: str = None
) -> Dict[str, Any]:
    """Extract insights from a transcript chunk using OpenAI."""
    start_formatted = format_time(chunk_start)
    end_formatted = format_time(chunk_end)
    
    # Check cache first if video_id is provided
    if redis_client and video_id:
        cache_key = f"insights:{video_id}:{start_formatted}"
        if redis_client.exists(cache_key):
            insights_json = redis_client.get(cache_key)
            return json.loads(insights_json)
    
    prompt = f"""
Video Title: {video_title}
Segment: {start_formatted}–{end_formatted}
Transcript: "{transcript_text}"

Task:
 1. List key topics.
 2. Extract decisions/votes.
 3. Note citizen concerns.
 4. Identify action items.
 5. Summarize in 3–5 sentences.

Format your response as a JSON object with these keys:
- topics (array of strings)
- decisions (array of strings)
- concerns (array of strings)
- actions (array of strings)
- summary (string)
"""

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts insights from meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse response as JSON
        result = json.loads(content)
        
        # Add chunk timing information
        result["start"] = start_formatted
        result["end"] = end_formatted
        result["chunk_id"] = f"{start_formatted}-{end_formatted}"
        
        # Cache the insights if video_id and redis_client are provided
        if redis_client and video_id:
            redis_client.setex(
                f"insights:{video_id}:{start_formatted}", 
                604800,  # 1 week in seconds
                json.dumps(result)
            )
        
        return result
    except Exception as e:
        # In case of error, return minimal data
        error_result = {
            "start": start_formatted,
            "end": end_formatted,
            "chunk_id": f"{start_formatted}-{end_formatted}",
            "topics": ["Error processing this segment"],
            "decisions": [],
            "concerns": [],
            "actions": [],
            "summary": f"Error extracting insights: {str(e)}"
        }
        
        # Still cache the error result to avoid repeated failures
        if redis_client and video_id:
            redis_client.setex(
                f"insights:{video_id}:{start_formatted}", 
                86400,  # 1 day in seconds
                json.dumps(error_result)
            )
        
        return error_result

async def process_video(video_id: str, chunk_size_minutes: int = 10, redis_client=None):
    """Process a video by chunking transcript and extracting insights."""
    try:
        # Mark as processing
        if redis_client:
            redis_client.setex(f"processing:{video_id}", 3600, STATUS_PROCESSING)
        
        # Check if already processed
        if redis_client and redis_client.exists(f"processed_video:{video_id}"):
            video_json = redis_client.get(f"processed_video:{video_id}")
            
            # Mark as completed
            redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
            
            # Remove processing flag
            redis_client.delete(f"processing:{video_id}")
            
            return json.loads(video_json)
        
        # Get transcript
        transcript = get_transcript(video_id, redis_client)
        
        if not transcript:
            error_msg = "No transcript available for this video"
            if redis_client:
                redis_client.setex(f"error:{video_id}", 86400, error_msg)
                redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
                redis_client.delete(f"processing:{video_id}")
            raise Exception(error_msg)
        
        # Get video info
        video_info = get_video_info(video_id, redis_client)
        
        # Calculate video duration from transcript for video_info
        if transcript:
            last_item = transcript[-1]
            total_duration_seconds = last_item['start'] + last_item.get('duration', 0)
            video_info["duration"] = format_time(total_duration_seconds)
        
        # Chunk the transcript
        chunks = chunk_transcript(transcript, chunk_size_minutes)
        
        # Process each chunk
        tasks = []
        for chunk_start, chunk_end, chunk_text in chunks:
            tasks.append(
                extract_insights(
                    video_info["title"], 
                    chunk_start, 
                    chunk_end, 
                    chunk_text,
                    redis_client,
                    video_id
                )
            )
        
        # Run tasks concurrently and collect results
        insights = []
        for task in asyncio.as_completed(tasks):
            result = await task
            insights.append(result)
        
        # Sort insights by start time
        insights.sort(key=lambda x: x["start"])
        
        # Assemble final result
        final_result = {
            "video_id": video_id,
            "video_info": video_info,
            "insights": insights
        }
        
        # Store in Redis with TTL of 1 week
        if redis_client:
            redis_client.setex(
                f"processed_video:{video_id}", 
                604800,  # 1 week in seconds
                json.dumps(final_result)
            )
            
            # Mark as completed
            redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
            
            # Remove processing flag
            redis_client.delete(f"processing:{video_id}")
        
        return final_result
    
    except Exception as e:
        # Handle errors
        error_message = str(e)
        if redis_client:
            # Store error
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                error_message
            )
            
            # Mark as error
            redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
            
            # Remove processing flag
            redis_client.delete(f"processing:{video_id}")
        
        raise Exception(f"Error processing video {video_id}: {error_message}")

def get_video_status(video_id: str, redis_client=None) -> str:
    """Get the processing status of a video."""
    if not redis_client:
        return STATUS_PENDING
    
    # Check if there's a stored status
    if redis_client.exists(f"status:{video_id}"):
        return redis_client.get(f"status:{video_id}").decode("utf-8")
    
    # Check if video is processed
    if redis_client.exists(f"processed_video:{video_id}"):
        # Update status for consistency
        redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
        return STATUS_COMPLETED
    
    # Check if video is being processed
    if redis_client.exists(f"processing:{video_id}"):
        return STATUS_PROCESSING
    
    # Check if there was an error
    if redis_client.exists(f"error:{video_id}"):
        # Update status for consistency
        redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
        return STATUS_ERROR
    
    # Default to pending
    return STATUS_PENDING

def get_video_insights(video_id: str, redis_client=None) -> Optional[Dict[str, Any]]:
    """Get the generated insights for a processed video."""
    if not redis_client:
        return None
    
    # Check if video is processed
    if redis_client.exists(f"processed_video:{video_id}"):
        video_json = redis_client.get(f"processed_video:{video_id}")
        return json.loads(video_json)
    
    # If not processed, check status
    status = get_video_status(video_id, redis_client)
    
    if status == STATUS_ERROR and redis_client.exists(f"error:{video_id}"):
        error_message = redis_client.get(f"error:{video_id}").decode("utf-8")
        return {
            "video_id": video_id,
            "status": STATUS_ERROR,
            "error": error_message
        }
    
    # Otherwise, insights are not available yet
    return None
