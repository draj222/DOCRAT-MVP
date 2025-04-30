# app/youtube_insights.py
import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import tempfile
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import pytube

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    logger.warning("WARNING: No OpenAI API key found in environment variables")

# Try to initialize client without organization ID first
try:
    client = OpenAI(api_key=api_key)
    async_client = AsyncOpenAI(api_key=api_key)
    logger.info(f"OpenAI API key configured: {'Yes' if api_key else 'No'} (key type: {'Project-scoped' if api_key.startswith('sk-proj-') else 'Standard' if api_key.startswith('sk-') else 'Unknown'})")
    logger.info("Successfully initialized OpenAI client without organization ID")
except Exception as e:
    logger.error(f"Error initializing OpenAI client without organization ID: {e}")
    
    # If that fails and it's a project key, try with an organization ID
    if api_key.startswith('sk-proj-'):
        try:
            # Try parsing the parts after "sk-proj-" using hyphens
            parts = api_key[8:].split('-')
            
            # Project ID is usually found in the first part 
            if parts and len(parts) > 0:
                org_id = parts[0]  # Use the first segment as org ID
                logger.info(f"Trying with extracted organization ID: {org_id}")
                
                client = OpenAI(api_key=api_key, organization=org_id)
                async_client = AsyncOpenAI(api_key=api_key, organization=org_id)
                logger.info(f"Successfully initialized OpenAI client with organization ID: {org_id}")
        except Exception as e2:
            logger.error(f"Error initializing OpenAI client with organization ID: {e2}")
            client = None
            async_client = None
    else:
        client = None
        async_client = None

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

def fallback_whisper(video_id: str) -> List[Dict[str, Any]]:
    """Download YouTube audio and transcribe using OpenAI Whisper when captions aren't available."""
    logger.info(f"Attempting Whisper fallback for video {video_id}")
    
    # Check if OpenAI client is available
    if not client or not api_key:
        logger.error("Cannot use Whisper fallback - no OpenAI API key configured")
        return []
    
    try:
        # Download audio from YouTube using pytube
        yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            logger.warning(f"No audio stream found for video {video_id}")
            return []
            
        # Create a temporary file to save the audio
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        # Download the audio
        logger.info(f"Downloading audio for video {video_id}")
        audio_stream.download(output_path=temp_dir, filename=f"{video_id}.mp4")
        
        # Transcribe with Whisper
        logger.info(f"Transcribing audio using Whisper for video {video_id}")
        with open(temp_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
        
        # Clean up the temporary file
        try:
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
        
        # Format the response into a transcript-like format
        # Whisper response contains segments with start, end times and text
        transcript = []
        if hasattr(response, 'segments'):
            for segment in response.segments:
                transcript.append({
                    'start': segment.start,
                    'duration': segment.end - segment.start,
                    'text': segment.text
                })
        else:
            # If no segments, create a single entry with the full text
            transcript.append({
                'start': 0.0,
                'duration': 0.0,  # We don't know the duration
                'text': response.text
            })
        
        logger.info(f"Successfully transcribed video {video_id} using Whisper")
        return transcript
        
    except Exception as e:
        logger.error(f"Error in Whisper fallback for video {video_id}: {str(e)}")
        return []

def get_transcript(video_id: str, redis_client=None) -> List[Dict[str, Any]]:
    """Get transcript for a YouTube video with caching and Whisper fallback."""
    # Check cache first
    if redis_client and redis_client.exists(f"transcript:{video_id}"):
        logger.info(f"Found cached transcript for video {video_id}")
        transcript_json = redis_client.get(f"transcript:{video_id}")
        return json.loads(transcript_json)
    
    try:
        # Try to get YouTube transcript first
        logger.info(f"Fetching YouTube transcript for video {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info(f"Successfully fetched YouTube transcript for video {video_id}")
        
        # Cache the transcript with TTL of 1 week (604800 seconds)
        if redis_client:
            redis_client.setex(
                f"transcript:{video_id}", 
                604800, 
                json.dumps(transcript)
            )
            logger.info(f"Cached YouTube transcript for video {video_id}")
        
        # Clear any existing error for this video since we succeeded
        if redis_client and redis_client.exists(f"error:{video_id}"):
            redis_client.delete(f"error:{video_id}")
            logger.info(f"Deleted existing error:{video_id} key after successful transcript fetch")
            
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.warning(f"YouTube transcript not available for video {video_id}: {str(e)}")
        
        # Clear any existing error for this video
        if redis_client and redis_client.exists(f"error:{video_id}"):
            redis_client.delete(f"error:{video_id}")
            logger.info(f"Deleted existing error:{video_id} key")
        
        # Try Whisper fallback
        fallback_transcript = fallback_whisper(video_id)
        
        if fallback_transcript:
            # Cache the fallback transcript
            if redis_client:
                redis_client.setex(
                    f"transcript:{video_id}", 
                    604800,  # 1 week in seconds
                    json.dumps(fallback_transcript)
                )
                logger.info(f"Cached Whisper transcript for video {video_id}")
            return fallback_transcript
        else:
            error_msg = "Failed to get transcript: No captions available and Whisper fallback failed or unavailable"
            logger.error(error_msg)
            
            # If fallback also failed, set error
            if redis_client:
                redis_client.setex(
                    f"error:{video_id}", 
                    86400,  # 1 day in seconds
                    error_msg
                )
                redis_client.setex(
                    f"status:{video_id}",
                    86400,  # 1 day in seconds
                    STATUS_ERROR
                )
                logger.warning(f"Set error and status keys for video {video_id}: both YouTube and Whisper failed")
            return []
    except Exception as e:
        error_msg = f"Unexpected error fetching transcript for video {video_id}: {str(e)}"
        logger.error(error_msg)
        if redis_client:
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                error_msg
            )
            redis_client.setex(
                f"status:{video_id}",
                86400,  # 1 day in seconds
                STATUS_ERROR
            )
            logger.warning(f"Set error and status keys for video {video_id}: {str(e)}")
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
        print(f"Attempting to call OpenAI API for video {video_id}, segment {start_formatted}-{end_formatted}")
        response = await async_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # This model supports JSON response format
            messages=[
                {"role": "system", "content": "You are an assistant that extracts insights from meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        print(f"Successfully received OpenAI response for video {video_id}, segment {start_formatted}-{end_formatted}")
        
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
        print(f"Error calling OpenAI API: {str(e)}")
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
        logger.info(f"Starting to process video {video_id} with chunk size {chunk_size_minutes} minutes")
        
        # Log Redis connection status
        if redis_client:
            try:
                redis_client.ping()
                logger.info(f"Redis connection is active for video {video_id}")
            except Exception as e:
                logger.error(f"Redis connection error for video {video_id}: {str(e)}")
        else:
            logger.warning(f"No Redis client provided for video {video_id}")
        
        # Clear any existing error or status for this video
        if redis_client:
            for key in [f"error:{video_id}", f"status:{video_id}"]:
                try:
                    if redis_client.exists(key):
                        redis_client.delete(key)
                        logger.info(f"Deleted existing {key} key to start fresh")
                except Exception as e:
                    logger.error(f"Error deleting Redis key {key}: {str(e)}")
        
        # Mark as processing
        if redis_client:
            try:
                redis_client.setex(f"status:{video_id}", 3600, STATUS_PROCESSING)
                logger.info(f"Set status:{video_id} to {STATUS_PROCESSING}")
                redis_client.setex(f"processing:{video_id}", 3600, STATUS_PROCESSING)
                logger.info(f"Set processing:{video_id} key")
            except Exception as e:
                logger.error(f"Error setting processing status for video {video_id}: {str(e)}")
        
        # Check if already processed
        if redis_client:
            try:
                if redis_client.exists(f"processed_video:{video_id}"):
                    video_json = redis_client.get(f"processed_video:{video_id}")
                    
                    # Mark as completed
                    redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
                    logger.info(f"Set status:{video_id} to {STATUS_COMPLETED} (already processed)")
                    
                    # Remove processing flag
                    redis_client.delete(f"processing:{video_id}")
                    logger.info(f"Deleted processing:{video_id} key (already processed)")
                    
                    logger.info(f"Returning cached results for video {video_id}")
                    return json.loads(video_json)
            except Exception as e:
                logger.error(f"Error checking if video {video_id} was already processed: {str(e)}")
        
        # Get transcript
        logger.info(f"Requesting transcript for video {video_id}")
        transcript = get_transcript(video_id, redis_client)
        
        if not transcript:
            error_msg = "No transcript available for this video"
            logger.error(f"{error_msg} for video {video_id}")
            if redis_client:
                try:
                    redis_client.setex(f"error:{video_id}", 86400, error_msg)
                    logger.info(f"Set error:{video_id} to '{error_msg}'")
                    redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
                    logger.info(f"Set status:{video_id} to {STATUS_ERROR}")
                    redis_client.delete(f"processing:{video_id}")
                    logger.info(f"Deleted processing:{video_id} key")
                except Exception as e:
                    logger.error(f"Error setting error status for video {video_id}: {str(e)}")
            raise Exception(error_msg)
        
        logger.info(f"Successfully retrieved transcript for video {video_id} with {len(transcript)} segments")
        
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
            logger.info(f"Cached processed_video:{video_id}")
            
            # Mark as completed
            redis_client.setex(f"status:{video_id}", 604800, STATUS_COMPLETED)
            logger.info(f"Set status:{video_id} to {STATUS_COMPLETED}")
            
            # Remove processing flag
            redis_client.delete(f"processing:{video_id}")
            logger.info(f"Deleted processing:{video_id} key")
        
        return final_result
    
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error in process_video for {video_id}: {error_message}")
        if redis_client:
            # Store error
            redis_client.setex(
                f"error:{video_id}", 
                86400,  # 1 day in seconds
                error_message
            )
            logger.info(f"Set error:{video_id} to '{error_message}'")
            
            # Mark as error
            redis_client.setex(f"status:{video_id}", 86400, STATUS_ERROR)
            logger.info(f"Set status:{video_id} to {STATUS_ERROR}")
            
            # Remove processing flag
            redis_client.delete(f"processing:{video_id}")
            logger.info(f"Deleted processing:{video_id} key")
        
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
