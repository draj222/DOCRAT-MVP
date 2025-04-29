import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import redis

from app.main import app
from app.youtube_insights import (
    STATUS_PENDING, 
    STATUS_PROCESSING, 
    STATUS_COMPLETED, 
    STATUS_ERROR
)

# Create a test client
client = TestClient(app)

# Mock Redis client
mock_redis = MagicMock()
mock_redis.exists.return_value = False

# Mock YouTube Transcript API
@pytest.fixture
def mock_youtube_transcript():
    with patch("app.youtube_insights.YouTubeTranscriptApi") as mock_api:
        mock_api.get_transcript.return_value = [
            {"text": "This is a test transcript.", "start": 0, "duration": 5},
            {"text": "It has multiple segments.", "start": 5, "duration": 5},
            {"text": "To simulate a real video.", "start": 10, "duration": 5}
        ]
        yield mock_api

# Mock OpenAI API
@pytest.fixture
def mock_openai():
    with patch("app.youtube_insights.openai") as mock_api:
        # Create a mock response for the ChatCompletion.acreate method
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "topics": ["Test Topic 1", "Test Topic 2"],
                        "decisions": ["Test Decision 1"],
                        "concerns": ["Test Concern 1", "Test Concern 2"],
                        "actions": ["Test Action 1"],
                        "summary": "This is a test summary."
                    })
                )
            )
        ]
        # Set up the async mock to return the mock response
        mock_acreate = MagicMock()
        mock_acreate.return_value = asyncio.Future()
        mock_acreate.return_value.set_result(mock_response)
        mock_api.ChatCompletion.acreate = mock_acreate
        yield mock_api

# Mock Redis connection
@pytest.fixture
def mock_redis_connection():
    with patch("app.main.redis_client", mock_redis):
        with patch("app.youtube_insights.redis_client", mock_redis):
            yield mock_redis

# Test root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    # Either HTML or JSON response is acceptable
    assert response.headers["content-type"].startswith("text/html") or response.json()["status"] == "operational"

# Test video processing endpoint with a valid URL
def test_process_video_valid_url(mock_youtube_transcript, mock_openai, mock_redis_connection):
    # Setup mock redis status check
    mock_redis.exists.return_value = False
    
    # Call the endpoint
    response = client.post(
        "/videos/process",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["status"] == STATUS_PENDING
    assert "message" in data

# Test video processing endpoint with an already processing video
def test_process_video_already_processing(mock_youtube_transcript, mock_openai, mock_redis_connection):
    # Setup mock redis for an already processing video
    def mock_get_status_side_effect(key):
        if key == f"status:dQw4w9WgXcQ":
            return True
        return False
    
    mock_redis.exists.side_effect = mock_get_status_side_effect
    mock_redis.get.return_value = STATUS_PROCESSING.encode()
    
    # Call the endpoint
    response = client.post(
        "/videos/process",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["status"] == STATUS_PROCESSING
    assert "already being processed" in data["message"]

# Test video processing endpoint with an invalid URL
def test_process_video_invalid_url():
    response = client.post(
        "/videos/process",
        json={"url": "https://www.example.com/not-a-youtube-url"}
    )
    assert response.status_code == 400

# Test video status endpoint
def test_video_status(mock_redis_connection):
    # Setup mock redis for a processing video
    mock_redis.exists.return_value = True
    mock_redis.get.return_value = STATUS_PROCESSING.encode()
    
    # Call the endpoint
    response = client.get("/videos/dQw4w9WgXcQ/status")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["status"] == STATUS_PROCESSING

# Test video insights endpoint - processing
def test_video_insights_processing(mock_redis_connection):
    # Setup mock redis for a processing video
    def mock_exists_side_effect(key):
        if key == f"processed_video:dQw4w9WgXcQ":
            return False
        elif key == f"status:dQw4w9WgXcQ":
            return True
        return False
    
    mock_redis.exists.side_effect = mock_exists_side_effect
    mock_redis.get.return_value = STATUS_PROCESSING.encode()
    
    # Call the endpoint
    response = client.get("/videos/dQw4w9WgXcQ/insights")
    
    # Should return 202 Accepted since video is still processing
    assert response.status_code == 202
    assert "still being processed" in response.json()["detail"]

# Test video insights endpoint - completed
def test_video_insights_completed(mock_redis_connection):
    # Setup mock redis for a completed video
    mock_redis.exists.return_value = True
    mock_redis.get.return_value = json.dumps({
        "video_id": "dQw4w9WgXcQ",
        "video_info": {
            "title": "Test Video",
            "duration": "00:00:15"
        },
        "insights": [
            {
                "chunk_id": "00:00:00-00:00:15",
                "start": "00:00:00",
                "end": "00:00:15",
                "topics": ["Test Topic"],
                "decisions": ["Test Decision"],
                "concerns": ["Test Concern"],
                "actions": ["Test Action"],
                "summary": "Test summary"
            }
        ]
    }).encode()
    
    # Call the endpoint
    response = client.get("/videos/dQw4w9WgXcQ/insights")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert "insights" in data
    assert len(data["insights"]) == 1
    assert "topics" in data["insights"][0]
    assert "Test Topic" in data["insights"][0]["topics"] 