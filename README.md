# DOCRAT-MVP

DocRAT (Document Reading and Analysis Tool) is an AI-powered service that automatically extracts actionable insights from YouTube videos, particularly useful for government meetings, town halls, and other public discussions.

## Features

- **Automated Video Analysis**: Submit a YouTube URL and get AI-generated insights.
- **Real-time Processing Status**: Check the status of your video processing request.
- **Structured Insights**: For each video segment, extract:
  - Key topics discussed
  - Decisions and votes taken
  - Citizen concerns raised
  - Action items identified
  - Concise segment summaries

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI**: OpenAI GPT-4
- **Caching**: Redis
- **Transcription**: YouTube Transcript API
- **Containerization**: Docker
- **Deployment**: Render

## Setup

### Prerequisites

- Python 3.9+
- Redis server
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/draj222/DOCRAT-MVP.git
   cd DOCRAT-MVP
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   echo "REDIS_URL=redis://localhost:6379/0" >> .env
   ```

5. Start Redis:
   ```bash
   # Install Redis if needed
   # On macOS: brew install redis
   # On Ubuntu: sudo apt-get install redis-server
   
   # Start Redis server
   redis-server
   ```

6. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

7. Access the API at http://localhost:8000

### Running Tests

Run tests with pytest:
```bash
pytest
```

## API Endpoints

### Process a Video

```
POST /videos/process
```

Request body:
```json
{
  "url": "https://www.youtube.com/watch?v=your_video_id",
  "chunk_size_minutes": 10
}
```

Response:
```json
{
  "video_id": "your_video_id",
  "status": "pending",
  "message": "Video processing started"
}
```

### Check Processing Status

```
GET /videos/{video_id}/status
```

Response:
```json
{
  "video_id": "your_video_id",
  "status": "processing",
  "message": null
}
```

Possible status values:
- `pending`: In the processing queue
- `processing`: Currently being processed
- `completed`: Processing complete
- `error`: An error occurred

### Get Video Insights

```
GET /videos/{video_id}/insights
```

Response:
```json
{
  "video_id": "your_video_id",
  "video_info": {
    "title": "Video Title",
    "duration": "01:30:45",
    "thumbnail": "https://img.youtube.com/vi/your_video_id/maxresdefault.jpg"
  },
  "insights": [
    {
      "chunk_id": "00:00:00-00:10:00",
      "start": "00:00:00",
      "end": "00:10:00",
      "topics": ["Budget approval", "Infrastructure"],
      "decisions": ["Approved $1.2M for road repairs"],
      "concerns": ["Timeline is too long", "Cost overruns from previous projects"],
      "actions": ["Public Works to provide quarterly updates", "Select contractor by June"],
      "summary": "Council discussed and approved the road repair budget after addressing timeline concerns and implementing stronger oversight measures."
    },
    // Additional chunks...
  ]
}
```

## Production API

Deployed at:

```
https://docrat-api.onrender.com
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379/0`)
- `PORT`: Port to run the server on (default: `8000`)
- `DEBUG`: Enable debug mode (default: `False`)

## Docker Deployment

Build and run with Docker:

```bash
docker build -t docrat-api .
docker run -p 8000:8000 --env-file .env docrat-api
```

## License

[MIT License](LICENSE)
