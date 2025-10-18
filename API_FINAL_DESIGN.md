# WhisperX API - Final RESTful Design

## Overview

The WhisperX API now follows RESTful conventions with a single resource-based endpoint (`/transcribes`) that handles all CRUD operations.

## Complete API Specification

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/transcribes` | Create a new transcription job |
| **GET** | `/transcribes/{job_id}` | Get job status (includes result when completed) |
| **DELETE** | `/transcribes/{job_id}` | Delete a transcription job |
| **GET** | `/health` | Health check |
| **GET** | `/result/{job_id}` | *(Legacy - use GET /transcribes/{job_id} instead)* |

## RESTful Resource: `/transcribes`

The `/transcribes` endpoint represents a collection of transcription jobs and supports:

### 1. CREATE - Submit Job

**POST** `/transcribes`

```bash
curl -X POST https://api.example.com/transcribes \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "diarize": true,
    "min_speakers": 2,
    "max_speakers": 4
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job submitted successfully"
}
```

### 2. READ - Get Status/Result

**GET** `/transcribes/{job_id}`

```bash
curl https://api.example.com/transcribes/550e8400-e29b-41d4-a716-446655440000
```

**Response (Processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 60,
  "message": "Performing word-level alignment...",
  "updated_at": "2025-01-20T12:05:30.000000"
}
```

**Response (Completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": "Transcription complete",
  "updated_at": "2025-01-20T12:08:45.000000",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Hello and welcome to this podcast.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95, "speaker": "SPEAKER_00"},
        {"word": "and", "start": 0.6, "end": 0.7, "score": 0.98, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "word_segments": [...]
}
```

### 3. DELETE - Remove Job

**DELETE** `/transcribes/{job_id}`

```bash
curl -X DELETE https://api.example.com/transcribes/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "message": "Job deleted successfully",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Workflow

### Standard Flow

```
1. Create Job (POST)
   ↓
   Returns: job_id

2. Poll Status (GET) - repeat every 2s
   ↓
   While status is "queued" or "processing"
   ↓
   Returns: progress updates

3. Get Result (same GET)
   ↓
   When status is "completed"
   ↓
   Returns: full transcription with segments

4. (Optional) Delete Job (DELETE)
   ↓
   Clean up storage
```

### Code Example

```python
import requests
import time

BASE_URL = "https://api.example.com/transcribes"

# 1. Create job
response = requests.post(BASE_URL, json={
    "audio_url": "https://example.com/audio.mp3",
    "diarize": True
})
job_id = response.json()["job_id"]
print(f"Job created: {job_id}")

# 2. Poll until complete
while True:
    response = requests.get(f"{BASE_URL}/{job_id}")
    data = response.json()

    print(f"Status: {data['status']} ({data['progress']}%)")

    if data["status"] == "completed":
        # 3. Result is already in response
        print(f"Language: {data['language']}")
        print(f"Segments: {len(data['segments'])}")
        for seg in data["segments"]:
            print(f"  [{seg['start']:.1f}s] {seg['text']}")
        break

    elif data["status"] == "failed":
        print(f"Error: {data['error']}")
        break

    time.sleep(2)

# 4. Clean up (optional)
requests.delete(f"{BASE_URL}/{job_id}")
```

## Design Principles

### 1. Resource-Based URLs
✅ `/transcribes` represents a collection
✅ `/transcribes/{job_id}` represents a specific resource
✅ HTTP verbs (POST, GET, DELETE) define operations

### 2. Consistent Response Structure
All responses include:
- `job_id`: Unique identifier
- `status`: Current state
- `progress`: 0-100 percentage
- `message`: Human-readable status
- `updated_at`: ISO 8601 timestamp

### 3. Progressive Enhancement
- Initial response: Minimal (job_id, status)
- In-progress response: Status updates
- Completed response: Full result automatically included

### 4. Single Source of Truth
- One endpoint for both status AND result
- No need for separate `/status` and `/result` endpoints
- Atomic operation - status and result are consistent

## Benefits

### Developer Experience
- **Simpler**: One endpoint to poll instead of two
- **Predictable**: Follows REST conventions
- **Efficient**: No extra request to fetch result

### API Design
- **RESTful**: Resource-based URLs
- **Discoverable**: Intuitive endpoint structure
- **Maintainable**: Standard CRUD pattern

### Performance
- **Fewer requests**: Status check includes result
- **Atomic**: No race conditions between status and result
- **Cacheable**: GET requests can be cached

## Status States

The transcription progresses through these states:

```
queued (0%)
   ↓
downloading (10%)
   ↓
processing (30-80%)
   ↓
completed (100%) OR failed (0%)
```

**State Details:**

| Status | Progress | Description |
|--------|----------|-------------|
| `queued` | 0% | Job waiting to start |
| `downloading` | 10% | Fetching audio from URL |
| `processing` | 30-80% | ASR → Alignment → Diarization |
| `completed` | 100% | Success - result included |
| `failed` | 0% | Error - details in `error` field |

## Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Successful GET/DELETE |
| 201 | Created | Successful POST |
| 400 | Bad Request | Invalid input |
| 404 | Not Found | Job ID doesn't exist |
| 500 | Server Error | Internal error |

### Error Response Format

```json
{
  "error": "Job not found",
  "job_id": "invalid-id"
}
```

For failed jobs:
```json
{
  "job_id": "abc-123",
  "status": "failed",
  "error": "HTTPError 404: Audio file not found",
  "message": "Error: HTTPError 404: Audio file not found",
  "stack_trace": "..."
}
```

## API Versioning (Future)

If breaking changes are needed in the future:

```
/v1/transcribes  (current)
/v2/transcribes  (future)
```

## Comparison: Old vs New

### Old Design (Non-RESTful)
```
POST   /transcribe          → create
GET    /status/{job_id}     → status
GET    /result/{job_id}     → result
DELETE /delete-job/{job_id} → delete
```

**Issues:**
❌ Inconsistent naming
❌ Redundant endpoints
❌ Extra request for result
❌ Not resource-based

### New Design (RESTful)
```
POST   /transcribes         → create
GET    /transcribes/{id}    → status + result
DELETE /transcribes/{id}    → delete
```

**Improvements:**
✅ Consistent resource naming
✅ Follows REST conventions
✅ Efficient (fewer requests)
✅ Simpler mental model

## Security Considerations

Current implementation:
- No authentication (suitable for development)
- Jobs stored in-memory (Modal Dict)

Production recommendations:
- Add API key authentication
- Rate limiting per API key
- Persistent storage (database)
- Webhook notifications for completion
- HTTPS only (enforced by Modal)

## Monitoring & Observability

Track these metrics:
- Jobs created per minute
- Success/failure rate
- Average processing time
- P95/P99 latency
- Active concurrent jobs

Example logging:
```python
logger.info(f"Job {job_id}: {status} - {progress}% - {message}")
```

## Future Enhancements

Potential additions:
1. **Batch operations**: `POST /transcribes/batch`
2. **Webhooks**: Callback URL when job completes
3. **Filtering**: `GET /transcribes?status=completed&limit=10`
4. **Pagination**: `GET /transcribes?page=2&per_page=20`
5. **Partial results**: Stream segments as they're transcribed

## Summary

The WhisperX API now provides a clean, RESTful interface centered around the `/transcribes` resource. This design:

- ✅ Follows industry standards (REST)
- ✅ Simplifies client implementation
- ✅ Reduces API calls
- ✅ Improves maintainability
- ✅ Scales efficiently

**One resource, three operations, infinite possibilities.** 🚀
