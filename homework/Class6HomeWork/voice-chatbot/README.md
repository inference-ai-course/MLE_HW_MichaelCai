# Voice Chatbot Project

Need to run TTS visual env:
source tts_env/bin/activate

A real-time voice chatbot application with FastAPI backend and React frontend.

## Features

- ✅ Audio input via HTTP
- ✅ Speech-to-Text (ASR) using Whisper
- ✅ LLM response generation using OpenAI GPT
- ✅ Text-to-Speech (TTS) using Google TTS
- ✅ 5-turn conversational memory
- ✅ Real-time audio processing
- ✅ Web-based interface

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │
│  React Frontend │───▶│ FastAPI Backend │
│     (Port 3000) │    │   (Port 3001)   │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

Simple two-tier architecture:
- **React Frontend**: JavaScript-based UI with audio recording
- **FastAPI Backend**: Single service handling ASR, LLM, TTS, and conversation memory

## Project Structure

```
voice-chatbot/
├── frontend/          # React JavaScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── VoiceChatbot.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
└── backend/           # FastAPI backend
    ├── main.py        # All endpoints (ASR, LLM, TTS, conversation)
    ├── requirements.txt
    └── .env.example
```

## Setup Instructions

### 1. Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- OpenAI API key (optional, fallback responses will be used)

### 2. Install Dependencies

#### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend (React)
```bash
cd frontend
npm install
```

### 3. Configuration

Create `.env` file in `backend/` directory:
```bash
cp backend/.env.example backend/.env
# Edit .env and add your OpenAI API key
```

### 4. Running the Application

Start both services (run each in a separate terminal):

#### Terminal 1: FastAPI Backend
```bash
cd backend
python main.py
# Runs on http://localhost:3001
```

#### Terminal 2: React Frontend
```bash
cd frontend
npm start
# Runs on http://localhost:3000
```

### 5. Usage

1. Open http://localhost:3000 in your browser
2. Grant microphone permissions when prompted
3. Click the microphone button to start recording
4. Speak your message
5. Click "Stop Recording" when done
6. Wait for the AI response (both text display and audio playback)

## API Endpoints

### FastAPI Backend (Port 3001)

- `POST /audio/chat` - Complete workflow: upload audio → ASR → LLM → TTS → return audio
- `POST /transcribe` - Speech-to-text using Whisper only
- `POST /chat` - Text-based chat with conversation history
- `POST /tts` - Convert text to speech
- `GET /conversation/{session_id}` - Get conversation history
- `DELETE /conversation/{session_id}` - Clear conversation history
- `GET /health` - Health check

## Technical Details

### Speech Recognition
- Uses OpenAI Whisper (base model) for high-quality transcription
- Supports various audio formats via FastAPI file upload

### Language Model
- OpenAI GPT-3.5-turbo for conversational responses
- Falls back to echo responses if no API key provided
- Maintains conversation context for natural dialogue

### Text-to-Speech
- Google Text-to-Speech (gTTS) for audio generation
- Returns MP3 audio files for web playback

### Conversation Memory
- Stores last 5 conversation turns per session
- In-memory storage (can be extended to database)
- Session-based isolation using unique session IDs

### Frontend Features
- Real-time audio recording using MediaRecorder API
- Audio playback of responses
- Conversation history display with timestamps
- Error handling and user feedback
- Responsive design

## Development

### Simplified Architecture
- No TypeScript complexity - pure JavaScript for faster development
- Single FastAPI service instead of multiple services
- Minimal dependencies and setup

### Error Handling
- Comprehensive error handling in FastAPI
- User-friendly error messages in the frontend
- Graceful fallbacks when OpenAI API is unavailable

### CORS Configuration
- FastAPI backend configured for React frontend on port 3000
- Proper handling of multipart form data for audio uploads

## API Usage Examples

### Complete Audio Chat
```bash
curl -X POST "http://localhost:3001/audio/chat" \
  -F "audio=@recording.wav" \
  -F "session_id=session_123"
```

### Text Chat
```bash
curl -X POST "http://localhost:3001/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "session_123"}'
```

### Get Conversation History
```bash
curl "http://localhost:3001/conversation/session_123"
```

## Deployment Considerations

- Set appropriate CORS origins for production
- Use environment variables for API keys
- Consider using Gunicorn for production FastAPI deployment
- Implement proper logging and monitoring
- Add database persistence for conversation history
- Add authentication and rate limiting

## Troubleshooting

### Common Issues

1. **Microphone not working**: Ensure browser permissions are granted and using HTTPS in production
2. **Audio not playing**: Check browser audio settings and content security policy
3. **Python dependencies**: Ensure PyTorch and other ML libraries install correctly
4. **Port conflicts**: Make sure ports 3000 and 3001 are available

### Development Tips

1. **Audio Format**: The app accepts various audio formats but converts to WAV/MP3 internally
2. **Session Management**: Each browser session gets a unique session ID for conversation isolation
3. **Memory Usage**: Whisper model loads on startup and stays in memory for fast inference

### Logs
Check the FastAPI console output for detailed error information and request logs.