import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import './VoiceChatbot.css';

const VoiceChatbot = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
        
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone. Please ensure microphone permissions are granted.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  }, [isRecording]);

  const processAudio = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');

      const response = await axios.post('/chat_langgraph', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      // Play the response audio
      const audioUrl = URL.createObjectURL(response.data);
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
      
      audio.play().catch(err => {
        console.error('Error playing audio:', err);
        setError('Failed to play audio response');
      });
      
    } catch (err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="voice-chatbot">
      <div className="conversation-display">
        <div className="welcome-message">
          <p>Welcome! Click the microphone button to start your conversation.</p>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="controls">
        <button
          className={`record-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <span>Processing...</span>
          ) : isRecording ? (
            <span>🔴 Stop Recording</span>
          ) : (
            <span>🎤 Start Recording</span>
          )}
        </button>
        
      </div>

      <div className="status">
        {isRecording && <p>🎙️ Recording... Click "Stop Recording" when done</p>}
        {isProcessing && <p>⏳ Processing your request...</p>}
        {!isRecording && !isProcessing && <p>💬 Ready for conversation</p>}
      </div>
    </div>
  );
};

export default VoiceChatbot;