import React from 'react';
import VoiceChatbot from './components/VoiceChatbot';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Voice Chatbot</h1>
        <p>Click the microphone to start a voice conversation</p>
      </header>
      <main>
        <VoiceChatbot />
      </main>
    </div>
  );
}

export default App;