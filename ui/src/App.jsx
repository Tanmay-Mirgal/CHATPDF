import React, { useState, useEffect, useRef } from 'react';
import { Upload, MessageCircle, Trash2, CheckCircle, Loader2, Send } from 'lucide-react';

const SmartDocumentAssistant = () => {
  const [initialized, setInitialized] = useState(false);
  const [documentsProcessed, setDocumentsProcessed] = useState(0);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const API_BASE_URL = 'http://localhost:5000/api';
  // const API_BASE_URL = 'https://172df50ffbdb.ngrok-free.app/api';

  useEffect(() => {
    checkStatus();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      const data = await response.json();
      setInitialized(data.initialized);
      setDocumentsProcessed(data.documents_processed);
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  const initializeServices = async () => {
    setIsLoading(true);
    setUploadStatus('Initializing...');
    
    try {
      const response = await fetch(`${API_BASE_URL}/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (data.success) {
        setInitialized(true);
        setUploadStatus('Ready');
        setTimeout(() => setUploadStatus(''), 2000);
      } else {
        setUploadStatus(`Error: ${data.error}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    
    for (const file of files) {
      await uploadFile(file);
    }
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    setUploadStatus(`Uploading ${file.name}...`);
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setDocumentsProcessed(prev => prev + data.chunks_processed);
        setUploadStatus(`${file.name} uploaded`);
        setTimeout(() => setUploadStatus(''), 3000);
      } else {
        setUploadStatus(`Failed: ${data.error}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !initialized) return;
    
    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: inputMessage })
      });
      
      const data = await response.json();
      
      if (data.response) {
        const botMessage = { role: 'assistant', content: data.response };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage = { role: 'assistant', content: `Error: ${data.error || 'Unknown error'}` };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = { role: 'assistant', content: `Error: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="h-screen bg-gray-100 flex flex-col">
      
      {/* Top Bar */}
      <div className="bg-white border-b p-4 flex items-center justify-between">
        <h1 className="text-xl font-medium">Document Chat</h1>
        
        <div className="flex items-center space-x-4">
          {!initialized ? (
            <button
              onClick={initializeServices}
              disabled={isLoading}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Starting...' : 'Start'}
            </button>
          ) : (
            <>
              <span className="text-sm text-gray-600">{documentsProcessed} chunks</span>
              
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
              >
                <Upload className="w-4 h-4 inline mr-1" />
                Upload
              </button>
              
              <button
                onClick={clearChat}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
              >
                <Trash2 className="w-4 h-4 inline mr-1" />
                Clear
              </button>
            </>
          )}
        </div>
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Status Bar */}
      {uploadStatus && (
        <div className="bg-blue-50 border-b px-4 py-2">
          <div className="text-sm text-blue-700">{uploadStatus}</div>
        </div>
      )}

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {!initialized ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-lg font-medium mb-2">Welcome</h2>
              <p className="text-gray-600">Click "Start" to initialize the system</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <MessageCircle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-600">Upload documents and start chatting</p>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border text-gray-900'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{message.content}</div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border p-3 rounded-lg">
                  <div className="flex items-center text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Thinking...
                  </div>
                </div>
              </div>
            )}
            
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      {initialized && (
        <div className="bg-white border-t p-4">
          <div className="max-w-4xl mx-auto flex space-x-3">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about your documents..."
              className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SmartDocumentAssistant;