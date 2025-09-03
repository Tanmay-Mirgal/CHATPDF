from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time
import hashlib
from typing import List
import PyPDF2
import docx
from io import BytesIO
import uuid
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class RAGChatbot:
    def __init__(self):
        self.embedder = None
        self.gemini_model = None
        self.pc = None
        self.index = None
        self.index_name = "smart-doc-assistant"
        self.initialized = False
        self.documents_processed = 0
        
    def initialize_services(self):
        """Initialize all AI services"""
        try:
            # Get API keys
            gemini_key = os.getenv("GEMINI_API_KEY")
            pinecone_key = os.getenv("PINECONE_API_KEY")
            
            if not gemini_key or not pinecone_key:
                return {"success": False, "error": "Missing API keys in environment variables"}
            
            # Initialize Gemini
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Initialize embeddings
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_key)
            
            # Setup index
            if self.setup_pinecone_index():
                self.initialized = True
                return {"success": True, "message": "All services initialized successfully"}
            else:
                return {"success": False, "error": "Failed to setup Pinecone index"}
            
        except Exception as e:
            return {"success": False, "error": f"Error initializing services: {str(e)}"}
    
    def setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        try:
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name in existing_indexes:
                self.index = self.pc.Index(self.index_name)
            else:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)  # Wait for index to be ready
                self.index = self.pc.Index(self.index_name)
            
            return True
            
        except Exception as e:
            print(f"Error setting up Pinecone: {str(e)}")
            return False
    
    def get_embedding(self, text: str):
        """Get embedding for text"""
        return self.embedder.encode(text).tolist()
    
    def extract_text_from_file(self, file_content, filename, content_type):
        """Extract text from uploaded file"""
        try:
            if content_type == "text/plain":
                return file_content.decode('utf-8')
            
            elif content_type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            
            elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(BytesIO(file_content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_content, filename, content_type):
        """Process uploaded document and store in vector database"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_content, filename, content_type)
            if not text:
                return {"success": False, "error": "Could not extract text from file"}
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Create embeddings and store
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                vector = self.get_embedding(chunk)
                doc_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
                
                vectors_to_upsert.append((
                    doc_id,
                    vector,
                    {
                        "text": chunk,
                        "filename": filename,
                        "chunk_id": i
                    }
                ))
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            self.documents_processed += len(chunks)
            
            return {"success": True, "chunks_processed": len(chunks)}
            
        except Exception as e:
            return {"success": False, "error": f"Error processing document: {str(e)}"}
    
    def query_documents(self, user_query: str):
        """Query the vector database and generate response"""
        try:
            # Get query embedding
            query_vector = self.get_embedding(user_query)
            
            # Search similar documents
            results = self.index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True
            )
            
            matches = results.get("matches", [])
            
            if not matches:
                return "I don't have any relevant information in my knowledge base to answer your question. Please upload some documents first! 📚"
            
            # Prepare context
            context = "\n\n".join([
                f"Source: {match['metadata']['filename']}\nContent: {match['metadata']['text']}"
                for match in matches
            ])
            
            # Generate response with Gemini
            prompt = f"""
            You are a helpful AI assistant that answers questions based on the provided document context.
            
            Context from documents:
            {context}
            
            User Question: {user_query}
            
            Instructions:
            - Answer the question based on the provided context
            - Be accurate and informative
            - If the context doesn't contain enough information, mention that
            - Cite the source documents when possible
            - Be conversational and helpful
            - Use emojis appropriately to make responses engaging
            
            Answer:
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Initialize chatbot instance
chatbot = RAGChatbot()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        "initialized": chatbot.initialized,
        "documents_processed": chatbot.documents_processed
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_services():
    """Initialize AI services"""
    result = chatbot.initialize_services()
    return jsonify(result)

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process document"""
    if not chatbot.initialized:
        return jsonify({"success": False, "error": "Services not initialized"}), 400
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Check file type
    allowed_types = [
        'text/plain',
        'application/pdf', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    
    if file.content_type not in allowed_types:
        return jsonify({"success": False, "error": "Unsupported file type"}), 400
    
    # Process the document
    file_content = file.read()
    result = chatbot.process_document(file_content, file.filename, file.content_type)
    
    return jsonify(result)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with documents"""
    if not chatbot.initialized:
        return jsonify({"error": "Services not initialized"}), 400
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message']
    response = chatbot.query_documents(user_message)
    
    return jsonify({"response": response})

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear chat history (for frontend state management)"""
    return jsonify({"success": True, "message": "Chat cleared"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    print("🚀 Starting Smart Document Assistant Backend...")
    print("📡 Server running on http://localhost:5000")
    print("🔧 Make sure to set GEMINI_API_KEY and PINECONE_API_KEY in your .env file")
    app.run(debug=True, host='0.0.0.0', port=5000)