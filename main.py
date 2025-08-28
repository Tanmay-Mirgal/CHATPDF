import streamlit as st
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

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="🤖 Smart Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        margin-right: 0;
    }
    
    .bot-message {
        background: #e9ecef;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        border-left: 4px solid #28a745;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 0.5rem 0;
    }
    
    .sidebar-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    def __init__(self):
        self.embedder = None
        self.gemini_model = None
        self.pc = None
        self.index = None
        self.index_name = "smart-doc-assistant"
        
    def initialize_services(self):
        """Initialize all AI services"""
        try:
            # Get API keys
            gemini_key = os.getenv("GEMINI_API_KEY")
            pinecone_key = os.getenv("PINECONE_API_KEY")
            
            if not gemini_key or not pinecone_key:
                st.error("❌ Please set GEMINI_API_KEY and PINECONE_API_KEY in your .env file")
                return False
            
            # Initialize Gemini
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Initialize embeddings
            with st.spinner("Loading embedding model..."):
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_key)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing services: {str(e)}")
            return False
    
    def setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        try:
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name in existing_indexes:
                st.info("📊 Using existing Pinecone index...")
                self.index = self.pc.Index(self.index_name)
            else:
                with st.spinner("🔧 Creating new Pinecone index..."):
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    time.sleep(10)  # Wait for index to be ready
                    self.index = self.pc.Index(self.index_name)
                st.success("✅ Pinecone index created successfully!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error setting up Pinecone: {str(e)}")
            return False
    
    def get_embedding(self, text: str):
        """Get embedding for text"""
        return self.embedder.encode(text).tolist()
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded file"""
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(BytesIO(uploaded_file.read()))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            else:
                st.error("❌ Unsupported file type. Please upload PDF, DOCX, or TXT files.")
                return None
                
        except Exception as e:
            st.error(f"❌ Error extracting text: {str(e)}")
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
    
    def process_document(self, uploaded_file):
        """Process uploaded document and store in vector database"""
        try:
            # Extract text
            text = self.extract_text_from_file(uploaded_file)
            if not text:
                return False
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Create embeddings and store
            with st.spinner(f"Processing {len(chunks)} text chunks..."):
                vectors_to_upsert = []
                
                for i, chunk in enumerate(chunks):
                    vector = self.get_embedding(chunk)
                    doc_id = f"{uploaded_file.name}_{i}_{uuid.uuid4().hex[:8]}"
                    
                    vectors_to_upsert.append((
                        doc_id,
                        vector,
                        {
                            "text": chunk,
                            "filename": uploaded_file.name,
                            "chunk_id": i
                        }
                    ))
                
                # Upsert to Pinecone
                self.index.upsert(vectors=vectors_to_upsert)
            
            return len(chunks)
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            return False
    
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
                return "I don't have any relevant information in my knowledge base to answer your question. Please upload some documents first!"
            
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
            
            Answer:
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Upload documents and chat with your AI-powered knowledge base!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.documents_processed = 0
    
    chatbot = st.session_state.chatbot
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.title("📁 Document Management")
        
        # Initialize services
        if not st.session_state.initialized:
            if st.button("🚀 Initialize AI Services", type="primary"):
                with st.spinner("Initializing AI services..."):
                    if chatbot.initialize_services() and chatbot.setup_pinecone_index():
                        st.session_state.initialized = True
                        st.success("✅ All services initialized!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to initialize services")
        
        if st.session_state.initialized:
            st.success("🟢 Services Ready")
            
            # File upload
            st.markdown("### 📤 Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['txt', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, or TXT files"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if st.button(f"Process {uploaded_file.name}", key=uploaded_file.name):
                        chunks = chatbot.process_document(uploaded_file)
                        if chunks:
                            st.success(f"✅ Processed {chunks} chunks from {uploaded_file.name}")
                            st.session_state.documents_processed += chunks
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Stats
            st.markdown("### 📊 Statistics")
            st.markdown(f"""
            <div class="metric-card">
                <h3>{st.session_state.documents_processed}</h3>
                <p>Text Chunks Processed</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.messages)}</h3>
                <p>Chat Messages</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clear chat
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="sidebar-section">
            <h4>💡 How to use:</h4>
            <ol>
                <li>Click "Initialize AI Services"</li>
                <li>Upload your documents</li>
                <li>Click "Process" for each file</li>
                <li>Start chatting with your documents!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Main chat area
    if st.session_state.initialized:
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if user_input := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                response = chatbot.query_documents(user_input)
            
            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()
    
    else:
        st.markdown("""
        <div class="chat-container">
            <h3>🚀 Welcome to Smart Document Assistant!</h3>
            <p>To get started:</p>
            <ol>
                <li>Click <strong>"Initialize AI Services"</strong> in the sidebar</li>
                <li>Upload your documents (PDF, DOCX, or TXT)</li>
                <li>Process the documents</li>
                <li>Start chatting!</li>
            </ol>
            <p><strong>Note:</strong> Make sure you have your GEMINI_API_KEY and PINECONE_API_KEY set in a .env file.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 Powered by Gemini AI & Pinecone Vector Database | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()