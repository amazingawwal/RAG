# RAG System with ChromaDB

A production-ready Retrieval-Augmented Generation (RAG) system built with Node.js, featuring semantic chunking, vector embeddings, and AI-powered question answering.

## Features

- **Semantic Document Chunking** - Intelligently splits documents based on sentence boundaries and configurable word count
- **Vector Embeddings** - Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for high-quality text embeddings
- **ChromaDB Vector Storage** - Efficient similarity search with persistent storage
- **AI-Powered Answers** - Leverages Google's Gemini LLM for natural language responses
- **Dynamic Rechunking** - Re-process documents with different chunk sizes without re-uploading
- **Context Filtering** - Organize and query documents by context/project
- **RESTful API** - Simple HTTP endpoints for all operations

## Prerequisites

- **Node.js** v16.x or higher
- **Docker Desktop** (for ChromaDB) 
- **API Keys:**
  - HuggingFace API Key (free): https://huggingface.co/settings/tokens
  - Google Gemini API Key (free): https://aistudio.google.com/apikey

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd rag-system
npm install
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required API Keys
HF_API_KEY=hf_your_huggingface_api_key_here
GEMINI_API_KEY=AIza_your_gemini_api_key_here

# Model Configuration
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=gemini-1.5-flash

# ChromaDB Configuration
CHROMA_DB_HOST=http://localhost:8000

# Application Settings
RAG_DATA_DIR=uploads/
CHUNK_LENGTH=150
PORT=3000
```

### 3. Start ChromaDB Server

**Option A: Using Docker (Recommended)**
```bash
docker run -p 8000:8000 chromadb/chroma
```

**Option B: Using Python**
```bash
pip install chromadb
chroma run --host localhost --port 8000
```

### 4. Start the Application

```bash
node main.js
```

You should see:
```
 ChromaDB collection initialized: rag_documents
 RAG server running on http://localhost:3000
 Chunk length: 150 words
 Embedding model: sentence-transformers/all-MiniLM-L6-v2
 LLM model: gemini-1.5-flash
 Vector store: ChromaDB at http://localhost:8000
```

##  API Documentation

### Base URL
```
http://localhost:3000
```

### Endpoints

#### 1. Upload Documents

**Endpoint:** `POST /upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `files` (required): One or more text files
- `context` (optional): Context identifier for organizing documents

**Example (cURL):**
```bash
curl -X POST http://localhost:3000/upload \
  -F "files=@document1.txt" \
  -F "files=@document2.txt" \
  -F "context=project-alpha"
```

**Example (PowerShell):**
```powershell
$file = Get-Item "document.txt"
$form = @{ files = $file; context = "project-alpha" }
Invoke-RestMethod -Uri "http://localhost:3000/upload" -Method Post -Form $form
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully indexed 15 chunks",
  "context": "project-alpha",
  "chunks": 15
}
```

---

#### 2. Query Documents (Chat)

**Endpoint:** `POST /prompt`

**Content-Type:** `application/json`

**Parameters:**
- `query` (required): Your question
- `k` (optional): Number of chunks to retrieve (default: 5)
- `context` (optional): Filter by context identifier

**Example (cURL):**
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 5,
    "context": "project-alpha"
  }'
```

**Example (PowerShell):**
```powershell
$body = @{
    query = "What is machine learning?"
    k = 5
    context = "project-alpha"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3000/prompt" `
  -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "answer": "Machine learning is a branch of artificial intelligence...",
  "retrieved": [
    {
      "text": "Machine learning involves...",
      "metadata": {
        "source": "ml_guide.txt",
        "part": 0,
        "context": "project-alpha"
      },
      "distance": 0.234
    }
  ]
}
```

---

#### 3. Rechunk Documents

**Endpoint:** `POST /rechunk`

**Content-Type:** `application/json`

**Parameters:**
- `chunkLength` (required): New chunk size in words
- `context` (optional): Only rechunk documents in this context

**Example (cURL):**
```bash
curl -X POST http://localhost:3000/rechunk \
  -H "Content-Type: application/json" \
  -d '{"chunkLength": 200, "context": "project-alpha"}'
```

**Example (PowerShell):**
```powershell
$body = @{
    chunkLength = 200
    context = "project-alpha"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3000/rechunk" `
  -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully re-chunked documents with chunk length 200",
  "oldChunks": 15,
  "newChunks": 12
}
```

---

#### 4. Health Check

**Endpoint:** `GET /health`

**Example:**
```bash
curl http://localhost:3000/health
```

**Response:**
```json
{
  "status": "ok",
  "chromaDB": "connected",
  "documentsIndexed": 15,
  "chromaHost": "http://localhost:8000"
}
```

##  Testing

### End-to-End Test Script (PowerShell)

```powershell
# Create test document
@"
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks attempt to simulate the behavior of the human brain to learn from large amounts of data.
"@ | Out-File -FilePath "test_ml.txt" -Encoding utf8

# 1. Upload document
Write-Host "📤 Uploading document..." -ForegroundColor Cyan
$file = Get-Item "test_ml.txt"
$form = @{ files = $file; context = "ml-test" }
$uploadResult = Invoke-RestMethod -Uri "http://localhost:3000/upload" -Method Post -Form $form
$uploadResult | ConvertTo-Json
Start-Sleep -Seconds 2

# 2. Query the system
Write-Host "`n🔍 Querying: 'What is deep learning?'" -ForegroundColor Cyan
$queryBody = @{
    query = "What is deep learning?"
    k = 3
    context = "ml-test"
} | ConvertTo-Json

$queryResult = Invoke-RestMethod -Uri "http://localhost:3000/prompt" -Method Post -Body $queryBody -ContentType "application/json"
Write-Host "Answer: $($queryResult.answer)" -ForegroundColor Green
Start-Sleep -Seconds 2

# 3. Rechunk with different size
Write-Host "`n🔄 Rechunking with size 200..." -ForegroundColor Cyan
$rechunkBody = @{
    chunkLength = 200
    context = "ml-test"
} | ConvertTo-Json

$rechunkResult = Invoke-RestMethod -Uri "http://localhost:3000/rechunk" -Method Post -Body $rechunkBody -ContentType "application/json"
$rechunkResult | ConvertTo-Json

# 4. Query again
Write-Host "`n🔍 Querying again after rechunking..." -ForegroundColor Cyan
$queryResult2 = Invoke-RestMethod -Uri "http://localhost:3000/prompt" -Method Post -Body $queryBody -ContentType "application/json"
Write-Host "Answer: $($queryResult2.answer)" -ForegroundColor Green

Write-Host "`n✅ All tests completed!" -ForegroundColor Green
```

##  Architecture

```
┌─────────────────┐
│   User Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Express API    │
│   (main.js)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│HuggingF│  │   ChromaDB   │
│  ace   │  │Vector Storage│
└────────┘  └──────────────┘
    │              │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │Google Gemini │
    │     LLM      │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Response   │
    └──────────────┘
```

### Data Flow

1. **Document Upload**
   - User uploads text files
   - System chunks text semantically
   - Generates embeddings via HuggingFace
   - Stores in ChromaDB with metadata

2. **Query Processing**
   - User asks a question
   - Question converted to embedding
   - ChromaDB finds similar chunks (cosine similarity)
   - Retrieved chunks sent to Gemini as context
   - Gemini generates natural language answer

3. **Rechunking**
   - Retrieves all stored documents
   - Re-processes with new chunk size
   - Regenerates embeddings
   - Updates ChromaDB storage

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_API_KEY` | *required* | HuggingFace API key for embeddings |
| `GEMINI_API_KEY` | *required* | Google Gemini API key for LLM |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier |
| `LLM_MODEL_NAME` | `gemini-1.5-flash` | Gemini model to use |
| `CHROMA_DB_HOST` | `http://localhost:8000` | ChromaDB server URL |
| `RAG_DATA_DIR` | `uploads/` | Temporary file storage directory |
| `CHUNK_LENGTH` | `150` | Target chunk size in words |
| `PORT` | `3000` | Application server port |

### Recommended Chunk Sizes

- **Small (50-100 words)**: Precise retrieval, more chunks needed
- **Medium (150-200 words)**: Balanced - **recommended**
- **Large (300-500 words)**: More context per chunk, less precise

##  Performance

### Rate Limits (Free Tier)

**HuggingFace:**
- No strict rate limits on the inference API
- Recommended: Batch requests when possible

**Google Gemini (gemini-1.5-flash):**
- 15 requests per minute
- 1,000,000 tokens per minute
- 1,500 requests per day

### Optimization Tips

1. **Batch Embeddings**: Generate embeddings for multiple chunks in one API call
2. **Adjust k Parameter**: Start with k=5, adjust based on results
3. **Use Context Filtering**: Narrow search scope for faster queries
4. **Monitor Chunk Size**: Experiment to find optimal size for your use case

##  Troubleshooting

### ChromaDB Connection Errors

**Error:** `Failed to connect to chromadb`

**Solutions:**
```bash
# Check if ChromaDB is running
docker ps | grep chroma

# Restart ChromaDB
docker stop $(docker ps -q --filter ancestor=chromadb/chroma)
docker run -p 8000:8000 chromadb/chroma

# Verify connection
curl http://localhost:8000/api/v1/heartbeat
```

### Rate Limit Errors

**Error:** `429 Too Many Requests`

**Solutions:**
- Use `gemini-1.5-flash` instead of experimental models
- Implement exponential backoff (already included in code)
- Space out requests

### No Relevant Documents Found

**Possible Causes:**
- No documents uploaded yet
- Query doesn't match document content
- Chunk size too large/small

**Solutions:**
```bash
# Check document count
curl http://localhost:3000/health

# Try different chunk size
curl -X POST http://localhost:3000/rechunk \
  -H "Content-Type: application/json" \
  -d '{"chunkLength": 100}'
```

### File Upload Issues

**Error:** `No files uploaded`

**Solution:**
```bash
# Ensure correct content-type
curl -X POST http://localhost:3000/upload \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.txt"
```

##  Security Considerations

### Production Deployment Checklist

- [ ] Use environment variables for all secrets
- [ ] Enable HTTPS/TLS
- [ ] Implement rate limiting (e.g., `express-rate-limit`)
- [ ] Add authentication (JWT, OAuth)
- [ ] Validate and sanitize file uploads
- [ ] Set up CORS properly
- [ ] Use helmet.js for security headers
- [ ] Implement request logging
- [ ] Set up monitoring and alerting

### Example: Add Rate Limiting

```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/prompt', limiter);
```

##  Dependencies

```json
{
  "@google/generative-ai": "^0.24.1",
  "@huggingface/inference": "^4.13.4",
  "chromadb": "^1.9.2",
  "dotenv": "^17.2.3",
  "express": "^5.1.0",
  "multer": "^2.0.2"
}
```

##  Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ChromaDB** - Vector database for AI applications
- **HuggingFace** - Transformer models and inference API
- **Google Gemini** - Large language model for natural responses
- **Sentence Transformers** - State-of-the-art text embeddings

##  Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [Link to docs]

---

**Built with ❤️ by amazingAwwal using Node.js, ChromaDB, and AI**