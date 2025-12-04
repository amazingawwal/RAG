require('dotenv').config();
const express = require('express');
const multer = require('multer');
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { HfInference } = require('@huggingface/inference');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(express.json());

// Load environment variables
const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME = process.env.EMBED_MODEL_NAME || 'sentence-transformers/all-MiniLM-L6-v2';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || 'gemini-2.0-flash-exp';
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || 'uploads/';
const CHUNK_LENGTH = parseInt(process.env.CHUNK_LENGTH || '150');
const PORT = process.env.PORT || 3000;

// Initialize clients
const hf = new HfInference(HF_API_KEY);
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// FAISS paths
const PY_SCRIPT = path.join(__dirname, 'embeddings_faiss.py');
const INDEX_DIR = path.join(__dirname, 'index_store');

// Ensure directories exist
if (!fs.existsSync(RAG_DATA_DIR)) {
  fs.mkdirSync(RAG_DATA_DIR, { recursive: true });
}
if (!fs.existsSync(INDEX_DIR)) {
  fs.mkdirSync(INDEX_DIR, { recursive: true });
}

// Configure multer for file uploads
const upload = multer({ dest: RAG_DATA_DIR });

// Semantic chunking function
function semanticChunk(text, targetWords = CHUNK_LENGTH) {
  // Split by sentences (basic regex for sentence boundaries)
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  let currentChunk = [];
  let currentWordCount = 0;

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;

    if (currentWordCount + wordCount > targetWords && currentChunk.length > 0) {
      // Save current chunk and start new one
      chunks.push(currentChunk.join(' '));
      currentChunk = [sentence];
      currentWordCount = wordCount;
    } else {
      currentChunk.push(sentence);
      currentWordCount += wordCount;
    }
  }

  // Add remaining chunk
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(' '));
  }

  return chunks;
}

// Generate embeddings using HuggingFace
async function generateEmbeddings(texts) {
  try {
    const embeddings = await hf.featureExtraction({
      model: EMBED_MODEL_NAME,
      inputs: texts
    });
    
    // Ensure embeddings is an array of arrays
    if (Array.isArray(embeddings[0])) {
      return embeddings;
    } else {
      // If single text, wrap in array
      return [embeddings];
    }
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw error;
  }
}

// Call Gemini LLM for generation
async function callGemini(prompt) {
  try {
    const model = genAI.getGenerativeModel({ model: LLM_MODEL_NAME });
    const result = await model.generateContent(prompt);
    return result.response.text();
  } catch (error) {
    console.error('Error calling Gemini:', error);
    throw error;
  }
}

// Upload endpoint - process and index documents
app.post('/upload', upload.array('files'), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }

    const allChunks = [];
    const context = req.body.context || `ctx-${crypto.randomUUID().replace(/-/g, '').slice(0, 8)}`;

    // Process each uploaded file
    for (const file of req.files) {
      try {
        const text = fs.readFileSync(file.path, 'utf-8');
        const chunks = semanticChunk(text, CHUNK_LENGTH);

        chunks.forEach((chunkText, index) => {
          allChunks.push({
            text: chunkText,
            metadata: {
              src: file.originalname,
              part: index,
              context: context
            },
            context: context
          });
        });

        // Clean up uploaded file
        fs.unlinkSync(file.path);
      } catch (error) {
        console.error(`Error processing file ${file.originalname}:`, error);
        // Clean up file even on error
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      }
    }

    if (allChunks.length === 0) {
      return res.status(400).json({ error: 'No valid text chunks extracted from files' });
    }

    // Generate embeddings for all chunks
    const embeddings = await generateEmbeddings(allChunks.map(c => c.text));

    // Add embeddings to chunks
    allChunks.forEach((chunk, i) => {
      chunk.embedding = embeddings[i];
    });

    // Save to FAISS index via Python script
    const indexFile = path.join(INDEX_DIR, 'to_index.json');
    fs.writeFileSync(indexFile, JSON.stringify({ chunks: allChunks }));

    // Call FAISS Python script
    const result = spawnSync('python', [PY_SCRIPT, 'index', indexFile, INDEX_DIR], {
      encoding: 'utf-8'
    });

    if (result.status !== 0) {
      console.error('FAISS indexing error:', result.stderr);
      return res.status(500).json({ error: 'Failed to index documents', details: result.stderr });
    }

    res.json({
      success: true,
      message: result.stdout.trim(),
      context: context,
      chunks: allChunks.length
    });

  } catch (error) {
    console.error('Error in /upload:', error);
    res.status(500).json({ error: error.message });
  }
});

// Prompt endpoint - query the RAG system
app.post('/prompt', async (req, res) => {
  try {
    const { query, k = 5, context = null } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Check if index exists
    const indexFile = path.join(INDEX_DIR, 'faiss.index');
    if (!fs.existsSync(indexFile)) {
      return res.json({
        answer: 'No documents have been uploaded yet. Please upload documents first.',
        retrieved: []
      });
    }

    // Generate embedding for the query
    const [queryEmbedding] = await generateEmbeddings([query]);

    // Query FAISS via Python script
    const result = spawnSync(
      'python',
      [
        PY_SCRIPT,
        'query',
        JSON.stringify({ embedding: queryEmbedding, k: k, context: context }),
        INDEX_DIR
      ],
      { encoding: 'utf-8' }
    );

    if (result.status !== 0) {
      console.error('FAISS query error:', result.stderr);
      return res.status(500).json({ error: 'Failed to query documents', details: result.stderr });
    }

    const retrieved = JSON.parse(result.stdout);

    if (!retrieved || retrieved.length === 0) {
      return res.json({
        answer: 'No relevant documents found in the knowledge base.',
        retrieved: []
      });
    }

    // Format context from retrieved documents
    const contextText = retrieved
      .map((doc) => {
        return `Source: ${doc.metadata.src}#${doc.metadata.part}\n${doc.text}`;
      })
      .join('\n\n---\n\n');

    // Create prompt for Gemini
    const prompt = `Use the following context to answer the question. If the answer is not in the context, say so.

Context:
${contextText}

Question: ${query}

Answer:`;

    // Get answer from Gemini
    const answer = await callGemini(prompt);

    res.json({
      answer: answer,
      retrieved: retrieved
    });

  } catch (error) {
    console.error('Error in /prompt:', error);
    res.status(500).json({ error: error.message });
  }
});

// Rechunk endpoint - re-process documents with new chunk length
app.post('/rechunk', async (req, res) => {
  try {
    const { chunkLength, context = null } = req.body;

    if (!chunkLength) {
      return res.status(400).json({ error: 'chunkLength is required' });
    }

    const newChunkLength = parseInt(chunkLength);
    if (isNaN(newChunkLength) || newChunkLength <= 0) {
      return res.status(400).json({ error: 'chunkLength must be a positive number' });
    }

    // Check if index exists
    const metaFile = path.join(INDEX_DIR, 'meta.json');
    if (!fs.existsSync(metaFile)) {
      return res.json({
        success: true,
        message: 'No documents found to rechunk',
        chunks: 0
      });
    }

    // Read existing metadata
    const metadata = JSON.parse(fs.readFileSync(metaFile, 'utf-8'));

    if (!metadata || metadata.length === 0) {
      return res.json({
        success: true,
        message: 'No documents found to rechunk',
        chunks: 0
      });
    }

    // Filter by context if provided
    let documentsToRechunk = metadata;
    if (context) {
      documentsToRechunk = metadata.filter(doc => doc.context === context);
    }

    if (documentsToRechunk.length === 0) {
      return res.json({
        success: true,
        message: 'No documents found with specified context',
        chunks: 0
      });
    }

    // Group documents by source file
    const fileGroups = {};
    documentsToRechunk.forEach((doc) => {
      const key = `${doc.context}-${doc.metadata.src}`;
      if (!fileGroups[key]) {
        fileGroups[key] = {
          context: doc.context,
          source: doc.metadata.src,
          texts: []
        };
      }
      fileGroups[key].texts.push(doc.text);
    });

    // Re-chunk and re-index
    const newChunks = [];
    for (const [key, group] of Object.entries(fileGroups)) {
      // Reconstruct original text (approximate)
      const originalText = group.texts.join(' ');
      
      // Re-chunk with new length
      const chunks = semanticChunk(originalText, newChunkLength);

      chunks.forEach((chunkText, index) => {
        newChunks.push({
          text: chunkText,
          metadata: {
            src: group.source,
            part: index,
            context: group.context
          },
          context: group.context
        });
      });
    }

    // Generate new embeddings
    const embeddings = await generateEmbeddings(newChunks.map(c => c.text));

    // Add embeddings to chunks
    newChunks.forEach((chunk, i) => {
      chunk.embedding = embeddings[i];
    });

    // Save to FAISS index via Python script (this will overwrite the existing index)
    const indexFile = path.join(INDEX_DIR, 'to_index.json');
    fs.writeFileSync(indexFile, JSON.stringify({ chunks: newChunks }));

    // Call FAISS Python script
    const result = spawnSync('python', [PY_SCRIPT, 'index', indexFile, INDEX_DIR], {
      encoding: 'utf-8'
    });

    if (result.status !== 0) {
      console.error('FAISS re-indexing error:', result.stderr);
      return res.status(500).json({ error: 'Failed to re-index documents', details: result.stderr });
    }

    res.json({
      success: true,
      message: `Successfully re-chunked documents with chunk length ${newChunkLength}`,
      oldChunks: documentsToRechunk.length,
      newChunks: newChunks.length
    });

  } catch (error) {
    console.error('Error in /rechunk:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  const indexExists = fs.existsSync(path.join(INDEX_DIR, 'faiss.index'));
  let documentCount = 0;
  
  if (indexExists) {
    try {
      const metaFile = path.join(INDEX_DIR, 'meta.json');
      if (fs.existsSync(metaFile)) {
        const metadata = JSON.parse(fs.readFileSync(metaFile, 'utf-8'));
        documentCount = metadata.length;
      }
    } catch (error) {
      console.error('Error reading metadata:', error);
    }
  }

  res.json({ 
    status: 'ok',
    indexExists: indexExists,
    documentsIndexed: documentCount
  });
});

// Start server
app.listen(PORT, () => {
  console.log(` RAG server running on http://localhost:${PORT}`);
  console.log(` Chunk length: ${CHUNK_LENGTH} words`);
  console.log(` Embedding model: ${EMBED_MODEL_NAME}`);
  console.log(` LLM model: ${LLM_MODEL_NAME}`);
});