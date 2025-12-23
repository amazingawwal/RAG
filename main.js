require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { HfInference } = require('@huggingface/inference');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { ChromaClient } = require('chromadb');

const app = express();
app.use(express.json());

// Load environment variables
const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME = process.env.EMBED_MODEL_NAME || 'sentence-transformers/all-MiniLM-L6-v2';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || 'gemini-1.5-flash';
const CHROMA_DB_HOST = process.env.CHROMA_DB_HOST || '';
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || 'uploads/';
const CHUNK_LENGTH = parseInt(process.env.CHUNK_LENGTH || '150');
const PORT = process.env.PORT || 3000;

// Initialize clients
const hf = new HfInference(HF_API_KEY);
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// Step 1: Create ChromaDB client
let client;
if (CHROMA_DB_HOST && CHROMA_DB_HOST.trim() !== '') {
  // Server mode - connect to external ChromaDB server
  client = new ChromaClient({ path: CHROMA_DB_HOST });
  console.log('Using ChromaDB server mode:', CHROMA_DB_HOST);
} else {
  // Embedded mode - run ChromaDB in-process
  client = new ChromaClient();
  console.log('Using ChromaDB embedded mode (in-memory)');
}

// Step 2: Create/get the collection
let collection;
const COLLECTION_NAME = 'rag_documents';

async function initializeCollection() {
  try {
    collection = await client.getOrCreateCollection({
      name: COLLECTION_NAME,
      metadata: { 'hnsw:space': 'cosine' }
    });
    console.log('✅ ChromaDB collection initialized:', COLLECTION_NAME);
  } catch (error) {
    console.error('❌ Error initializing ChromaDB collection:', error.message);
    throw error;
  }
}

// Ensure upload directory exists
if (!fs.existsSync(RAG_DATA_DIR)) {
  fs.mkdirSync(RAG_DATA_DIR, { recursive: true });
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

// Call Gemini LLM for generation with retry logic
async function callGemini(prompt, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const model = genAI.getGenerativeModel({ model: LLM_MODEL_NAME });
      const result = await model.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      console.error(`Error calling Gemini (attempt ${i + 1}/${retries}):`, error.message);
      
      // Check if it's a rate limit error
      if (error.message && error.message.includes('429')) {
        // Extract wait time from error message
        const waitMatch = error.message.match(/Please retry in ([\d.]+)s/);
        const waitTime = waitMatch ? Math.ceil(parseFloat(waitMatch[1])) : 60;
        
        if (i < retries - 1) {
          console.log(`Rate limit hit. Waiting ${waitTime} seconds before retry...`);
          await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
          continue;
        }
      }
      
      throw error;
    }
  }
}

// Upload endpoint - process and index documents
app.post('/upload', upload.array('files'), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }

    const allChunks = [];
    const context = req.body.context || `ctx-${crypto.randomUUID().slice(0, 8)}`;

    // Process each uploaded file
    for (const file of req.files) {
      try {
        const text = fs.readFileSync(file.path, 'utf-8');
        const chunks = semanticChunk(text, CHUNK_LENGTH);

        chunks.forEach((chunkText, index) => {
          allChunks.push({
            text: chunkText,
            metadata: {
              source: file.originalname,
              part: index,
              context: context
            }
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

    // Step 3: Add embeddings to ChromaDB collection
    const ids = allChunks.map((_, i) => `${context}-${i}-${Date.now()}`);
    const documents = allChunks.map(c => c.text);
    const metadatas = allChunks.map(c => c.metadata);

    await collection.add({
      ids: ids,
      embeddings: embeddings,
      documents: documents,
      metadatas: metadatas
    });

    res.json({
      success: true,
      message: `Successfully indexed ${allChunks.length} chunks`,
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

    // Generate embedding for the query
    const [queryEmbedding] = await generateEmbeddings([query]);

    // Step 4: Query ChromaDB collection
    const queryOptions = {
      queryEmbeddings: [queryEmbedding],
      nResults: k
    };

    // Add context filter if provided
    if (context) {
      queryOptions.where = { context: context };
    }

    const results = await collection.query(queryOptions);

    if (!results.documents || !results.documents[0] || results.documents[0].length === 0) {
      return res.json({
        answer: 'No relevant documents found in the knowledge base.',
        retrieved: []
      });
    }

    // Format context from retrieved documents
    const contextText = results.documents[0]
      .map((doc, i) => {
        const meta = results.metadatas[0][i];
        return `Source: ${meta.source} (Part ${meta.part})\n${doc}`;
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
      retrieved: results.documents[0].map((doc, i) => ({
        text: doc,
        metadata: results.metadatas[0][i],
        distance: results.distances ? results.distances[0][i] : null
      }))
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

    // Get all existing documents from ChromaDB
    const getOptions = {};
    if (context) {
      getOptions.where = { context: context };
    }

    const allDocs = await collection.get(getOptions);

    if (!allDocs.documents || allDocs.documents.length === 0) {
      return res.json({
        success: true,
        message: 'No documents found to rechunk',
        chunks: 0
      });
    }

    // Group documents by source file
    const fileGroups = {};
    allDocs.documents.forEach((doc, i) => {
      const meta = allDocs.metadatas[i];
      const key = `${meta.context}-${meta.source}`;
      if (!fileGroups[key]) {
        fileGroups[key] = {
          context: meta.context,
          source: meta.source,
          texts: []
        };
      }
      fileGroups[key].texts.push(doc);
    });

    // Delete old chunks from ChromaDB
    await collection.delete({
      ids: allDocs.ids
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
            source: group.source,
            part: index,
            context: group.context
          }
        });
      });
    }

    // Generate new embeddings
    const embeddings = await generateEmbeddings(newChunks.map(c => c.text));

    // Add back to ChromaDB
    const ids = newChunks.map((_, i) => `rechunk-${i}-${Date.now()}`);
    const documents = newChunks.map(c => c.text);
    const metadatas = newChunks.map(c => c.metadata);

    await collection.add({
      ids: ids,
      embeddings: embeddings,
      documents: documents,
      metadatas: metadatas
    });

    res.json({
      success: true,
      message: `Successfully re-chunked documents with chunk length ${newChunkLength}`,
      oldChunks: allDocs.documents.length,
      newChunks: newChunks.length
    });

  } catch (error) {
    console.error('Error in /rechunk:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const count = await collection.count();
    res.json({ 
      status: 'ok',
      chromaDB: 'connected',
      documentsIndexed: count
    });
  } catch (error) {
    res.json({
      status: 'error',
      chromaDB: 'disconnected',
      error: error.message
    });
  }
});

// Start server
async function startServer() {
  try {
    // Initialize ChromaDB collection first
    await initializeCollection();
    
    // Start Express server
    app.listen(PORT, () => {
      console.log(`✅ RAG server running on http://localhost:${PORT}`);
      console.log(`📊 Chunk length: ${CHUNK_LENGTH} words`);
      console.log(`🤖 Embedding model: ${EMBED_MODEL_NAME}`);
      console.log(`🧠 LLM model: ${LLM_MODEL_NAME}`);
      console.log(`💾 Vector store: ChromaDB`);
      console.log(`📁 Upload directory: ${RAG_DATA_DIR}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();