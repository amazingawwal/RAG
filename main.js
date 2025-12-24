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

//Environment variables
const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME = process.env.EMBED_MODEL_NAME || 'sentence-transformers/all-MiniLM-L6-v2';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || 'gemini-1.5-flash';
const CHROMA_DB_HOST = process.env.CHROMA_DB_HOST || '';
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || 'uploads/';
const CHUNK_LENGTH = parseInt(process.env.CHUNK_LENGTH || '150');
const PORT = process.env.PORT || 3000;


const hf = new HfInference(HF_API_KEY);
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);



let client;
if (CHROMA_DB_HOST && CHROMA_DB_HOST.trim() !== '') {
  
  client = new ChromaClient({ path: CHROMA_DB_HOST });
  console.log('Using ChromaDB server mode:', CHROMA_DB_HOST);
} else {
  
  client = new ChromaClient();
  console.log('Using ChromaDB embedded mode (in-memory)');
}


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


if (!fs.existsSync(RAG_DATA_DIR)) {
  fs.mkdirSync(RAG_DATA_DIR, { recursive: true });
}


const upload = multer({ dest: RAG_DATA_DIR });


function semanticChunk(text, targetWords = CHUNK_LENGTH) {
  
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  let currentChunk = [];
  let currentWordCount = 0;

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;

    if (currentWordCount + wordCount > targetWords && currentChunk.length > 0) {
      
      chunks.push(currentChunk.join(' '));
      currentChunk = [sentence];
      currentWordCount = wordCount;
    } else {
      currentChunk.push(sentence);
      currentWordCount += wordCount;
    }
  }

  
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(' '));
  }

  return chunks;
}


async function generateEmbeddings(texts) {
  try {
    const embeddings = await hf.featureExtraction({
      model: EMBED_MODEL_NAME,
      inputs: texts
    });
    
    
    if (Array.isArray(embeddings[0])) {
      return embeddings;
    } else {
      
      return [embeddings];
    }
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw error;
  }
}


async function callGemini(prompt, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const model = genAI.getGenerativeModel({ model: LLM_MODEL_NAME });
      const result = await model.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      console.error(`Error calling Gemini (attempt ${i + 1}/${retries}):`, error.message);
      
      
      if (error.message && error.message.includes('429')) {
        
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


app.post('/upload', upload.array('files'), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }

    const allChunks = [];
    const context = req.body.context || `ctx-${crypto.randomUUID().slice(0, 8)}`;

    
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

        
        fs.unlinkSync(file.path);
      } catch (error) {
        console.error(`Error processing file ${file.originalname}:`, error);
        
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      }
    }

    if (allChunks.length === 0) {
      return res.status(400).json({ error: 'No valid text chunks extracted from files' });
    }

    
    const embeddings = await generateEmbeddings(allChunks.map(c => c.text));

    
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


app.post('/chat', async (req, res) => {
  try {
    const { query, k = 5, context = null } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    
    const [queryEmbedding] = await generateEmbeddings([query]);

    
    const queryOptions = {
      queryEmbeddings: [queryEmbedding],
      nResults: k
    };

    
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

    
    const contextText = results.documents[0]
      .map((doc, i) => {
        const meta = results.metadatas[0][i];
        return `Source: ${meta.source} (Part ${meta.part})\n${doc}`;
      })
      .join('\n\n---\n\n');

    
    const prompt = `Use the following context to answer the question. If the answer is not in the context, say so.

Context:
${contextText}

Question: ${query}

Answer:`;

    
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

    
    await collection.delete({
      ids: allDocs.ids
    });

    
    const newChunks = [];
    for (const [key, group] of Object.entries(fileGroups)) {
      
      const originalText = group.texts.join(' ');
      
      
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

    
    const embeddings = await generateEmbeddings(newChunks.map(c => c.text));

    
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


async function startServer() {
  try {
   
    await initializeCollection();
    
    
    app.listen(PORT, () => {
      console.log(` RAG server running on http://localhost:${PORT}`);
      console.log(`Chunk length: ${CHUNK_LENGTH} words`);
      console.log(`Embedding model: ${EMBED_MODEL_NAME}`);
      console.log(`LLM model: ${LLM_MODEL_NAME}`);
      console.log(`Vector store: ChromaDB`);
      console.log(`Upload directory: ${RAG_DATA_DIR}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();