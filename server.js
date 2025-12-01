// Node.js RAG template (Express)
// Endpoints: /upload, /prompt, /rechunk

const express = require('express');
const app = express();
app.use(express.json());

// Upload endpoint
app.post('/upload', (req, res) => {
  // TODO: implement upload handling
  res.json({ message: 'Upload endpoint not yet implemented' });
});

// Prompt endpoint
app.post('/prompt', (req, res) => {
  // TODO: implement prompt handling
  res.json({ message: 'Prompt endpoint not yet implemented' });
});

// Rechunk endpoint
app.post('/rechunk', (req, res) => {
  // TODO: implement rechunk handling
  res.json({ message: 'Rechunk endpoint not yet implemented' });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
