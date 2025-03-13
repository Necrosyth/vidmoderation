// results-api.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Store results in memory (just for demo purposes)
let latestResult = null;
let allResults = [];

// Endpoint to receive results from the Python API
app.post('/results', (req, res) => {
  console.log('Received video processing result:');
  console.log(JSON.stringify(req.body, null, 2));
  
  // Store the result
  latestResult = req.body;
  allResults.push(req.body);
  
  return res.status(200).json({ 
    success: true, 
    message: 'Result received successfully'
  });
});

// Endpoint to get the latest result
app.get('/results/latest', (req, res) => {
  if (latestResult) {
    return res.status(200).json(latestResult);
  } else {
    return res.status(404).json({ message: 'No results received yet' });
  }
});

// Endpoint to get all results
app.get('/results', (req, res) => {
  return res.status(200).json(allResults);
});

// Start the server
app.listen(PORT, () => {
  console.log(`Results API running on http://localhost:${PORT}`);
  console.log('Ready to receive video processing results');
});