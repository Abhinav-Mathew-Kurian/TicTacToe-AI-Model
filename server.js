// server.js - Express server with 3 AI Models
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Store three models
const models = {
  base: null,      // 85% base model (21K samples)
  external: null,  // External UCI data (958 samples)
  kaggle: null     // Kaggle dataset (255K games!)
};

let currentModel = 'base'; // Default model

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Load all available models
async function loadModels() {
  console.log('\n🤖 Loading AI Models...\n');
  
  // Try to load base model
  try {
    if (fs.existsSync('./models/base/model.json')) {
      models.base = await tf.loadLayersModel('file://./models/base/model.json');
      console.log('✓ Base Model (85% - 21K samples) loaded');
    } else if (fs.existsSync('./models/easy/model.json')) {
      models.base = await tf.loadLayersModel('file://./models/easy/model.json');
      console.log('✓ Base Model (85%) loaded from easy folder');
    } else if (fs.existsSync('./tictactoe-model/model.json')) {
      models.base = await tf.loadLayersModel('file://./tictactoe-model/model.json');
      console.log('✓ Base Model (85%) loaded from default location');
    }
  } catch (error) {
    console.log('⚠ Base model not found');
  }
  
  // Try to load external UCI model
  try {
    if (fs.existsSync('./models/external/model.json')) {
      models.external = await tf.loadLayersModel('file://./models/external/model.json');
      console.log('✓ External UCI Model (958 samples) loaded');
    }
  } catch (error) {
    console.log('⚠ External UCI model not found');
  }
  
  // Try to load Kaggle model
  try {
    if (fs.existsSync('./models/kaggle/model.json')) {
      models.kaggle = await tf.loadLayersModel('file://./models/kaggle/model.json');
      console.log('✓ Kaggle Model (255K games!) loaded');
    }
  } catch (error) {
    console.log('⚠ Kaggle model not found - run: node trainBetterExternal.js tictactoe_games.csv');
  }
  
  // Set default to first available model
  if (models.base) currentModel = 'base';
  else if (models.kaggle) currentModel = 'kaggle';
  else if (models.external) currentModel = 'external';
  else {
    console.error('\n❌ No models found! Please train a model first.');
    console.log('\nRun one of these commands:');
    console.log('  node train.js                              (Base model)');
    console.log('  node trainExternal.js tictactoe_data.csv   (UCI model)');
    console.log('  node trainBetterExternal.js tictactoe_games.csv (Kaggle model)\n');
    process.exit(1);
  }
  
  console.log(`\n🎯 Current Model: ${currentModel.toUpperCase()}\n`);
}

// AI prediction function
function predictMove(board) {
  const model = models[currentModel];
  
  if (!model) {
    throw new Error(`Model "${currentModel}" not loaded`);
  }
  
  // Convert board to tensor
  const input = tf.tensor2d([board], [1, 9]);
  
  // Get prediction
  const prediction = model.predict(input);
  const probabilities = prediction.dataSync();
  
  // Get available moves
  const availableMoves = board
    .map((cell, idx) => cell === 0 ? idx : -1)
    .filter(idx => idx !== -1);
  
  // Filter predictions to only available moves
  const validMoves = availableMoves.map(idx => ({
    move: idx,
    probability: probabilities[idx]
  }));
  
  // Sort by probability and pick best move
  validMoves.sort((a, b) => b.probability - a.probability);
  
  // Clean up tensors
  input.dispose();
  prediction.dispose();
  
  return validMoves[0].move;
}

// Check winner
function checkWinner(board) {
  const lines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
  ];

  for (const [a, b, c] of lines) {
    if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }

  return board.includes(0) ? null : 0;
}

// API Routes

// Get available models
app.get('/api/models', (req, res) => {
  const availableModels = Object.keys(models).filter(key => models[key] !== null);
  res.json({
    available: availableModels,
    current: currentModel
  });
});

// Switch model
app.post('/api/switch-model', (req, res) => {
  const { model } = req.body;
  
  if (!models[model]) {
    return res.status(400).json({ 
      error: `Model "${model}" not available`,
      available: Object.keys(models).filter(key => models[key] !== null)
    });
  }
  
  currentModel = model;
  console.log(`🔄 Switched to ${model.toUpperCase()} model`);
  
  res.json({ 
    success: true, 
    currentModel: model 
  });
});

// Make AI move
app.post('/api/move', (req, res) => {
  try {
    const { board } = req.body;
    
    if (!board || board.length !== 9) {
      return res.status(400).json({ error: 'Invalid board state' });
    }
    
    // Check if game is already over
    const winner = checkWinner(board);
    if (winner !== null) {
      return res.json({ move: null, winner });
    }
    
    // Get AI move
    const move = predictMove(board);
    
    // Make the move
    const newBoard = [...board];
    newBoard[move] = -1; // AI is -1 (O)
    
    // Check winner after AI move
    const newWinner = checkWinner(newBoard);
    
    res.json({
      move,
      board: newBoard,
      winner: newWinner,
      model: currentModel
    });
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok',
    currentModel,
    modelsLoaded: Object.keys(models).filter(key => models[key] !== null)
  });
});

// Serve the game
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`🎮 Tic-Tac-Toe AI Server running at http://localhost:${PORT}`);
    console.log(`📊 Current Model: ${currentModel.toUpperCase()}`);
    console.log('\nAvailable models:');
    Object.keys(models).forEach(key => {
      if (models[key]) {
        const labels = {
          base: 'Base (21K samples - Self-play)',
          external: 'External UCI (958 samples)',
          kaggle: 'Kaggle (255K games!) 🔥'
        };
        console.log(`  ✓ ${key} - ${labels[key]}`);
      }
    });
    console.log('\nOpen your browser and start playing!\n');
  });
});