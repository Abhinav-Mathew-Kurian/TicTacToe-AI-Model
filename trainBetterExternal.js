// trainBetterExternal.js - Train from Kaggle's 255K game dataset
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

/**
 * Kaggle Tic-Tac-Toe Dataset Format:
 * Winner,Move 1-X (Row-Col),Move 2-O (Row-Col),Move 3-X (Row-Col),...
 * X,0-0,0-1,1-0,0-2,2-0,---,---,---,---
 * 
 * Row-Col format: 0-0 to 2-2 (3x3 grid)
 * --- means no move (game ended)
 */

// Convert Row-Col to board index (0-8)
function rowColToIndex(rowCol) {
  if (rowCol === '---') return -1;
  const [row, col] = rowCol.split('-').map(Number);
  return row * 3 + col;
}

// Parse Kaggle CSV format
function parseKaggleCSV(csvPath) {
  console.log(`📂 Reading Kaggle dataset from: ${csvPath}`);
  
  if (!fs.existsSync(csvPath)) {
    console.error(`❌ Error: File not found at ${csvPath}`);
    process.exit(1);
  }
  
  const content = fs.readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  
  console.log(`📄 Found ${lines.length} lines in CSV`);
  
  // Skip header line
  const dataLines = lines.slice(1);
  
  const trainingData = [];
  let skipped = 0;
  let gamesProcessed = 0;
  
  console.log('\n🔄 Processing games...');
  
  for (let i = 0; i < dataLines.length; i++) {
    const line = dataLines[i].trim();
    if (!line) {
      skipped++;
      continue;
    }
    
    const parts = line.split(',');
    if (parts.length < 2) {
      skipped++;
      continue;
    }
    
    // Extract moves (skip Winner column)
    const moves = parts.slice(1);
    
    // Reconstruct the game move by move
    const board = Array(9).fill(0);
    let currentPlayer = 1; // X starts
    
    for (let moveIdx = 0; moveIdx < moves.length; moveIdx++) {
      const move = moves[moveIdx];
      
      if (move === '---') break; // Game ended
      
      const boardIdx = rowColToIndex(move);
      
      if (boardIdx === -1 || board[boardIdx] !== 0) {
        // Invalid move, skip this game
        skipped++;
        break;
      }
      
      // Before making the move, save current state and optimal next move
      // Only save X's positions (player 1)
      if (currentPlayer === 1) {
        const optimalMove = findBestMove([...board], currentPlayer);
        
        if (optimalMove !== null) {
          trainingData.push({
            board: [...board],
            move: optimalMove
          });
        }
      }
      
      // Make the move
      board[boardIdx] = currentPlayer;
      
      // Check if game ended
      if (checkWinner(board) !== null) {
        break;
      }
      
      // Switch player
      currentPlayer = -currentPlayer;
    }
    
    gamesProcessed++;
    
    if ((gamesProcessed) % 50000 === 0) {
      console.log(`   Processed ${gamesProcessed.toLocaleString()} games...`);
    }
  }
  
  console.log(`\n✅ Parsed ${trainingData.length.toLocaleString()} training samples from ${gamesProcessed.toLocaleString()} games`);
  if (skipped > 0) {
    console.log(`⚠ Skipped ${skipped.toLocaleString()} invalid lines/games`);
  }
  
  return trainingData;
}

// Minimax to find optimal move
function findBestMove(board, player) {
  const available = board.map((cell, idx) => cell === 0 ? idx : -1).filter(idx => idx !== -1);
  
  if (available.length === 0) return null;
  
  let bestScore = player === 1 ? -Infinity : Infinity;
  let bestMove = available[0];
  
  for (const move of available) {
    const newBoard = [...board];
    newBoard[move] = player;
    const score = minimax(newBoard, -player, 0, player === -1);
    
    if ((player === 1 && score > bestScore) || (player === -1 && score < bestScore)) {
      bestScore = score;
      bestMove = move;
    }
  }
  
  return bestMove;
}

function minimax(board, player, depth, isMaximizing) {
  const winner = checkWinner(board);
  
  if (winner === 1) return 10 - depth;
  if (winner === -1) return depth - 10;
  if (winner === 0) return 0;
  
  const available = board.map((cell, idx) => cell === 0 ? idx : -1).filter(idx => idx !== -1);
  
  if (isMaximizing) {
    let bestScore = -Infinity;
    for (const move of available) {
      const newBoard = [...board];
      newBoard[move] = 1;
      const score = minimax(newBoard, -1, depth + 1, false);
      bestScore = Math.max(score, bestScore);
    }
    return bestScore;
  } else {
    let bestScore = Infinity;
    for (const move of available) {
      const newBoard = [...board];
      newBoard[move] = -1;
      const score = minimax(newBoard, 1, depth + 1, true);
      bestScore = Math.min(score, bestScore);
    }
    return bestScore;
  }
}

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

// Create larger model for more data
function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 512, activation: 'relu', inputShape: [9] }),
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 256, activation: 'relu' }),
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 64, activation: 'relu' }),
      tf.layers.dense({ units: 9, activation: 'softmax' })
    ]
  });

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Main training function
async function trainFromKaggle(csvPath) {
  console.log('\n=== Training from Kaggle 255K Game Dataset ===\n');
  
  const startTime = Date.now();
  
  // Parse CSV
  const data = parseKaggleCSV(csvPath);
  
  if (data.length === 0) {
    console.error('❌ ERROR: No valid training data found in CSV!');
    process.exit(1);
  }
  
  console.log(`\n📊 Dataset Statistics:`);
  console.log(`   Total samples: ${data.length.toLocaleString()}`);
  console.log(`   Expected training time: ${Math.round(data.length / 20000)} - ${Math.round(data.length / 10000)} minutes\n`);
  
  // Optionally sample data if too large (for faster training on old laptop)
  let trainingData = data;
  const MAX_SAMPLES = 50000; // Limit for old laptop
  
  if (data.length > MAX_SAMPLES) {
    console.log(`⚡ Sampling ${MAX_SAMPLES.toLocaleString()} random samples for faster training...`);
    console.log(`   (Full dataset training would take ${Math.round(data.length / 10000)} minutes)\n`);
    
    // Random sampling
    trainingData = [];
    const indices = new Set();
    while (indices.size < MAX_SAMPLES) {
      indices.add(Math.floor(Math.random() * data.length));
    }
    indices.forEach(idx => trainingData.push(data[idx]));
  }
  
  // Convert to tensors
  console.log('🔄 Converting data to tensors...');
  const xs = tf.tensor2d(trainingData.map(d => d.board));
  const ys = tf.tensor2d(trainingData.map(d => {
    const oneHot = Array(9).fill(0);
    oneHot[d.move] = 1;
    return oneHot;
  }));
  
  console.log('✅ Tensors created');
  console.log(`   Input shape: [${trainingData.length.toLocaleString()}, 9]`);
  console.log(`   Output shape: [${trainingData.length.toLocaleString()}, 9]`);
  
  // Create model
  console.log('\n🤖 Creating larger neural network model...');
  const model = createModel();
  model.summary();
  
  // Train model
  console.log('\n🎯 Training model (this will take 10-20 minutes)...\n');
  
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 128,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 5 === 0) {
          console.log(
            `Epoch ${epoch + 1}/50: ` +
            `loss=${logs.loss.toFixed(4)}, ` +
            `acc=${logs.acc.toFixed(4)}, ` +
            `val_loss=${logs.val_loss.toFixed(4)}, ` +
            `val_acc=${logs.val_acc.toFixed(4)}`
          );
        }
      }
    }
  });
  
  const trainingTime = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
  console.log(`\n⏱ Training completed in ${trainingTime} minutes`);
  
  // Save model
  console.log('\n💾 Saving model...');
  if (!fs.existsSync('./models/kaggle')) {
    fs.mkdirSync('./models/kaggle', { recursive: true });
  }
  await model.save('file://./models/kaggle');
  console.log('✅ Kaggle Model saved to ./models/kaggle/');
  
  // Test predictions
  console.log('\n🧪 Testing model predictions...');
  testModel(model, trainingData);
  
  // Clean up
  xs.dispose();
  ys.dispose();
  
  console.log('\n=== Training Complete ===');
  console.log('\n🚀 Next steps:');
  console.log('   1. Run: npm start');
  console.log('   2. Open: http://localhost:3000');
  console.log('   3. Select "Kaggle 255K Dataset" from dropdown');
  console.log('   4. This AI should be MUCH harder to beat!\n');
  
  console.log('📊 Expected Performance:');
  console.log('   Base Model:     71% player win rate (easiest)');
  console.log('   External UCI:   80% player win rate (easy)');
  console.log('   Kaggle 255K:    40-50% player win rate (HARD!) 🔥\n');
}

// Test the model
function testModel(model, data) {
  const testSamples = 5;
  const samples = [];
  for (let i = 0; i < testSamples; i++) {
    const idx = Math.floor(Math.random() * data.length);
    samples.push(data[idx]);
  }
  
  let correct = 0;
  samples.forEach((sample, idx) => {
    const pred = model.predict(tf.tensor2d([sample.board]));
    const probs = pred.dataSync();
    const predictedMove = probs.indexOf(Math.max(...probs));
    
    const match = predictedMove === sample.move;
    if (match) correct++;
    
    console.log(`Test ${idx + 1}: ${match ? '✅' : '❌'} (expected: ${sample.move}, got: ${predictedMove})`);
    pred.dispose();
  });
  
  console.log(`\nAccuracy on test samples: ${(correct/testSamples * 100).toFixed(0)}%`);
}

// Get CSV path from command line
const csvPath = process.argv[2] || 'tictactoe_games.csv';

if (!fs.existsSync(csvPath)) {
  console.error(`\n❌ Error: File not found: ${csvPath}\n`);
  console.log('Usage:');
  console.log('  node trainBetterExternal.js <path-to-kaggle-csv>\n');
  console.log('Example:');
  console.log('  node trainBetterExternal.js tictactoe_games.csv\n');
  process.exit(1);
}

// Check file size
const stats = fs.statSync(csvPath);
const sizeMB = (stats.size / 1024 / 1024).toFixed(2);
console.log(`\n📁 File: ${csvPath} (${sizeMB} MB)`);

if (stats.size < 100000) {
  console.log('\n⚠️  Warning: File seems small. Expected Kaggle dataset is ~50MB+');
  console.log('   Make sure you downloaded the correct file from:');
  console.log('   https://www.kaggle.com/datasets/anthonytherrien/tic-tac-toe-game-dataset\n');
}

// Start training
trainFromKaggle(csvPath).catch(error => {
  console.error('\n❌ Training failed:', error.message);
  console.error(error.stack);
  process.exit(1);
});