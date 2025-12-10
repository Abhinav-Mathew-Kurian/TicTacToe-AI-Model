// trainExternal.js - Train from external CSV data
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

/**
 * EXPECTED CSV FORMAT (UCI Tic-Tac-Toe Dataset):
 * 
 * x,x,x,x,o,o,x,o,o,positive
 * x,x,x,x,o,o,o,x,o,positive
 * x,x,x,x,o,o,o,o,x,positive
 * 
 * Where:
 * - First 9 values: board positions (x, o, or b for blank)
 * - Last value: game outcome (positive/negative)
 * 
 * This script will:
 * 1. Parse the CSV
 * 2. Convert to numeric format (1=X, -1=O, 0=blank)
 * 3. Calculate optimal moves using minimax
 * 4. Train a neural network
 */

// Parse UCI format CSV
function parseCSV(csvPath) {
  console.log(`📂 Reading CSV from: ${csvPath}`);
  
  if (!fs.existsSync(csvPath)) {
    console.error(`❌ Error: File not found at ${csvPath}`);
    process.exit(1);
  }
  
  const content = fs.readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  
  console.log(`📄 Found ${lines.length} lines in CSV`);
  
  const data = [];
  let skipped = 0;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Skip empty lines or comments
    if (!line || line.startsWith('#')) {
      skipped++;
      continue;
    }
    
    const parts = line.split(',');
    
    if (parts.length < 9) {
      console.log(`⚠ Skipping line ${i + 1}: not enough values (${parts.length})`);
      skipped++;
      continue;
    }
    
    // Convert board notation to numbers
    const board = parts.slice(0, 9).map(cell => {
      const c = cell.trim().toLowerCase();
      if (c === 'x') return 1;
      if (c === 'o') return -1;
      if (c === 'b') return 0;  // 'b' means blank
      return 0;  // Default to empty
    });
    
    // Only use positions where X (player 1) can make a move
    const hasEmptySpots = board.includes(0);
    const xCount = board.filter(x => x === 1).length;
    const oCount = board.filter(x => x === -1).length;
    
    // Valid positions: X's turn (equal counts or X has one less)
    if (hasEmptySpots && (xCount === oCount || xCount === oCount + 1)) {
      const optimalMove = findBestMove(board);
      
      if (optimalMove !== null) {
        data.push({ board, move: optimalMove });
      }
    } else {
      skipped++;
    }
  }
  
  console.log(`✅ Parsed ${data.length} valid training samples`);
  if (skipped > 0) {
    console.log(`⚠ Skipped ${skipped} invalid/unusable lines`);
  }
  
  return data;
}

// Minimax algorithm to find optimal move
function findBestMove(board) {
  const available = board.map((cell, idx) => cell === 0 ? idx : -1).filter(idx => idx !== -1);
  
  if (available.length === 0) return null;
  
  let bestScore = -Infinity;
  let bestMove = available[0];
  
  for (const move of available) {
    const newBoard = [...board];
    newBoard[move] = 1; // X's move
    const score = minimax(newBoard, -1, 0, false);
    
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }
  
  return bestMove;
}

function minimax(board, player, depth, isMaximizing) {
  const winner = checkWinner(board);
  
  // Terminal states
  if (winner === 1) return 10 - depth;  // X wins (good)
  if (winner === -1) return depth - 10; // O wins (bad)
  if (winner === 0) return 0;           // Draw
  
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
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
    [0, 4, 8], [2, 4, 6]             // Diagonals
  ];

  for (const [a, b, c] of lines) {
    if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }

  return board.includes(0) ? null : 0; // null = ongoing, 0 = draw
}

// Create neural network model
function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 256, activation: 'relu', inputShape: [9] }),
      tf.layers.dropout({ rate: 0.3 }),
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
async function trainFromExternal(csvPath) {
  console.log('\n=== Training from External CSV Data ===\n');
  
  // Parse CSV
  const data = parseCSV(csvPath);
  
  if (data.length === 0) {
    console.error('❌ ERROR: No valid training data found in CSV!');
    console.log('\nExpected format:');
    console.log('x,x,x,x,o,o,x,o,o,positive');
    console.log('x,x,x,x,o,o,o,x,o,positive');
    console.log('\nWhere: x=X player, o=O player, b=blank/empty\n');
    process.exit(1);
  }
  
  // Check if we have enough data
  if (data.length < 50) {
    console.log(`⚠ Warning: Only ${data.length} samples. Model may not train well.`);
    console.log('Consider using a larger dataset or generating more data.\n');
  }
  
  // Convert to tensors
  console.log('\n🔄 Converting data to tensors...');
  const xs = tf.tensor2d(data.map(d => d.board));
  const ys = tf.tensor2d(data.map(d => {
    const oneHot = Array(9).fill(0);
    oneHot[d.move] = 1;
    return oneHot;
  }));
  
  console.log('✅ Tensors created');
  console.log(`   Input shape: [${data.length}, 9]`);
  console.log(`   Output shape: [${data.length}, 9]`);
  
  // Create model
  console.log('\n🤖 Creating neural network model...');
  const model = createModel();
  model.summary();
  
  // Train model
  console.log('\n🎯 Training model...');
  console.log('This may take 5-10 minutes depending on your CPU\n');
  
  const startTime = Date.now();
  
  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 10 === 0) {
          console.log(
            `Epoch ${epoch + 1}/100: ` +
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
  if (!fs.existsSync('./models/external')) {
    fs.mkdirSync('./models/external', { recursive: true });
  }
  await model.save('file://./models/external');
  console.log('✅ External Model saved to ./models/external/');
  
  // Test predictions
  console.log('\n🧪 Testing model predictions...');
  testModel(model, data);
  
  // Clean up
  xs.dispose();
  ys.dispose();
  
  console.log('\n=== Training Complete ===');
  console.log('\n🚀 Next steps:');
  console.log('   1. Run: npm start');
  console.log('   2. Open: http://localhost:3000');
  console.log('   3. Select "External CSV Dataset" from dropdown');
  console.log('   4. Compare with Base model!\n');
}

// Test the model with sample predictions
function testModel(model, data) {
  // Take a few random samples
  const testSamples = 3;
  const samples = [];
  for (let i = 0; i < testSamples; i++) {
    const idx = Math.floor(Math.random() * data.length);
    samples.push(data[idx]);
  }
  
  samples.forEach((sample, idx) => {
    console.log(`\nTest ${idx + 1}:`);
    console.log('Board:', sample.board);
    console.log('Expected move:', sample.move);
    
    const pred = model.predict(tf.tensor2d([sample.board]));
    const probs = pred.dataSync();
    const predictedMove = probs.indexOf(Math.max(...probs));
    
    console.log('Predicted move:', predictedMove);
    console.log('Match:', predictedMove === sample.move ? '✅' : '❌');
    
    pred.dispose();
  });
}

// Get CSV path from command line
const csvPath = process.argv[2];

if (!csvPath) {
  console.error('\n❌ Error: Please provide CSV file path\n');
  console.log('Usage:');
  console.log('  node trainExternal.js <path-to-csv>\n');
  console.log('Example:');
  console.log('  node trainExternal.js tictactoe_data.csv');
  console.log('  node trainExternal.js ./data/games.csv\n');
  process.exit(1);
}

// Start training
trainFromExternal(csvPath).catch(error => {
  console.error('\n❌ Training failed:', error.message);
  console.error(error.stack);
  process.exit(1);
});