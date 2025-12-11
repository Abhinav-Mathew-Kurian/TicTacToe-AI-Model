// trainBetterExternal.js - Train from Kaggle's 255K game dataset with Data Augmentation
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

/**
 * Kaggle Tic-Tac-Toe Dataset Format:
 * Winner,Move 1-X (Row-Col),Move 2-O (Row-Col),Move 3-X (Row-Col),...
 * X,0-0,0-1,1-0,0-2,2-0,---,---,---,---
 */

// Convert Row-Col to board index (0-8)
function rowColToIndex(rowCol) {
  if (rowCol === '---') return -1;
  const [row, col] = rowCol.split('-').map(Number);
  return row * 3 + col;
}

// Check for winner (needed to correctly track game state)
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

// --- Symmetry Functions for Data Augmentation (Crucial for high accuracy) ---

// Rotate board clockwise 90 degrees
function rotateBoard(board) {
    return [
        board[6], board[3], board[0],
        board[7], board[4], board[1],
        board[8], board[5], board[2]
    ];
}

// Get the new index after rotating an old move index
function rotateMove(index) {
    const row = Math.floor(index / 3);
    const col = index % 3;
    const newRow = col;
    const newCol = 2 - row;
    return newRow * 3 + newCol;
}

// Flip board horizontally
function flipBoard(board) {
    return [
        board[2], board[1], board[0],
        board[5], board[4], board[3],
        board[8], board[7], board[6]
    ];
}

// Get the new index after flipping an old move index
function flipMove(index) {
    const row = Math.floor(index / 3);
    const col = index % 3;
    const newCol = 2 - col;
    return row * 3 + newCol;
}

// Generate all 8 symmetries (4 rotations x 2 flips)
function generateSymmetries(board, move) {
    const symmetries = [];
    let currentBoard = [...board];
    let currentMove = move;

    for (let rotation = 0; rotation < 4; rotation++) {
        // 1. Add current rotation (original or 90/180/270 rotated)
        symmetries.push({ board: [...currentBoard], move: currentMove });

        // 2. Add flipped version of the current rotation
        const flippedBoard = flipBoard(currentBoard);
        const flippedMove = flipMove(currentMove);
        symmetries.push({ board: flippedBoard, move: flippedMove });

        // Rotate for the next iteration
        currentBoard = rotateBoard(currentBoard);
        currentMove = rotateMove(currentMove);
    }
    return symmetries;
}
// --- End Symmetry Functions ---


// Parse Kaggle CSV format - uses symmetry augmentation
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
  
  console.log('\n🔄 Processing games and applying 8x symmetry augmentation...');
  
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
      
      // *** MODIFIED LOGIC: GENERATE ALL 8 SYMMETRIES ***
      if (currentPlayer === 1) {
        // We only train for player X (1)
        const symmetries = generateSymmetries(board, boardIdx);
        trainingData.push(...symmetries);
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
    
    // Provide an update frequently
    if ((gamesProcessed) % 10000 === 0) {
       console.log(`   Processed ${gamesProcessed.toLocaleString()} games. Found ${trainingData.length.toLocaleString()} augmented samples...`);
    }
  }
  
  console.log(`\n✅ Parsed ${trainingData.length.toLocaleString()} augmented training samples from ${gamesProcessed.toLocaleString()} games`);
  if (skipped > 0) {
    console.log(`⚠ Skipped ${skipped.toLocaleString()} invalid lines/games`);
  }
  
  return trainingData;
}


// Create smaller, optimized model for better generalization with augmented data
function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 256, activation: 'relu', inputShape: [9] }), 
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 128, activation: 'relu' }), 
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 64, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.2 }),
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
  console.log('\n=== Training from Kaggle 255K Game Dataset (Augmented) ===\n');
  
  const startTime = Date.now();
  
  // Parse CSV
  // This step will be slightly slower than before due to 8x data copying, but still fast.
  const data = parseKaggleCSV(csvPath);
  
  const parsingTimeMinutes = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
  console.log(`\n⏱ Data parsing (including augmentation) completed in ${parsingTimeMinutes} minutes.`);
  
  if (data.length === 0) {
    console.error('❌ ERROR: No valid training data found in CSV!');
    process.exit(1);
  }
  
  console.log(`\n📊 Dataset Statistics:`);
  console.log(`   Total augmented samples: ${data.length.toLocaleString()}`);
  
  // --- Training Parameters ---
  let trainingData = data;
  const MAX_SAMPLES = 150000; // Optimal sample size for your laptop with this augmented data
  const EPOCHS = 100; // Increased epochs for better convergence
  
  // Adjusted Training Time Estimate (150k samples, 100 epochs, smaller model)
  const estimatedTrainingTimeMinutes = (MAX_SAMPLES / 50000) * (EPOCHS / 50) * 15; 
  console.log(`   Estimated training time (${MAX_SAMPLES.toLocaleString()} samples, ${EPOCHS} epochs): ${Math.round(estimatedTrainingTimeMinutes)} - ${Math.round(estimatedTrainingTimeMinutes * 1.5)} minutes.\n`);

  
  // Optionally sample data if too large (for faster training on old laptop)
  if (data.length > MAX_SAMPLES) {
    console.log(`⚡ Sampling ${MAX_SAMPLES.toLocaleString()} random samples for faster training...`);
    
    // Random sampling
    trainingData = [];
    const indices = new Set();
    while (indices.size < MAX_SAMPLES) {
      indices.add(Math.floor(Math.random() * data.length));
    }
    indices.forEach(idx => trainingData.push(data[idx]));
    
    console.log(`   New training size: ${trainingData.length.toLocaleString()} samples.\n`);
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
  console.log('\n🤖 Creating smaller, optimized neural network model...');
  const model = createModel();
  model.summary();
  
  // Train model
  console.log(`\n🎯 Training model (${EPOCHS} epochs) - expect much higher accuracy this time!`);
  const trainingStart = Date.now();
  
  await model.fit(xs, ys, {
    epochs: EPOCHS,
    batchSize: 128,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 5 === 0) {
          console.log(
            `Epoch ${epoch + 1}/${EPOCHS}: ` +
            `loss=${logs.loss.toFixed(4)}, ` +
            `acc=${logs.acc.toFixed(4)}, ` +
            `val_loss=${logs.val_loss.toFixed(4)}, ` +
            `val_acc=${logs.val_acc.toFixed(4)}`
          );
        }
      }
    }
  });
  
  const trainingTime = ((Date.now() - trainingStart) / 1000 / 60).toFixed(2);
  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
  
  console.log(`\n⏱ Model training completed in ${trainingTime} minutes`);
  console.log(`⏱ Total script runtime (including parsing): ${totalTime} minutes`);
  
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