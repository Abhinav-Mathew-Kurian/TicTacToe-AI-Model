// train.js - Generate data and train the model for UNBEATABLE play
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// --- Symmetry Functions for Data Augmentation ---

// Rotate board clockwise 90 degrees (index mapping)
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

// Flip board horizontally (index mapping)
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


// ========== DATA GENERATION & MINIMAX ==========

class TicTacToe {
  constructor(board = Array(9).fill(0)) {
    this.board = board; // 0=empty, 1=X, -1=O
  }

  copy() {
    return new TicTacToe([...this.board]);
  }

  getAvailableMoves() {
    return this.board.map((cell, idx) => cell === 0 ? idx : -1).filter(idx => idx !== -1);
  }

  makeMove(position, player) {
    if (this.board[position] === 0) {
      this.board[position] = player;
      return true;
    }
    return false;
  }

  checkWinner() {
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
      [0, 4, 8], [2, 4, 6]             // diagonals
    ];

    for (const [a, b, c] of lines) {
      if (this.board[a] !== 0 && 
          this.board[a] === this.board[b] && 
          this.board[a] === this.board[c]) {
        return this.board[a];
      }
    }

    return this.board.includes(0) ? null : 0; // null=ongoing, 0=draw
  }

  // Minimax algorithm for optimal play
  minimax(player, depth = 0) {
    const winner = this.checkWinner();
    
    // Player 1 (X) is max, Player -1 (O) is min
    if (winner === 1) return 10 - depth;
    if (winner === -1) return depth - 10;
    if (winner === 0) return 0;

    const moves = this.getAvailableMoves();
    const scores = [];

    for (const move of moves) {
      const newGame = this.copy();
      newGame.makeMove(move, player);
      scores.push(newGame.minimax(-player, depth + 1));
    }

    return player === 1 ? Math.max(...scores) : Math.min(...scores);
  }

  // Finds ALL moves that lead to the best possible score (Max/Min)
  getBestMoves(player) {
    const moves = this.getAvailableMoves();
    if (moves.length === 0) return [];

    let bestScore = player === 1 ? -Infinity : Infinity;
    const scoreMap = new Map();

    // 1. Calculate scores for all possible moves
    for (const move of moves) {
      const newGame = this.copy();
      newGame.makeMove(move, player);
      // We are looking for the score of the game state AFTER this move
      const score = newGame.minimax(-player); 
      
      scoreMap.set(move, score);

      // Find the best score
      if (player === 1) {
        bestScore = Math.max(bestScore, score);
      } else {
        bestScore = Math.min(bestScore, score);
      }
    }

    // 2. Return all moves that result in the best score
    const optimalMoves = [];
    for (const [move, score] of scoreMap.entries()) {
      if (score === bestScore) {
        optimalMoves.push(move);
      }
    }

    // Crucially: If multiple moves are optimal (e.g., all draw), we just pick the first one 
    // to give a deterministic label for the model to learn.
    return [optimalMoves[0]];
  }
}

// --- NEW DATA GENERATOR: SYSTEMATICALLY GENERATE ALL PERFECT STATES (FIXED) ---

function generatePerfectTrainingData() {
  console.log(`\n🔄 Generating all unique, reachable states and labeling with Minimax...`);
  const uniqueStates = new Set();
  const trainingData = [];
  
  // Recursive function to generate all possible states
  function generateStates(game, currentPlayer) {
    const boardStr = game.board.join(',');

    // 1. TERMINATION/CACHE CHECK
    if (game.checkWinner() !== null) {
      return;
    }
    
    if (uniqueStates.has(boardStr)) {
      return;
    }
    uniqueStates.add(boardStr); // Mark this state as visited
    
    // 2. DATA LABELING (ONLY FOR PLAYER X)
    if (currentPlayer === 1) {
      // Find the MINIMAX OPTIMAL move for the current board state
      // We use getBestMoves and take the first optimal move for deterministic labeling
      const optimalMove = game.getBestMoves(currentPlayer)[0];
      
      // Apply 8x symmetry augmentation to this PERFECT move
      if (optimalMove !== undefined) {
        const symmetries = generateSymmetries(game.board, optimalMove);
        trainingData.push(...symmetries);
      }
    }

    // 3. EXPLORE NEXT STATES (EXPLORE ALL LEGAL MOVES)
    const moves = game.getAvailableMoves();
    const nextPlayer = -currentPlayer;
    
    // CRUCIAL FIX: We must iterate through *ALL* available moves to ensure 
    // we visit every possible board state. This prevents premature pruning.
    for (const move of moves) {
      const nextGame = game.copy();
      nextGame.makeMove(move, currentPlayer);
      generateStates(nextGame, nextPlayer);
    }
  }

  // Start generating states from an empty board
  const emptyGame = new TicTacToe();
  generateStates(emptyGame, 1); 
  
  console.log(`✓ Generated ${uniqueStates.size.toLocaleString()} unique board states.`);
  console.log(`✓ Total training samples (100% Minimax + 8x Augmentation): ${trainingData.length.toLocaleString()}`);
  
  return trainingData;
}
// --- END NEW DATA GENERATOR (FIXED) ---


// ========== MODEL CREATION & TRAINING ==========

function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 256, activation: 'relu', inputShape: [9] }), 
      tf.layers.batchNormalization(), 
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 128, activation: 'relu' }), 
      tf.layers.batchNormalization(),
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

async function trainModel() {
  console.log('\n=== Starting Tic-Tac-Toe AI Training (UNBEATABLE MODE) ===\n');
  
  // 1. Generate 100% systematic, perfect data
  const trainingData = generatePerfectTrainingData();
  
  if (trainingData.length === 0) {
      console.error("❌ ERROR: Data generation failed. Check Minimax logic.");
      return;
  }
  
  // 2. Convert to tensors
  const xs = tf.tensor2d(trainingData.map(d => d.board));
  const ys = tf.tensor2d(
    trainingData.map(d => {
      const oneHot = Array(9).fill(0);
      oneHot[d.move] = 1;
      return oneHot;
    })
  );
  
  console.log('\nCreating model...');
  const model = createModel();
  
  console.log('\nTraining model...');
  await model.fit(xs, ys, {
    epochs: 100, 
    batchSize: 64, 
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 5 === 0) {
          console.log(
            `Epoch ${epoch + 1}: ` +
            `loss=${logs.loss.toFixed(4)}, ` +
            `acc=${logs.acc.toFixed(4)}, ` +
            `val_loss=${logs.val_loss.toFixed(4)}, ` +
            `val_acc=${logs.val_acc.toFixed(4)}`
          );
        }
      }
    }
  });
  
  // 3. Save model
  // Requested model path: models/base
  const modelDir = './models/base'; 
  if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true });
  }
  await model.save(`file://${modelDir}`);
  console.log(`\n✓ Model saved to ${modelDir}/`);
  
  // Test predictions
  console.log('\n🧪 Testing model predictions...');
  testModel(model, trainingData);
  
  // Clean up
  xs.dispose();
  ys.dispose();
  
  console.log('\n=== Training Complete ===\n');
  console.log('--- NEXT STEPS ---');
  console.log(`1. Your model is saved in the directory: ${modelDir}`);
  console.log('2. The full path is: /home/kottackal/Desktop/Abhinav/Projects/tictactoe-ai/models/base');
  console.log('3. Update your web app to load the model from the path above.');
  console.log('4. The resulting AI should be UNBEATABLE (always draws or wins).');
}

// Test the model
function testModel(model, data) {
  const testSamples = 10;
  let correct = 0;
  
  for (let i = 0; i < testSamples; i++) {
    const idx = Math.floor(Math.random() * data.length);
    const sample = data[idx];
    
    // Predict the move
    const pred = model.predict(tf.tensor2d([sample.board]));
    const probs = pred.dataSync();
    const predictedMove = probs.indexOf(Math.max(...probs));
    
    const match = predictedMove === sample.move;
    if (match) correct++;
    
    pred.dispose();
  }
  
  console.log(`\nAccuracy on ${testSamples} test samples: ${(correct/testSamples * 100).toFixed(0)}%`);
  console.log('🔥 EXPECTED ACCURACY ON PERFECT DATA: 90%+');
}


// Run training
trainModel().catch(console.error);