// train.js - Generate data and train the model
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// ========== DATA GENERATION ==========

class TicTacToe {
  constructor() {
    this.board = Array(9).fill(0); // 0=empty, 1=X, -1=O
  }

  copy() {
    const game = new TicTacToe();
    game.board = [...this.board];
    return game;
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

  getBestMove(player) {
    const moves = this.getAvailableMoves();
    let bestScore = player === 1 ? -Infinity : Infinity;
    let bestMove = moves[0];

    for (const move of moves) {
      const newGame = this.copy();
      newGame.makeMove(move, player);
      const score = newGame.minimax(-player);

      if ((player === 1 && score > bestScore) || 
          (player === -1 && score < bestScore)) {
        bestScore = score;
        bestMove = move;
      }
    }

    return bestMove;
  }
}

// Generate training data
function generateTrainingData(numGames = 5000) {
  console.log(`Generating ${numGames} games...`);
  const data = [];
  
  for (let i = 0; i < numGames; i++) {
    const game = new TicTacToe();
    let currentPlayer = 1;
    
    while (game.checkWinner() === null) {
      const moves = game.getAvailableMoves();
      
      // Mix of optimal and random moves for variety
      let move;
      if (Math.random() < 0.8) { // 80% optimal moves
        move = game.getBestMove(currentPlayer);
      } else { // 20% random moves
        move = moves[Math.floor(Math.random() * moves.length)];
      }
      
      // Store board state and move for player 1 (X)
      if (currentPlayer === 1) {
        data.push({
          board: [...game.board],
          move: move
        });
      }
      
      game.makeMove(move, currentPlayer);
      currentPlayer = -currentPlayer;
    }
    
    if ((i + 1) % 1000 === 0) {
      console.log(`Generated ${i + 1} games...`);
    }
  }
  
  console.log(`Total training samples: ${data.length}`);
  return data;
}

// ========== MODEL CREATION & TRAINING ==========

function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 128, activation: 'relu', inputShape: [9] }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 64, activation: 'relu' }),
      tf.layers.dense({ units: 9, activation: 'softmax' }) // 9 possible moves
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
  console.log('\n=== Starting Tic-Tac-Toe AI Training ===\n');
  
  // Generate data
  const trainingData = generateTrainingData(5000);
  
  // Convert to tensors
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
  model.summary();
  
  console.log('\nTraining model...');
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
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
  
  // Save model
  await model.save('file://./tictactoe-model');
  console.log('\n✓ Model saved to ./tictactoe-model/');
  
  // Clean up
  xs.dispose();
  ys.dispose();
  
  console.log('\n=== Training Complete ===\n');
}

// Run training
trainModel().catch(console.error);