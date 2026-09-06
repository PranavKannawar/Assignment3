#include "connect4_common.h"
#include <curand_kernel.h>
#include <iostream>
#include <vector>

// Device random state generator using xorshift32 for high throughput within thread
__device__ inline uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// CUDA Kernel: Runs massive parallel random rollouts for each candidate column move
// Each thread evaluates a candidate column (or a partition of candidate columns)
// and simulates random play until win/loss/draw.
__global__ void mcts_rollout_kernel(Board initialBoard, int myPlayer, int opponent, 
                                    int simulationsPerCol, int* d_winCounts, uint32_t baseSeed) {
    int candidateCol = blockIdx.x; // Block per candidate column (0 to COLS-1)
    int tid = blockIdx.y * blockDim.x + threadIdx.x; // Thread index within simulations

    if (candidateCol >= COLS || tid >= simulationsPerCol) return;

    // Check if initial candidate column is valid
    if (!initialBoard.isValidMove(candidateCol)) {
        return;
    }

    // Initialize fast per-thread RNG
    uint32_t rngState = baseSeed ^ (candidateCol * 199999 + tid * 7919 + 1);
    if (rngState == 0) rngState = 123456789;

    // Thread-local board copy in registers/local memory
    Board board = initialBoard;
    board.dropPiece(candidateCol, myPlayer);

    // If immediate winning move, count as win
    if (board.checkWin(myPlayer)) {
        atomicAdd(&d_winCounts[candidateCol], 2); // 2 points for win
        return;
    }

    // Rollout loop
    int currentPlayer = opponent;
    bool isTerminal = false;
    int resultScore = 0; // 2 for win, 1 for draw, 0 for loss

    for (int step = 0; step < BOARD_SIZE; ++step) {
        if (board.isFull()) {
            resultScore = 1; // Draw
            isTerminal = true;
            break;
        }

        // Find available columns
        int validCols[COLS];
        int numValid = 0;
        for (int c = 0; c < COLS; ++c) {
            if (board.isValidMove(c)) {
                validCols[numValid++] = c;
            }
        }

        if (numValid == 0) {
            resultScore = 1; // Draw
            break;
        }

        // Pick random valid column
        int chosenCol = validCols[xorshift32(rngState) % numValid];
        board.dropPiece(chosenCol, currentPlayer);

        if (board.checkWin(currentPlayer)) {
            if (currentPlayer == myPlayer) {
                resultScore = 2; // Win
            } else {
                resultScore = 0; // Loss
            }
            isTerminal = true;
            break;
        }

        // Switch turn
        currentPlayer = (currentPlayer == myPlayer) ? opponent : myPlayer;
    }

    if (!isTerminal) {
        resultScore = 1; // Count unfinished as draw
    }

    if (resultScore > 0) {
        atomicAdd(&d_winCounts[candidateCol], resultScore);
    }
}

int computeBestMove_MCTS(const Board& h_board, int player, int deviceId, int numSimulations) {
    cudaSetDevice(deviceId);

    int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;

    // Check for immediate win or immediate block on CPU first for efficiency
    for (int c = 0; c < COLS; ++c) {
        if (h_board.isValidMove(c)) {
            Board testBoard = h_board;
            testBoard.dropPiece(c, player);
            if (testBoard.checkWin(player)) return c; // Take winning move immediately
        }
    }
    for (int c = 0; c < COLS; ++c) {
        if (h_board.isValidMove(c)) {
            Board testBoard = h_board;
            testBoard.dropPiece(c, opponent);
            if (testBoard.checkWin(opponent)) return c; // Block opponent win immediately
        }
    }

    // Allocate memory on GPU
    int* d_winCounts = nullptr;
    cudaMalloc(&d_winCounts, COLS * sizeof(int));
    cudaMemset(d_winCounts, 0, COLS * sizeof(int));

    int threadsPerBlock = 256;
    int simulationsPerCol = numSimulations / COLS;
    int blocksY = (simulationsPerCol + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(COLS, blocksY);
    dim3 blockDim(threadsPerBlock);

    uint32_t seed = static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Launch MCTS rollout kernel
    mcts_rollout_kernel<<<gridDim, blockDim>>>(h_board, player, opponent, simulationsPerCol, d_winCounts, seed);
    cudaDeviceSynchronize();

    int h_winCounts[COLS];
    cudaMemcpy(h_winCounts, d_winCounts, COLS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_winCounts);

    // Pick candidate column with highest win score among valid moves
    int bestCol = -1;
    int maxScore = -1;

    // Favor center column on ties (order: 3, 2, 4, 1, 5, 0, 6)
    const int colOrder[COLS] = {3, 2, 4, 1, 5, 0, 6};
    for (int i = 0; i < COLS; ++i) {
        int c = colOrder[i];
        if (h_board.isValidMove(c)) {
            if (h_winCounts[c] > maxScore) {
                maxScore = h_winCounts[c];
                bestCol = c;
            }
        }
    }

    return (bestCol != -1) ? bestCol : 3;
}
