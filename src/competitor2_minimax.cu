#include "connect4_common.h"
#include <iostream>
#include <vector>
#include <algorithm>

// Static evaluation heuristic for Connect-4 board
__device__ inline int evaluateWindow(int w0, int w1, int w2, int w3, int player, int opponent) {
    int countP = (w0 == player) + (w1 == player) + (w2 == player) + (w3 == player);
    int countO = (w0 == opponent) + (w1 == opponent) + (w2 == opponent) + (w3 == opponent);
    int countE = (w0 == EMPTY) + (w1 == EMPTY) + (w2 == EMPTY) + (w3 == EMPTY);

    if (countP == 4) return 100000;
    if (countP == 3 && countE == 1) return 100;
    if (countP == 2 && countE == 2) return 10;

    if (countO == 3 && countE == 1) return -120; // Prioritize blocking 3 opponent pieces
    if (countO == 2 && countE == 2) return -10;

    return 0;
}

__device__ int evaluateBoard(const Board& b, int player, int opponent) {
    int score = 0;

    // Center column preference (tactical advantage in Connect-4)
    for (int r = 0; r < ROWS; ++r) {
        if (b.get(r, 3) == player) score += 6;
        else if (b.get(r, 3) == opponent) score -= 6;
    }

    // Horizontal windows
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c <= COLS - 4; ++c) {
            score += evaluateWindow(b.get(r, c), b.get(r, c + 1), b.get(r, c + 2), b.get(r, c + 3), player, opponent);
        }
    }

    // Vertical windows
    for (int c = 0; c < COLS; ++c) {
        for (int r = 0; r <= ROWS - 4; ++r) {
            score += evaluateWindow(b.get(r, c), b.get(r + 1, c), b.get(r + 2, c), b.get(r + 3, c), player, opponent);
        }
    }

    // Positive diagonal windows
    for (int r = 0; r <= ROWS - 4; ++r) {
        for (int c = 0; c <= COLS - 4; ++c) {
            score += evaluateWindow(b.get(r, c), b.get(r + 1, c + 1), b.get(r + 2, c + 2), b.get(r + 3, c + 3), player, opponent);
        }
    }

    // Negative diagonal windows
    for (int r = 3; r < ROWS; ++r) {
        for (int c = 0; c <= COLS - 4; ++c) {
            score += evaluateWindow(b.get(r, c), b.get(r - 1, c + 1), b.get(r - 2, c + 2), b.get(r - 3, c + 3), player, opponent);
        }
    }

    return score;
}

// Recursive/iterative device minimax search
__device__ int minimaxDevice(Board b, int depth, bool isMaximizing, int alpha, int beta, int player, int opponent) {
    if (b.checkWin(player)) return 1000000 + depth;
    if (b.checkWin(opponent)) return -1000000 - depth;
    if (b.isFull() || depth == 0) return evaluateBoard(b, player, opponent);

    const int moveOrder[COLS] = {3, 2, 4, 1, 5, 0, 6};

    if (isMaximizing) {
        int maxEval = -2000000;
        for (int i = 0; i < COLS; ++i) {
            int c = moveOrder[i];
            if (b.isValidMove(c)) {
                Board child = b;
                child.dropPiece(c, player);
                int eval = minimaxDevice(child, depth - 1, false, alpha, beta, player, opponent);
                if (eval > maxEval) maxEval = eval;
                if (eval > alpha) alpha = eval;
                if (beta <= alpha) break; // Alpha-beta pruning
            }
        }
        return maxEval;
    } else {
        int minEval = 2000000;
        for (int i = 0; i < COLS; ++i) {
            int c = moveOrder[i];
            if (b.isValidMove(c)) {
                Board child = b;
                child.dropPiece(c, opponent);
                int eval = minimaxDevice(child, depth - 1, true, alpha, beta, player, opponent);
                if (eval < minEval) minEval = eval;
                if (eval < beta) beta = eval;
                if (beta <= alpha) break; // Alpha-beta pruning
            }
        }
        return minEval;
    }
}

// CUDA Kernel: Evaluates the search tree for root branches across GPU warps/threads
__global__ void minimax_kernel(Board rootBoard, int depth, int player, int opponent, int* d_colScores) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= COLS) return;

    if (!rootBoard.isValidMove(c)) {
        d_colScores[c] = -9999999;
        return;
    }

    Board child = rootBoard;
    child.dropPiece(c, player);

    if (child.checkWin(player)) {
        d_colScores[c] = 2000000;
        return;
    }

    // Minimax search from opponent's response
    int score = minimaxDevice(child, depth - 1, false, -2000000, 2000000, player, opponent);
    d_colScores[c] = score;
}

int computeBestMove_Minimax(const Board& h_board, int player, int deviceId, int maxDepth) {
    cudaSetDevice(deviceId);

    int opponent = (player == PLAYER_1) ? PLAYER_2 : PLAYER_1;

    // Check immediate moves on CPU first
    for (int c = 0; c < COLS; ++c) {
        if (h_board.isValidMove(c)) {
            Board testBoard = h_board;
            testBoard.dropPiece(c, player);
            if (testBoard.checkWin(player)) return c;
        }
    }
    for (int c = 0; c < COLS; ++c) {
        if (h_board.isValidMove(c)) {
            Board testBoard = h_board;
            testBoard.dropPiece(c, opponent);
            if (testBoard.checkWin(opponent)) return c;
        }
    }

    int* d_scores = nullptr;
    cudaMalloc(&d_scores, COLS * sizeof(int));

    minimax_kernel<<<1, COLS>>>(h_board, maxDepth, player, opponent, d_scores);
    cudaDeviceSynchronize();

    int h_scores[COLS];
    cudaMemcpy(h_scores, d_scores, COLS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_scores);

    int bestCol = -1;
    int bestScore = -99999999;
    const int colOrder[COLS] = {3, 2, 4, 1, 5, 0, 6};

    for (int i = 0; i < COLS; ++i) {
        int c = colOrder[i];
        if (h_board.isValidMove(c)) {
            if (h_scores[c] > bestScore) {
                bestScore = h_scores[c];
                bestCol = c;
            }
        }
    }

    return (bestCol != -1) ? bestCol : 3;
}
