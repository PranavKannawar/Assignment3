#ifndef CONNECT4_COMMON_H_
#define CONNECT4_COMMON_H_

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int BOARD_SIZE = ROWS * COLS; // 42

constexpr int EMPTY = 0;
constexpr int PLAYER_1 = 1; // e.g., MCTS Strategy
constexpr int PLAYER_2 = 2; // e.g., Minimax Strategy

// Board state represented as a flat row-major array: cell at (r, c) = board[r * COLS + c]
// Row 0 is the bottom row, Row 5 is the top row.
struct Board {
    int cells[BOARD_SIZE];

    __host__ __device__ void init() {
        for (int i = 0; i < BOARD_SIZE; ++i) {
            cells[i] = EMPTY;
        }
    }

    __host__ __device__ int get(int r, int c) const {
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return -1;
        return cells[r * COLS + c];
    }

    __host__ __device__ void set(int r, int c, int val) {
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
            cells[r * COLS + c] = val;
        }
    }

    // Check if a move in column c is valid (i.e. top row is empty)
    __host__ __device__ bool isValidMove(int c) const {
        if (c < 0 || c >= COLS) return false;
        return cells[(ROWS - 1) * COLS + c] == EMPTY;
    }

    // Drop piece in column c for player. Returns row placed, or -1 if column is full.
    __host__ __device__ int dropPiece(int c, int player) {
        if (c < 0 || c >= COLS) return -1;
        for (int r = 0; r < ROWS; ++r) {
            if (cells[r * COLS + c] == EMPTY) {
                cells[r * COLS + c] = player;
                return r;
            }
        }
        return -1;
    }

    // Check if the board is completely full (Draw)
    __host__ __device__ bool isFull() const {
        for (int c = 0; c < COLS; ++c) {
            if (cells[(ROWS - 1) * COLS + c] == EMPTY) return false;
        }
        return true;
    }

    // Check if player has 4 consecutive pieces in any direction
    __host__ __device__ bool checkWin(int player) const {
        // Horizontal check
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c <= COLS - 4; ++c) {
                if (get(r, c) == player &&
                    get(r, c + 1) == player &&
                    get(r, c + 2) == player &&
                    get(r, c + 3) == player) {
                    return true;
                }
            }
        }

        // Vertical check
        for (int c = 0; c < COLS; ++c) {
            for (int r = 0; r <= ROWS - 4; ++r) {
                if (get(r, c) == player &&
                    get(r + 1, c) == player &&
                    get(r + 2, c) == player &&
                    get(r + 3, c) == player) {
                    return true;
                }
            }
        }

        // Positive diagonal check (/)
        for (int r = 0; r <= ROWS - 4; ++r) {
            for (int c = 0; c <= COLS - 4; ++c) {
                if (get(r, c) == player &&
                    get(r + 1, c + 1) == player &&
                    get(r + 2, c + 2) == player &&
                    get(r + 3, c + 3) == player) {
                    return true;
                }
            }
        }

        // Negative diagonal check (\)
        for (int r = 3; r < ROWS; ++r) {
            for (int c = 0; c <= COLS - 4; ++c) {
                if (get(r, c) == player &&
                    get(r - 1, c + 1) == player &&
                    get(r - 2, c + 2) == player &&
                    get(r - 3, c + 3) == player) {
                    return true;
                }
            }
        }

        return false;
    }
};

// Lock file synchronization helper functions for turn-based multi-GPU or multi-thread execution
namespace LockSync {
    inline void acquireLock(const std::string& lockFileName, int pollingMs = 10) {
        while (true) {
            std::ifstream test(lockFileName);
            if (!test.good()) {
                // File doesn't exist, attempt creation
                std::ofstream lock(lockFileName);
                if (lock.is_open()) {
                    lock << "locked" << std::endl;
                    lock.close();
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(pollingMs));
        }
    }

    inline void releaseLock(const std::string& lockFileName) {
        std::remove(lockFileName.c_str());
    }

    inline void waitForTurn(const std::string& turnFileName, int myPlayerId, int pollingMs = 10) {
        while (true) {
            std::ifstream file(turnFileName);
            if (file.is_open()) {
                int currentTurn = -1;
                file >> currentTurn;
                file.close();
                if (currentTurn == myPlayerId) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(pollingMs));
        }
    }

    inline void setTurn(const std::string& turnFileName, int nextPlayerId) {
        std::ofstream file(turnFileName, std::ios::trunc);
        if (file.is_open()) {
            file << nextPlayerId << std::endl;
            file.close();
        }
    }
}

// Function declarations for the two GPU competitor strategies
int computeBestMove_MCTS(const Board& h_board, int player, int deviceId, int numSimulations = 200000);
int computeBestMove_Minimax(const Board& h_board, int player, int deviceId, int maxDepth = 6);

#endif // CONNECT4_COMMON_H_
