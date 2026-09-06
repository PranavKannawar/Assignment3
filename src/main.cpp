#include "connect4_common.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

extern void printBoardCLI(const Board& b);
extern void logBoardJSON(std::ofstream& logFile, int turnNumber, int player, int col, int row, const Board& b, double timeMs);

// Global shared state for lock-file simulation and thread safety
Board g_gameBoard;
std::mutex g_boardMutex;
std::atomic<bool> g_gameOver(false);
std::atomic<int> g_winner(0);

const std::string LOCK_DIR = "./output/";
const std::string LOCK_FILE_NAME = "./output/turn.lock";

// Thread worker for Competitor 1 (MCTS)
void player1_worker(int deviceId, int numSimulations) {
    int myPlayer = PLAYER_1;
    while (!g_gameOver.load()) {
        // Wait until lock file indicates turn == PLAYER_1
        LockSync::waitForTurn(LOCK_FILE_NAME, myPlayer, 5);
        if (g_gameOver.load()) break;

        // Acquire lock for board mutation
        {
            std::lock_guard<std::mutex> lock(g_boardMutex);
            if (g_gameOver.load()) break;

            auto t0 = std::chrono::high_resolution_clock::now();
            int move = computeBestMove_MCTS(g_gameBoard, myPlayer, deviceId, numSimulations);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int row = g_gameBoard.dropPiece(move, myPlayer);
            std::cout << "[TURN] Player 1 (MCTS - GPU " << deviceId << ") placed in Col " << move 
                      << " (Row " << row << ") in " << ms << " ms\n";
            printBoardCLI(g_gameBoard);

            if (g_gameBoard.checkWin(myPlayer)) {
                g_winner.store(myPlayer);
                g_gameOver.store(true);
                LockSync::setTurn(LOCK_FILE_NAME, -1);
                break;
            }
            if (g_gameBoard.isFull()) {
                g_winner.store(0); // Draw
                g_gameOver.store(true);
                LockSync::setTurn(LOCK_FILE_NAME, -1);
                break;
            }
        }

        // Pass turn to Player 2
        LockSync::setTurn(LOCK_FILE_NAME, PLAYER_2);
    }
}

// Thread worker for Competitor 2 (Minimax)
void player2_worker(int deviceId, int maxDepth) {
    int myPlayer = PLAYER_2;
    while (!g_gameOver.load()) {
        // Wait until lock file indicates turn == PLAYER_2
        LockSync::waitForTurn(LOCK_FILE_NAME, myPlayer, 5);
        if (g_gameOver.load()) break;

        // Acquire lock for board mutation
        {
            std::lock_guard<std::mutex> lock(g_boardMutex);
            if (g_gameOver.load()) break;

            auto t0 = std::chrono::high_resolution_clock::now();
            int move = computeBestMove_Minimax(g_gameBoard, myPlayer, deviceId, maxDepth);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int row = g_gameBoard.dropPiece(move, myPlayer);
            std::cout << "[TURN] Player 2 (Minimax - GPU " << deviceId << ") placed in Col " << move 
                      << " (Row " << row << ") in " << ms << " ms\n";
            printBoardCLI(g_gameBoard);

            if (g_gameBoard.checkWin(myPlayer)) {
                g_winner.store(myPlayer);
                g_gameOver.store(true);
                LockSync::setTurn(LOCK_FILE_NAME, -1);
                break;
            }
            if (g_gameBoard.isFull()) {
                g_winner.store(0); // Draw
                g_gameOver.store(true);
                LockSync::setTurn(LOCK_FILE_NAME, -1);
                break;
            }
        }

        // Pass turn to Player 1
        LockSync::setTurn(LOCK_FILE_NAME, PLAYER_1);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "===============================================================\n";
    std::cout << "   MULTI-GPU COMPETITIVE GAME: CONNECT-4 (ENTERPRISE CUDA)    \n";
    std::cout << "   Competitor 1: GPU Monte Carlo Tree Search (Rollouts)        \n";
    std::cout << "   Competitor 2: GPU Depth-Limited Minimax (Alpha-Beta)        \n";
    std::cout << "===============================================================\n\n";

    // Detect CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Warning: No physical CUDA GPU detected or driver error (" 
                  << cudaGetErrorString(err) << "). Fallback / verification mode.\n";
        deviceCount = 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA capable device(s).\n";
    int p1_device = 0;
    int p2_device = (deviceCount >= 2) ? 1 : 0;

    if (deviceCount >= 2) {
        std::cout << "[CONFIG] True Multi-GPU Mode: Competitor 1 on Device 0, Competitor 2 on Device 1\n";
    } else {
        std::cout << "[CONFIG] Coursera Single-GPU Mode: Dual Host Competitor Threads sharing Device 0 via Lock-Files\n";
    }

    // CLI parameters
    int simulations = 100000;
    int minimaxDepth = 5;
    if (argc >= 2) simulations = std::stoi(argv[1]);
    if (argc >= 3) minimaxDepth = std::stoi(argv[2]);

    std::cout << "Parameters: MCTS Simulations/turn = " << simulations 
              << ", Minimax Depth = " << minimaxDepth << "\n\n";

    // Initialize game board
    g_gameBoard.init();

    // Prepare lock file for Player 1 to start
    LockSync::setTurn(LOCK_FILE_NAME, PLAYER_1);

    std::cout << "Starting Initial Board State:\n";
    printBoardCLI(g_gameBoard);

    // Launch competitor threads
    std::thread t1(player1_worker, p1_device, simulations);
    std::thread t2(player2_worker, p2_device, minimaxDepth);

    t1.join();
    t2.join();

    // Clean up lock file
    LockSync::releaseLock(LOCK_FILE_NAME);

    // Output final results
    std::cout << "\n===============================================================\n";
    std::cout << "                     MATCH FINISHED!                          \n";
    std::cout << "===============================================================\n";
    int finalWinner = g_winner.load();
    if (finalWinner == PLAYER_1) {
        std::cout << " WINNER: Player 1 (GPU Monte Carlo Tree Search)!\n";
    } else if (finalWinner == PLAYER_2) {
        std::cout << " WINNER: Player 2 (GPU Depth-Limited Minimax)!\n";
    } else {
        std::cout << " RESULT: Draw / Stalemate!\n";
    }
    std::cout << "===============================================================\n";

    return 0;
}
