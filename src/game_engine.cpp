#include "connect4_common.h"
#include <iomanip>
#include <sstream>

void printBoardCLI(const Board& b) {
    std::cout << "\n+---+---+---+---+---+---+---+\n";
    for (int r = ROWS - 1; r >= 0; --r) {
        std::cout << "|";
        for (int c = 0; c < COLS; ++c) {
            int cell = b.get(r, c);
            if (cell == PLAYER_1) {
                std::cout << " X |"; // Player 1 (Red / MCTS)
            } else if (cell == PLAYER_2) {
                std::cout << " O |"; // Player 2 (Yellow / Minimax)
            } else {
                std::cout << " . |";
            }
        }
        std::cout << "\n+---+---+---+---+---+---+---+\n";
    }
    std::cout << "  0   1   2   3   4   5   6  (Columns)\n\n";
}

void logBoardJSON(std::ofstream& logFile, int turnNumber, int player, int col, int row, const Board& b, double timeMs) {
    logFile << "    {\n";
    logFile << "      \"turn\": " << turnNumber << ",\n";
    logFile << "      \"player\": " << player << ",\n";
    logFile << "      \"strategy\": \"" << (player == PLAYER_1 ? "GPU-1: MCTS Rollouts" : "GPU-2: Minimax Alpha-Beta") << "\",\n";
    logFile << "      \"column\": " << col << ",\n";
    logFile << "      \"row\": " << row << ",\n";
    logFile << "      \"decision_time_ms\": " << std::fixed << std::setprecision(2) << timeMs << ",\n";
    logFile << "      \"board\": [";
    for (int i = 0; i < BOARD_SIZE; ++i) {
        logFile << b.cells[i] << (i < BOARD_SIZE - 1 ? "," : "");
    }
    logFile << "]\n    }";
}
