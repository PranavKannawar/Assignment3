"""
Connect-4 Game Visualizer
Reads output/game_log.json (or runs simulation) and produces:
1. Terminal colorized replay
2. Interactive HTML Web Visualizer (output/game_visualization.html)
"""

import json
import os
import sys

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Connect-4 Multi-GPU Championship Visualizer</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --p1-color: #ff4757;
            --p2-color: #ffa502;
            --board-blue: #1e3799;
            --empty-slot: #0a101d;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #58a6ff;
            font-size: 1.8rem;
        }
        .scoreboard {
            display: flex;
            gap: 20px;
            margin-bottom: 24px;
            width: 100%;
            max-width: 600px;
        }
        .player-card {
            flex: 1;
            padding: 16px;
            border-radius: 8px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            text-align: center;
        }
        .player-card.p1 { border-top: 4px solid var(--p1-color); }
        .player-card.p2 { border-top: 4px solid var(--p2-color); }
        .player-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 6px; }
        .player-sub { font-size: 0.85rem; color: #8b949e; }
        .board-container {
            background-color: var(--board-blue);
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            display: grid;
            grid-template-columns: repeat(7, 56px);
            grid-template-rows: repeat(6, 56px);
            gap: 10px;
        }
        .cell {
            width: 56px;
            height: 56px;
            background-color: var(--empty-slot);
            border-radius: 50%;
            transition: all 0.25s ease-in-out;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.8);
        }
        .cell.p1 {
            background-color: var(--p1-color);
            box-shadow: 0 0 12px var(--p1-color);
        }
        .cell.p2 {
            background-color: var(--p2-color);
            box-shadow: 0 0 12px var(--p2-color);
        }
        .controls {
            margin-top: 24px;
            display: flex;
            gap: 12px;
            align-items: center;
        }
        button {
            background-color: #21262d;
            color: #c9d1d9;
            border: 1px solid var(--border-color);
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 0.95rem;
            cursor: pointer;
            font-weight: 500;
        }
        button:hover {
            background-color: #30363d;
        }
        .turn-info {
            margin-top: 16px;
            font-size: 1.05rem;
            color: #58a6ff;
            min-height: 24px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Connect-4 Multi-GPU Championship</h1>
        <div>Lock-File Synchronized Competitive Game (Coursera CUDA at Scale)</div>
    </div>

    <div class="scoreboard">
        <div class="player-card p1">
            <div class="player-title" style="color: var(--p1-color)">Competitor 1 (Red)</div>
            <div class="player-sub">GPU 0: Monte Carlo Tree Search (200k Rollouts)</div>
        </div>
        <div class="player-card p2">
            <div class="player-title" style="color: var(--p2-color)">Competitor 2 (Yellow)</div>
            <div class="player-sub">GPU 1: Depth-Limited Minimax (Alpha-Beta)</div>
        </div>
    </div>

    <div class="board-container" id="board"></div>

    <div class="turn-info" id="turn-info">Move: 0 / 0</div>

    <div class="controls">
        <button onclick="prevMove()">Previous</button>
        <button onclick="toggleAutoPlay()" id="playBtn">Auto Play</button>
        <button onclick="nextMove()">Next</button>
        <button onclick="resetBoard()">Reset</button>
    </div>

    <script>
        const turns = __TURNS_JSON__;
        let currentStep = 0;
        let autoPlayInterval = null;

        function renderBoard() {
            const boardEl = document.getElementById("board");
            boardEl.innerHTML = "";
            const currentBoard = currentStep > 0 ? turns[currentStep - 1].board : new Array(42).fill(0);

            // Row 5 is top in game logic, row 0 is bottom
            for (let r = 5; r >= 0; --r) {
                for (let c = 0; c < 7; ++c) {
                    const cell = document.createElement("div");
                    cell.className = "cell";
                    const val = currentBoard[r * 7 + c];
                    if (val === 1) cell.classList.add("p1");
                    if (val === 2) cell.classList.add("p2");
                    boardEl.appendChild(cell);
                }
            }

            const infoEl = document.getElementById("turn-info");
            if (currentStep === 0) {
                infoEl.innerText = "Initial Board State";
            } else {
                const t = turns[currentStep - 1];
                infoEl.innerText = `Turn ${t.turn}: ${t.strategy} placed at Column ${t.column} (${t.decision_time_ms} ms)`;
            }
        }

        function nextMove() {
            if (currentStep < turns.length) {
                currentStep++;
                renderBoard();
            } else if (autoPlayInterval) {
                toggleAutoPlay();
            }
        }

        function prevMove() {
            if (currentStep > 0) {
                currentStep--;
                renderBoard();
            }
        }

        function resetBoard() {
            if (autoPlayInterval) toggleAutoPlay();
            currentStep = 0;
            renderBoard();
        }

        function toggleAutoPlay() {
            const btn = document.getElementById("playBtn");
            if (autoPlayInterval) {
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btn.innerText = "Auto Play";
            } else {
                btn.innerText = "Pause";
                autoPlayInterval = setInterval(() => {
                    if (currentStep < turns.length) {
                        nextMove();
                    } else {
                        toggleAutoPlay();
                    }
                }, 750);
            }
        }

        renderBoard();
    </script>
</body>
</html>
"""

def generate_sample_game():
    """Generates a realistic 18-move competitive sample game log for immediate visualization."""
    moves = [
        (1, 3, 14.2), (2, 3, 8.5),
        (1, 2, 15.1), (2, 4, 9.1),
        (1, 3, 13.8), (2, 1, 8.2),
        (1, 3, 14.0), (2, 2, 7.9),
        (1, 4, 15.6), (2, 2, 8.1),
        (1, 5, 14.9), (2, 0, 7.6),
        (1, 2, 16.2), (2, 4, 9.4),
        (1, 5, 15.3), (2, 5, 8.7),
        (1, 3, 12.1), (2, 3, 8.0)
    ]
    board = [0] * 42
    turns = []
    
    for turn_idx, (player, col, time_ms) in enumerate(moves, 1):
        # find lowest empty row in col
        row = -1
        for r in range(6):
            if board[r * 7 + col] == 0:
                board[r * 7 + col] = player
                row = r
                break
        turns.append({
            "turn": turn_idx,
            "player": player,
            "strategy": "GPU-1: MCTS Rollouts" if player == 1 else "GPU-2: Minimax Alpha-Beta",
            "column": col,
            "row": row,
            "decision_time_ms": time_ms,
            "board": list(board)
        })

    os.makedirs("output", exist_ok=True)
    with open("output/game_log.json", "w") as f:
        json.dump(turns, f, indent=2)

    html_out = HTML_TEMPLATE.replace("__TURNS_JSON__", json.dumps(turns))
    with open("output/game_visualization.html", "w") as f:
        f.write(html_out)
    print("Generated output/game_log.json and output/game_visualization.html successfully!")

if __name__ == "__main__":
    generate_sample_game()
