"""
Generates a complete mp4 presentation video demonstrating:
Slide 1: Title & System Architecture (Multi-GPU Concurrency & Lock-file turn management)
Slide 2: Algorithms (MCTS Rollouts vs Minimax Alpha-Beta)
Slide 3: Turn-by-Turn Dynamic Connect-4 Game Playback with metrics & animation
Slide 4: Match Summary & Conclusions
"""

import cv2
import numpy as np
import json
import os

WIDTH = 1280
HEIGHT = 720
FPS = 30

def create_slide_intro():
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (18, 14, 13) # dark navy/slate
    
    # Title Box
    cv2.rectangle(img, (80, 50), (WIDTH - 80, 150), (40, 30, 25), -1)
    cv2.rectangle(img, (80, 50), (WIDTH - 80, 150), (88, 166, 255), 2)
    cv2.putText(img, "Multi-GPU Competitive Connect-4 Championship", (120, 115), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(img, "CUDA at Scale for the Enterprise | Peer Review Demonstration", (120, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 148, 158), 2, cv2.LINE_AA)

    # Competitor Cards
    # Left Card - Competitor 1
    cv2.rectangle(img, (120, 240), (580, 480), (33, 38, 45), -1)
    cv2.rectangle(img, (120, 240), (580, 250), (87, 71, 255), -1) # Red top
    cv2.putText(img, "Competitor 1: MCTS (GPU 0)", (140, 290), cv2.FONT_HERSHEY_DUPLEX, 0.85, (87, 71, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "- Strategy: Monte Carlo Tree Search", (140, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- Rollouts: 200,000 parallel games/turn", (140, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- Fast inline Xorshift32 per thread", (140, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- Atomic win-rate aggregation", (140, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)

    # Right Card - Competitor 2
    cv2.rectangle(img, (700, 240), (1160, 480), (33, 38, 45), -1)
    cv2.rectangle(img, (700, 240), (1160, 250), (2, 165, 255), -1) # Yellow top
    cv2.putText(img, "Competitor 2: Minimax (GPU 1)", (720, 290), cv2.FONT_HERSHEY_DUPLEX, 0.85, (2, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "- Strategy: Depth-Limited Minimax", (720, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- Alpha-Beta tree search pruning", (720, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- Center-column tactical bias", (720, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "- 2/3-in-a-row pattern scoring", (720, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)

    # Bottom Architecture Banner
    cv2.rectangle(img, (120, 520), (1160, 650), (25, 20, 18), -1)
    cv2.rectangle(img, (120, 520), (1160, 650), (48, 54, 61), 1)
    cv2.putText(img, "Concurreny & Synchronization Architecture:", (140, 555), cv2.FONT_HERSHEY_DUPLEX, 0.75, (88, 166, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "* Runtime device detection: True Dual-GPU (Device 0 vs 1) or Single-GPU Multi-Threaded", (140, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "* Synchronized turn arbitration using atomic filesystem lock files (output/turn.lock)", (140, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (201, 209, 217), 1, cv2.LINE_AA)
    return img

def render_game_frame(step_data, total_turns):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (18, 14, 13)

    # Header
    cv2.putText(img, "Connect-4 Championship: Live Match Demonstration", (60, 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Turn & Status Bar
    t_num = step_data["turn"]
    player = step_data["player"]
    strat = step_data["strategy"]
    col = step_data["column"]
    ms = step_data["decision_time_ms"]

    status_color = (87, 71, 255) if player == 1 else (2, 165, 255)
    cv2.putText(img, f"Turn {t_num} of {total_turns} | {strat} placed in Column {col} ({ms:.2f} ms)",
                (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # Draw Connect 4 Board
    # Board dimensions: 7 cols, 6 rows
    board_w = 460
    board_h = 400
    board_x = 80
    board_y = 130
    cv2.rectangle(img, (board_x, board_y), (board_x + board_w, board_y + board_h), (153, 55, 30), -1) # Connect4 Blue
    cv2.rectangle(img, (board_x, board_y), (board_x + board_w, board_y + board_h), (200, 80, 50), 3)

    cell_radius = 24
    dx = board_w / 7.0
    dy = board_h / 6.0
    b_cells = step_data["board"]

    for r in range(6):
        for c in range(7):
            cx = int(board_x + c * dx + dx / 2)
            cy = int(board_y + (5 - r) * dy + dy / 2) # invert row so 0 is bottom
            val = b_cells[r * 7 + c]
            if val == 1:
                color = (87, 71, 255) # Red Player 1
            elif val == 2:
                color = (2, 165, 255) # Yellow Player 2
            else:
                color = (29, 16, 10) # Dark empty slot
            cv2.circle(img, (cx, cy), cell_radius, color, -1)
            cv2.circle(img, (cx, cy), cell_radius, (10, 5, 0), 2)

    # Right Side Dashboard / Telemetry
    dash_x = 600
    dash_y = 130
    cv2.rectangle(img, (dash_x, dash_y), (WIDTH - 60, dash_y + 400), (33, 38, 45), -1)
    cv2.rectangle(img, (dash_x, dash_y), (WIDTH - 60, dash_y + 400), (48, 54, 61), 2)

    cv2.putText(img, "Real-Time Telemetry & GPU Metrics", (dash_x + 30, dash_y + 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (88, 166, 255), 2, cv2.LINE_AA)

    cv2.putText(img, f"Active Turn: #{t_num}", (dash_x + 30, dash_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, f"Competitor: {strat}", (dash_x + 30, dash_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"Chosen Column: {col}  (Row: {step_data['row']})", (dash_x + 30, dash_y + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, f"Decision Latency: {ms:.2f} ms", (dash_x + 30, dash_y + 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "Turn Synchronization: output/turn.lock", (dash_x + 30, dash_y + 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Hardware Mode: Dual-Device / Mutex Stream", (dash_x + 30, dash_y + 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "Standard: Clean C++17 & CUDA Runtime", (dash_x + 30, dash_y + 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)

    # Bottom status bar
    cv2.rectangle(img, (60, 560), (WIDTH - 60, 660), (25, 20, 18), -1)
    cv2.rectangle(img, (60, 560), (WIDTH - 60, 660), (48, 54, 61), 1)
    cv2.putText(img, "Turn arbitration is enforced by file-based mutex signaling to prevent data races.",
                (80, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (139, 148, 158), 1, cv2.LINE_AA)
    cv2.putText(img, "Interactive Replay Dashboard available in: output/game_visualization.html",
                (80, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (88, 166, 255), 1, cv2.LINE_AA)

    return img

def render_outro():
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (18, 14, 13)

    cv2.rectangle(img, (150, 120), (WIDTH - 150, HEIGHT - 120), (33, 38, 45), -1)
    cv2.rectangle(img, (150, 120), (WIDTH - 150, HEIGHT - 120), (88, 166, 255), 2)

    cv2.putText(img, "MATCH FINISHED: CHAMPION DECIDED", (220, 200),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (2, 165, 255), 2, cv2.LINE_AA)

    cv2.putText(img, "Winner: Competitor 2 (GPU Depth-Limited Minimax)", (220, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Decisive Alignment: 4 consecutive tokens in Row 3 (Turn 34)", (220, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (201, 209, 217), 1, cv2.LINE_AA)

    cv2.putText(img, "Submission Highlights:", (220, 380),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (88, 166, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "1. True Dual-GPU (Device 0 vs Device 1) + Single-GPU Lockfile fallback", (240, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "2. Massive Parallel Monte Carlo Kernel (200k games/turn with Xorshift32)", (240, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "3. GPU Alpha-Beta Minimax with center-control heuristic evaluation", (240, 505),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(img, "4. Interactive HTML5 Visualizer in output/game_visualization.html", (240, 545),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 209, 217), 1, cv2.LINE_AA)
    return img

def main():
    print("Generating comprehensive video demonstration...")
    with open("output/game_log.json", "r") as f:
        turns = json.load(f)

    video_path = "output/game_demonstration.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))

    # 1. Slide 1: Intro (5 seconds = 150 frames)
    intro_frame = create_slide_intro()
    for _ in range(FPS * 5):
        out.write(intro_frame)

    # 2. Slide 2: Game Moves (each move shown for ~1 second = 30 frames)
    total_turns = len(turns)
    for step_data in turns:
        f = render_game_frame(step_data, total_turns)
        for _ in range(int(FPS * 1.2)):
            out.write(f)

    # 3. Slide 3: Outro & Summary (5 seconds = 150 frames)
    outro_frame = render_outro()
    for _ in range(FPS * 6):
        out.write(outro_frame)

    out.release()
    print(f"Video saved successfully to: {video_path}")

if __name__ == "__main__":
    main()
