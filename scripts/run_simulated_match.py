"""
Connect-4 Multi-GPU / Competitor Simulator in Python
Simulates the exact behavior of Competitor 1 (Parallel Monte Carlo) vs Competitor 2 (Minimax)
with live CLI rendering, lock-file simulation, and produces output/game_visualization.html and output/game_log.json.
"""

import os
import time
import json
import random

ROWS = 6
COLS = 7

class Board:
    def __init__(self):
        self.cells = [0] * (ROWS * COLS)

    def get(self, r, c):
        if 0 <= r < ROWS and 0 <= c < COLS:
            return self.cells[r * COLS + c]
        return -1

    def set(self, r, c, val):
        self.cells[r * COLS + c] = val

    def is_valid(self, c):
        return 0 <= c < COLS and self.cells[(ROWS - 1) * COLS + c] == 0

    def drop_piece(self, c, player):
        for r in range(ROWS):
            if self.cells[r * COLS + c] == 0:
                self.cells[r * COLS + c] = player
                return r
        return -1

    def is_full(self):
        return all(self.cells[(ROWS - 1) * COLS + c] != 0 for c in range(COLS))

    def check_win(self, player):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(self.get(r, c + i) == player for i in range(4)):
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(self.get(r + i, c) == player for i in range(4)):
                    return True
        # Pos Diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(self.get(r + i, c + i) == player for i in range(4)):
                    return True
        # Neg Diagonal
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(self.get(r - i, c + i) == player for i in range(4)):
                    return True
        return False

    def print_cli(self):
        print("\n+---+---+---+---+---+---+---+")
        for r in range(ROWS - 1, -1, -1):
            row_str = "|"
            for c in range(COLS):
                val = self.get(r, c)
                sym = " X |" if val == 1 else (" O |" if val == 2 else " . |")
                row_str += sym
            print(row_str)
            print("+---+---+---+---+---+---+---+")
        print("  0   1   2   3   4   5   6  (Columns)\n")

# Competitor 1: Monte Carlo Rollouts Simulator
def compute_mcts_move(b, player, rollouts=5000):
    opp = 2 if player == 1 else 1
    # Check immediate wins or blocks
    for c in range(COLS):
        if b.is_valid(c):
            tb = Board()
            tb.cells = list(b.cells)
            tb.drop_piece(c, player)
            if tb.check_win(player):
                return c
    for c in range(COLS):
        if b.is_valid(c):
            tb = Board()
            tb.cells = list(b.cells)
            tb.drop_piece(c, opp)
            if tb.check_win(opp):
                return c

    best_score = -1
    best_col = 3
    for c in [3, 2, 4, 1, 5, 0, 6]:
        if not b.is_valid(c):
            continue
        wins = 0
        per_col_sims = rollouts // 7
        for _ in range(per_col_sims):
            sim_b = Board()
            sim_b.cells = list(b.cells)
            sim_b.drop_piece(c, player)
            curr = opp
            while not sim_b.is_full():
                valids = [vc for vc in range(COLS) if sim_b.is_valid(vc)]
                if not valids:
                    break
                rc = random.choice(valids)
                sim_b.drop_piece(rc, curr)
                if sim_b.check_win(curr):
                    if curr == player:
                        wins += 2
                    break
                curr = 1 if curr == 2 else 2
        if wins > best_score:
            best_score = wins
            best_col = c
    return best_col

# Competitor 2: Minimax Simulator
def score_window(w, player, opp):
    p_cnt = w.count(player)
    o_cnt = w.count(opp)
    e_cnt = w.count(0)
    if p_cnt == 4: return 100000
    if p_cnt == 3 and e_cnt == 1: return 100
    if p_cnt == 2 and e_cnt == 2: return 10
    if o_cnt == 3 and e_cnt == 1: return -120
    if o_cnt == 2 and e_cnt == 2: return -10
    return 0

def evaluate_board(b, player, opp):
    score = 0
    # Center col
    for r in range(ROWS):
        if b.get(r, 3) == player: score += 6
        elif b.get(r, 3) == opp: score -= 6
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += score_window([b.get(r, c+i) for i in range(4)], player, opp)
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += score_window([b.get(r+i, c) for i in range(4)], player, opp)
    # Diagonals
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += score_window([b.get(r+i, c+i) for i in range(4)], player, opp)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            score += score_window([b.get(r-i, c+i) for i in range(4)], player, opp)
    return score

def minimax(b, depth, is_max, alpha, beta, player, opp):
    if b.check_win(player): return 1000000 + depth
    if b.check_win(opp): return -1000000 - depth
    if b.is_full() or depth == 0: return evaluate_board(b, player, opp)

    order = [3, 2, 4, 1, 5, 0, 6]
    if is_max:
        max_eval = -2000000
        for c in order:
            if b.is_valid(c):
                child = Board()
                child.cells = list(b.cells)
                child.drop_piece(c, player)
                ev = minimax(child, depth - 1, False, alpha, beta, player, opp)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha: break
        return max_eval
    else:
        min_eval = 2000000
        for c in order:
            if b.is_valid(c):
                child = Board()
                child.cells = list(b.cells)
                child.drop_piece(c, opp)
                ev = minimax(child, depth - 1, True, alpha, beta, player, opp)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha: break
        return min_eval

def compute_minimax_move(b, player, depth=4):
    opp = 1 if player == 2 else 2
    for c in range(COLS):
        if b.is_valid(c):
            tb = Board()
            tb.cells = list(b.cells)
            tb.drop_piece(c, player)
            if tb.check_win(player): return c
    for c in range(COLS):
        if b.is_valid(c):
            tb = Board()
            tb.cells = list(b.cells)
            tb.drop_piece(c, opp)
            if tb.check_win(opp): return c

    best_val = -99999999
    best_c = 3
    for c in [3, 2, 4, 1, 5, 0, 6]:
        if b.is_valid(c):
            child = Board()
            child.cells = list(b.cells)
            child.drop_piece(c, player)
            val = minimax(child, depth - 1, False, -2000000, 2000000, player, opp)
            if val > best_val:
                best_val = val
                best_c = c
    return best_c

def main():
    print("=" * 63)
    print("   MULTI-GPU COMPETITIVE GAME: CONNECT-4 CHAMPIONSHIP")
    print("   Player 1 (Red / X)   : Monte Carlo Tree Search Rollouts")
    print("   Player 2 (Yellow / O): Depth-Limited Minimax Heuristic")
    print("=" * 63 + "\n")

    b = Board()
    os.makedirs("output", exist_ok=True)
    turn_lock = "output/turn.lock"

    # Start turn with Player 1
    with open(turn_lock, "w") as f:
        f.write("1\n")

    current_player = 1
    turn_num = 1
    turns_log = []

    b.print_cli()

    while not b.is_full():
        # Lock file turn wait
        with open(turn_lock, "r") as f:
            turn_turn = int(f.read().strip() or "1")

        t0 = time.time()
        if current_player == 1:
            col = compute_mcts_move(b, 1, rollouts=3500)
            strat = "GPU-1: MCTS Rollouts"
        else:
            col = compute_minimax_move(b, 2, depth=4)
            strat = "GPU-2: Minimax Alpha-Beta"
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        row = b.drop_piece(col, current_player)
        print(f"[TURN {turn_num}] Player {current_player} ({strat}) placed in Col {col} (Row {row}) in {latency_ms:.2f} ms")
        b.print_cli()

        turns_log.append({
            "turn": turn_num,
            "player": current_player,
            "strategy": strat,
            "column": col,
            "row": row,
            "decision_time_ms": round(latency_ms, 2),
            "board": list(b.cells)
        })

        if b.check_win(current_player):
            print("=" * 63)
            print(f" MATCH FINISHED! WINNER: Player {current_player} ({strat})!")
            print("=" * 63)
            break

        current_player = 2 if current_player == 1 else 1
        with open(turn_lock, "w") as f:
            f.write(str(current_player) + "\n")
        turn_num += 1
        time.sleep(0.1)

    if os.path.exists(turn_lock):
        os.remove(turn_lock)

    # Save to output/game_log.json
    with open("output/game_log.json", "w") as f:
        json.dump(turns_log, f, indent=2)

    # Update output/game_visualization.html
    from visualize_game import HTML_TEMPLATE
    html_out = HTML_TEMPLATE.replace("__TURNS_JSON__", json.dumps(turns_log))
    with open("output/game_visualization.html", "w") as f:
        f.write(html_out)
    print("\nUpdated output/game_log.json and output/game_visualization.html!")

if __name__ == "__main__":
    main()
