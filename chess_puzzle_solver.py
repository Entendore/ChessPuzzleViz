#!/usr/bin/env python3
"""
Chess Puzzle Solver — Procedural Generation Edition (Expert Engine Performance)
47 themes, unlimited algorithmically-generated puzzles, difficulty 1-10, dark UI.
Includes: Negamax + AB Pruning, MCTS (UCT), and Genetic Algorithm (GA) board optimization.
"""

import sys, os, random, math, chess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTextEdit, QFrame, QSplitter,
    QScrollArea, QCheckBox, QGroupBox, QSizePolicy, QToolBar, QStatusBar,
    QProgressBar, QSpacerItem, QToolButton, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QPoint, QRect, QThread
from PySide6.QtGui import (
    QIcon, QFont, QColor, QPalette, QAction, QCursor, QPixmap, QPainter, QBrush, QPen
)

# ═══════════════════════════════════════════════════════════════════════════════════
# THEME REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class ThemeInfo:
    name: str; display_name: str; category: str; base_difficulty: int; description: str; icon_hint: str

THEME_REGISTRY: Dict[str, ThemeInfo] = {
    "fork":             ThemeInfo("fork",             "Fork / Double Attack",      "Attack",     3, "Attack two or more pieces simultaneously with a single move, typically with a knight or pawn.", "\u265e"),
    "pin":              ThemeInfo("pin",              "Pin",                       "Attack",     2, "Restrict an opponent's piece from moving because it would expose a more valuable piece behind it.", "\u2659"),
    "skewer":           ThemeInfo("skewer",           "Skewer",                    "Attack",     4, "Force a valuable piece to move, revealing a less valuable piece behind it that can then be captured.", "\u265c"),
    "discovered_attack":ThemeInfo("discovered_attack","Discovered Attack",         "Attack",     4, "Move one piece to reveal an attack by another piece behind it.", "\u2657"),
    "double_check":     ThemeInfo("double_check",     "Double Check",             "Attack",     6, "Deliver check with two pieces simultaneously, forcing the king to move.", "\u265b"),
    "xray_attack":      ThemeInfo("xray_attack",      "X-Ray Attack",             "Attack",     5, "Attack through an enemy piece to target a piece behind it.", "\u265c"),
    "windmill":         ThemeInfo("windmill",         "Windmill",                 "Attack",     8, "A discovered check pattern where the moving piece repeatedly checks and captures, rotating like a windmill.", "\u265e"),
    "back_rank_mate":   ThemeInfo("back_rank_mate",   "Back Rank Mate",           "Checkmate",  2, "Deliver checkmate on the 1st/8th rank when the king is trapped behind its own pawns.", "\u265c"),
    "smothered_mate":   ThemeInfo("smothered_mate",   "Smothered Mate",           "Checkmate",  5, "Checkmate delivered by a knight when the king is completely surrounded by its own pieces.", "\u265e"),
    "anastasia_mate":   ThemeInfo("anastasia_mate",   "Anastasia's Mate",         "Checkmate",  6, "Checkmate by a knight and rook, with the king trapped between the edge and a pawn.", "\u265e"),
    "arabian_mate":     ThemeInfo("arabian_mate",     "Arabian Mate",             "Checkmate",  5, "Checkmate by a knight and rook working together on adjacent edge squares.", "\u265e"),
    "greco_mate":       ThemeInfo("greco_mate",       "Greco's Mate",             "Checkmate",  5, "Checkmate by a bishop and queen, trapping the king in a corner behind a pawn shield.", "\u2657"),
    "boden_mate":       ThemeInfo("boden_mate",       "Boden's Mate",             "Checkmate",  6, "Two bishops deliver checkmate on crisscrossing diagonals.", "\u2657"),
    "epaulette_mate":   ThemeInfo("epaulette_mate",   "Epaulette Mate",           "Checkmate",  7, "Checkmate where the king is flanked by its own rooks on both sides.", "\u265b"),
    "legal_mate":       ThemeInfo("legal_mate",       "Legal's Mate",             "Checkmate",  4, "A sacrificial theme where pieces lure the king into a smothered mate pattern.", "\u265e"),
    "scholar_mate":     ThemeInfo("scholar_mate",     "Scholar's Mate",           "Checkmate",  1, "The classic 4-move checkmate using queen and bishop targeting f7/f2.", "\u265b"),
    "blackburne_mate":  ThemeInfo("blackburne_mate",  "Blackburne's Mate",        "Checkmate",  7, "Checkmate by a bishop and two knights with the king trapped by its own pawn.", "\u2658"),
    "morphy_mate":      ThemeInfo("morphy_mate",      "Morphy's Mate",            "Checkmate",  6, "Checkmate by a bishop and rook where the bishop controls escape squares.", "\u265c"),
    "damiano_mate":     ThemeInfo("damiano_mate",     "Damiano's Mate",           "Checkmate",  5, "Checkmate by a queen with support from a pawn, trapping the king at the edge.", "\u265b"),
    "opera_mate":       ThemeInfo("opera_mate",       "Opera Mate",               "Checkmate",  5, "A rook delivers checkmate on the back rank supported by a bishop.", "\u265c"),
    "queen_sacrifice":  ThemeInfo("queen_sacrifice",  "Queen Sacrifice",          "Sacrifice",  7, "Sacrifice the queen to gain a decisive advantage or checkmate.", "\u265b"),
    "rook_sacrifice":   ThemeInfo("rook_sacrifice",   "Rook Sacrifice",           "Sacrifice",  6, "Sacrifice a rook to open lines, deflect defenders, or force a mating sequence.", "\u265c"),
    "bishop_sacrifice": ThemeInfo("bishop_sacrifice", "Bishop Sacrifice",         "Sacrifice",  5, "Sacrifice a bishop to expose the king or destroy pawn cover.", "\u2657"),
    "knight_sacrifice": ThemeInfo("knight_sacrifice", "Knight Sacrifice",         "Sacrifice",  5, "Sacrifice a knight to remove a key defender or open lines.", "\u2658"),
    "greek_gift":       ThemeInfo("greek_gift",       "Greek Gift (Bxh7+/Bxh2+)", "Sacrifice",  6, "The classic bishop sacrifice on h7 (or h2) followed by Ng5+ and a crushing attack.", "\u2657"),
    "pawn_sacrifice":   ThemeInfo("pawn_sacrifice",   "Pawn Sacrifice",           "Sacrifice",  4, "Sacrifice a pawn to gain tempo, open lines, or deflect enemy pieces.", "\u2659"),
    "exchange_sacrifice":ThemeInfo("exchange_sacrifice","Exchange Sacrifice",      "Sacrifice",  6, "Sacrifice a rook for a bishop or knight for positional/tactical advantage.", "\u265c"),
    "deflection":       ThemeInfo("deflection",       "Deflection",               "Tactic",     4, "Lure an enemy piece away from a key defensive square or duty.", "\u265b"),
    "decoy":            ThemeInfo("decoy",            "Decoy / Lure",             "Tactic",     4, "Entice an enemy piece to a specific square where it becomes vulnerable.", "\u265b"),
    "clearance":        ThemeInfo("clearance",        "Clearance",                "Tactic",     5, "Move a piece out of the way to clear a line for a more powerful piece.", "\u265c"),
    "zwischenzug":      ThemeInfo("zwischenzug",      "Zwischenzug (Intermezzo)", "Tactic",     6, "An intermediate move between expected moves that changes the outcome.", "\u265b"),
    "remove_defender":  ThemeInfo("remove_defender",  "Remove the Defender",      "Tactic",     5, "Capture or chase away a piece defending a key square or valuable piece.", "\u265c"),
    "underpromotion":   ThemeInfo("underpromotion",   "Underpromotion",           "Tactic",     7, "Promote to a knight, bishop, or rook instead of queen to deliver check or avoid stalemate.", "\u2658"),
    "interference":     ThemeInfo("interference",     "Interference",             "Tactic",     6, "Place a piece between two enemy pieces that defend each other, breaking coordination.", "\u265e"),
    "trapped_piece":    ThemeInfo("trapped_piece",    "Trapped Piece",            "Tactic",     3, "Trap an opponent's piece so it has no safe squares to move to.", "\u265f"),
    "weak_back_rank":   ThemeInfo("weak_back_rank",   "Weak Back Rank",           "Tactic",     3, "Exploit the opponent's lack of back-rank protection where the king is trapped.", "\u265c"),
    "overloaded_piece": ThemeInfo("overloaded_piece", "Overloaded Piece",         "Tactic",     5, "Exploit a piece defending multiple targets that cannot handle all threats.", "\u265b"),
    "attraction":       ThemeInfo("attraction",       "Attraction",               "Tactic",     5, "Force an enemy piece to a specific square where it becomes a liability.", "\u265b"),
    "line_clearance":   ThemeInfo("line_clearance",   "Line Clearance",           "Tactic",     4, "Sacrifice or move a piece to open a file, rank, or diagonal for a decisive attack.", "\u2656"),
    "square_vacation":  ThemeInfo("square_vacation",  "Square Vacation",          "Tactic",     5, "Move a piece from a critical square so another piece can use it.", "\u2658"),
    "zugzwang":         ThemeInfo("zugzwang",         "Zugzwang",                 "Tactic",     7, "Force the opponent into a position where any move worsens their position.", "\u265f"),
    "stalemate_trick":  ThemeInfo("stalemate_trick",  "Stalemate Trick",          "Tactic",     6, "Deliberately aim for stalemate when in a losing position to salvage a draw.", "\u265f"),
    "king_pawn_endgame":ThemeInfo("king_pawn_endgame","King & Pawn Endgame",       "Endgame",    4, "Key king and pawn endgame techniques: opposition, triangulation, and promotion races.", "\u2659"),
    "rook_endgame":     ThemeInfo("rook_endgame",     "Rook Endgame",             "Endgame",    5, "Essential rook endgame patterns: Lucena position, Philidor defense, back-rank threats.", "\u265c"),
    "bishop_endgame":   ThemeInfo("bishop_endgame",   "Bishop Endgame",           "Endgame",    5, "Bishop and pawn endgames including wrong-colored bishop and opposite-colored draws.", "\u2657"),
    "knight_endgame":   ThemeInfo("knight_endgame",   "Knight Endgame",           "Endgame",    5, "Knight endgame tactics: knight vs. pawns, knight fork tricks, king-knight coordination.", "\u2658"),
    "queen_endgame":    ThemeInfo("queen_endgame",    "Queen Endgame",            "Endgame",    6, "Queen endgame patterns including perpetual check, queen vs. pawn, and mating nets.", "\u265b"),
}
CATEGORIES = sorted(set(t.category for t in THEME_REGISTRY.values()))
def get_themes_by_category(cat): return [t for t in THEME_REGISTRY.values() if t.category == cat]
def get_all_themes(): return list(THEME_REGISTRY.values())

FILLED_STAR = '\u2605'
EMPTY_STAR  = '\u2606'

# ═══════════════════════════════════════════════════════════════════════════════════
# EXPERT SEARCH ENGINES (Negamax + AB, MCTS, Genetic Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Puzzle:
    fen: str; theme_key: str; theme_name: str; difficulty: int
    moves: List[str]; description: str; hint: str; category: str

# Piece Values
PV = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
MATE_SCORE = 100000

# Global Transposition Table
TT = {}

def _eval_negamax(board):
    if board.is_checkmate(): return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return 0
    s = 0
    for sq, p in board.piece_map().items():
        v = PV[p.piece_type]
        if p.piece_type in (chess.KNIGHT, chess.BISHOP):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            v += max(0, int((7 - abs(3.5 - f) - abs(3.5 - r)) * 5))
        s += v if p.color == board.turn else -v
    return s

def _quiesce(board, alpha, beta):
    if board.is_checkmate(): return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material(): return 0
    stand_pat = _eval_negamax(board)
    if stand_pat >= beta: return beta
    if alpha < stand_pat: alpha = stand_pat
    if board.is_check(): moves = list(board.legal_moves)
    else: moves = list(board.generate_legal_captures())
        
    def _ms(m):
        victim = board.piece_type_at(m.to_square)
        if victim is None: victim = chess.PAWN
        attacker = board.piece_type_at(m.from_square)
        return PV.get(victim, 0) * 10 - PV.get(attacker, 0)
    moves.sort(key=_ms, reverse=True)

    for move in moves:
        board.push(move)
        score = -_quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta: return beta
        if score > alpha: alpha = score
    return alpha

def _search(board, depth, alpha, beta):
    if board.is_checkmate(): return None, -MATE_SCORE - depth
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return None, 0
    if depth == 0: return None, _quiesce(board, alpha, beta)
        
    key = board._transposition_key()
    alpha_orig = alpha
    tt_move = None
    if key in TT:
        tt_depth, tt_score, tt_flag, tt_move = TT[key]
        if tt_depth >= depth:
            if tt_flag == 0: return tt_move, tt_score
            elif tt_flag == 1: alpha = max(alpha, tt_score)
            elif tt_flag == 2: beta = min(beta, tt_score)
            if alpha >= beta: return tt_move, tt_score

    moves = list(board.legal_moves)
    def _ms(m):
        s = 0
        if m == tt_move: s += 100000
        if m.promotion: s += 10000 + PV.get(m.promotion, 0)
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square)
            if victim is None: victim = chess.PAWN
            attacker = board.piece_type_at(m.from_square)
            s += 1000 + (PV.get(victim, 0) * 10) - PV.get(attacker, 0)
        if board.gives_check(m): s += 500
        return -s
    moves.sort(key=_ms)
    
    bm, bs = None, -float('inf')
    for m in moves:
        board.push(m)
        _, sc = _search(board, depth - 1, -beta, -alpha)
        sc = -sc
        board.pop()
        if sc > bs:
            bs, bm = sc, m
        if bs > alpha: alpha = bs
        if alpha >= beta: break
            
    flag = 0 if bs <= alpha_orig else (2 if bs >= beta else 1)
    TT[key] = (depth, bs, flag, bm)
    return bm, bs

def _search_id(board, max_depth=3):
    bm, bs = None, 0
    for d in range(1, max_depth + 1):
        bm, bs = _search(board, d, -float('inf'), float('inf'))
        if bs > MATE_SCORE - 100: break
    return bm, bs

def _get_solution(board, first_move, max_depth=4):
    moves = [first_move.uci()]
    b = board.copy(); b.push(first_move)
    for _ in range(max_depth):
        if b.is_game_over(): break
        m, _ = _search_id(b, 2)
        if m is None: break
        moves.append(m.uci()); b.push(m)
    return moves

# ─── MCTS (Monte Carlo Tree Search) with UCT ────────────────────────────────────

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_uct_child(self, c_param=1.414):
        log_n = math.log(self.visits) if self.visits > 0 else 0
        return max(self.children, key=lambda c: (c.wins / c.visits) + c_param * math.sqrt(log_n / c.visits))

def _mcts_search(board, iterations=800):
    root = MCTSNode(board.copy())
    
    for _ in range(iterations):
        node = root
        sim_board = board.copy()
        
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_uct_child()
            sim_board.push(node.move)
            
        # 2. Expansion
        if not sim_board.is_game_over() and node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            sim_board.push(move)
            child_node = MCTSNode(sim_board.copy(), parent=node, move=move)
            node.children.append(child_node)
            node = child_node
            
        # 3. Simulation (Depth-limited capture-biased rollout)
        score = 0
        if sim_board.is_checkmate():
            score = 1 if sim_board.turn != board.turn else -1
        else:
            curr = sim_board.copy()
            for _ in range(8):  # Increased depth for deeper mates
                if curr.is_game_over(): break
                moves = list(curr.legal_moves)
                caps = [m for m in moves if curr.is_capture(m)]
                checks = [m for m in moves if curr.gives_check(m)]
                # Prioritize checks, then captures
                pool = checks if checks else (caps if caps else moves)
                m = random.choice(pool)
                curr.push(m)
            if curr.is_checkmate():
                score = 1 if curr.turn != board.turn else -1
            else:
                ev = _eval_negamax(curr)
                score = max(-1.0, min(1.0, ev / 2000.0))
                if board.turn != curr.turn: score = -score
                
        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.wins += score
            score = -score
            node = node.parent
            
    if not root.children: return None, 0
    best = max(root.children, key=lambda c: c.visits)
    return best.move, best.wins / best.visits

# ─── Genetic Algorithm (GA) for Board Optimization ──────────────────────────────

@dataclass
class Individual:
    pieces: List[Tuple[int, int, int]] # sq, ptype, color
    fitness: float = 0.0

def _ga_optimize_filler(core_pieces, eval_fn, population_size=15, generations=5):
    def random_individual():
        pieces = list(core_pieces)
        num_fillers = random.randint(0, 4)
        for _ in range(num_fillers):
            sq = random.randint(0, 63)
            if any(p[0] == sq for p in pieces): continue
            ptype = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK])
            color = random.choice([chess.WHITE, chess.BLACK])
            pieces.append((sq, ptype, color))
        return Individual(pieces)
        
    population = [random_individual() for _ in range(population_size)]
    
    for gen in range(generations):
        for ind in population:
            b = _eb()
            for sq, pt, c in ind.pieces: _p(b, sq, pt, c)
            ind.fitness = eval_fn(b)
            
        population.sort(key=lambda x: x.fitness, reverse=True)
        next_gen = population[:max(2, population_size // 5)]
        
        while len(next_gen) < population_size:
            p1 = random.choice(population[:max(2, population_size//2)])
            p2 = random.choice(population[:max(2, population_size//2)])
            
            core_set = set(core_pieces)
            child_pieces = list(core_set)
            p1_fillers = [p for p in p1.pieces if p not in core_set]
            p2_fillers = [p for p in p2.pieces if p not in core_set]
            
            for p in p1_fillers[:len(p1_fillers)//2] + p2_fillers[len(p2_fillers)//2:]:
                if not any(cp[0] == p[0] for cp in child_pieces):
                    child_pieces.append(p)
                    
            if random.random() < 0.4:
                child_fillers = [p for p in child_pieces if p not in core_set]
                if child_fillers and random.random() < 0.5:
                    child_pieces.remove(random.choice(child_fillers))
                else:
                    sq = random.randint(0, 63)
                    if not any(cp[0] == sq for cp in child_pieces):
                        ptype = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK])
                        color = random.choice([chess.WHITE, chess.BLACK])
                        child_pieces.append((sq, ptype, color))
            next_gen.append(Individual(child_pieces))
            
        population = next_gen
        
    for ind in population:
        b = _eb()
        for sq, pt, c in ind.pieces: _p(b, sq, pt, c)
        ind.fitness = eval_fn(b)
        
    return max(population, key=lambda x: x.fitness).pieces

# ─── Board construction helpers ─────────────────────────────────────────────────

def _eb():
    b = chess.Board.empty()
    b.turn = chess.WHITE
    b.castling_rights = chess.BB_EMPTY
    b.ep_square = None
    return b

def _p(b, sq, pt, c): b.set_piece_at(sq, chess.Piece(pt, c))

def _knight_atks(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq); out = []
    for df, dr in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nf, nr = f+df, r+dr
        if 0 <= nf <= 7 and 0 <= nr <= 7: out.append(chess.square(nf, nr))
    return out

def _slide(sq, dirs):
    f, r = chess.square_file(sq), chess.square_rank(sq); out = []
    for df, dr in dirs:
        nf, nr = f+df, r+dr
        while 0 <= nf <= 7 and 0 <= nr <= 7: out.append(chess.square(nf, nr)); nf += df; nr += dr
    return out

def _diags(sq): return _slide(sq, [(-1,-1),(-1,1),(1,-1),(1,1)])
def _orth(sq): return _slide(sq, [(-1,0),(1,0),(0,-1),(0,1)])

def _line_clear(b, s1, s2):
    f1,r1 = chess.square_file(s1),chess.square_rank(s1)
    f2,r2 = chess.square_file(s2),chess.square_rank(s2)
    df = (1 if f2>f1 else -1) if f2!=f1 else 0
    dr = (1 if r2>r1 else -1) if r2!=r1 else 0
    f,r = f1+df, r1+dr
    while (f,r)!=(f2,r2):
        if b.piece_at(chess.square(f,r)): return False
        f+=df; r+=dr
    return True

def _add_filler(b, n_white, n_black, avoid=set()):
    pieces = [chess.PAWN]*3 + [chess.KNIGHT]*2 + [chess.BISHOP]*2 + [chess.ROOK]*1
    for c, cnt in [(chess.WHITE, n_white), (chess.BLACK, n_black)]:
        for _ in range(cnt * 5):
            if cnt <= 0: break
            sq = random.randint(0, 63)
            if sq in avoid or b.piece_at(sq): continue
            ranks = range(1,7) if c == chess.WHITE else range(1,7)
            if chess.square_rank(sq) not in ranks: continue
            _p(b, sq, random.choice(pieces), c); cnt -= 1

def _place_kings(b, wking_ranks=range(0,2), bking_ranks=range(6,8)):
    for _ in range(100):
        wk = chess.square(random.randint(0,7), random.choice(list(wking_ranks)))
        bk = chess.square(random.randint(0,7), random.choice(list(bking_ranks)))
        if abs(chess.square_file(wk)-chess.square_file(bk)) + abs(chess.square_rank(wk)-chess.square_rank(bk)) < 3: continue
        _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK); return wk, bk
    return None, None

def _rand_diff(dr): return random.randint(max(1,dr[0]), min(10,dr[1]))

# ═══════════════════════════════════════════════════════════════════════════════════
# PROCEDURAL PUZZLE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════════

class ProceduralGenerator:
    def generate(self, theme_key=None, category=None, difficulty=None, difficulty_range=(1,10), count=10):
        global TT
        TT.clear()
        
        themes = [theme_key] if theme_key else (
            [t.name for t in THEME_REGISTRY.values() if t.category == category] if category else list(THEME_REGISTRY.keys()))
        puzzles = []
        for tk in themes:
            gen = getattr(self, f"_g_{tk}", None)
            if not gen: gen = self._g_general
            for _ in range(count * 5):
                p = gen(tk, difficulty_range)
                if p:
                    if difficulty and p.difficulty != difficulty: continue
                    puzzles.append(p)
                    if len(puzzles) >= count: break
        random.shuffle(puzzles)
        return puzzles[:count]

    def get_random_puzzle(self, difficulty_range=(1,10)):
        tk = random.choice(list(THEME_REGISTRY.keys()))
        p = self.generate(theme_key=tk, difficulty_range=difficulty_range, count=1)
        return p[0] if p else None

    def get_available_themes(self):
        r = {}
        for tk in THEME_REGISTRY: r.setdefault(THEME_REGISTRY[tk].category, []).append(tk)
        return r

    def _is_unique_mate(self, b, best_move, depth):
        """Ensure no other move also leads to a mate."""
        for move in b.legal_moves:
            if move == best_move: continue
            b.push(move)
            _, sc = _search_id(b, max(1, depth - 1))
            b.pop()
            if -sc >= MATE_SCORE - 100:
                return False
        return True

    def _is_unique_tactic(self, b, best_move, min_adv, depth):
        """Ensure no other move yields a similarly high advantage."""
        for move in b.legal_moves:
            if move == best_move: continue
            b.push(move)
            _, sc = _search_id(b, max(1, depth - 1))
            b.pop()
            if -sc >= min_adv - 50:
                return False
        return True

    def _try_mate(self, tk, dr, setup_fn, check_fn=None, depth=2):
        for _ in range(80):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, s = _search_id(b, depth)
            if not m or s < 9000: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            if not self._is_unique_mate(b, m, depth): continue  # Uniqueness validation
            
            pv = _get_solution(b, m, 3)
            ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_mate_mcts(self, tk, dr, setup_fn, check_fn=None, iterations=600):
        for _ in range(40):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, wr = _mcts_search(b, iterations)
            if not m: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            
            # Verify with regular search and uniqueness
            m_search, s_search = _search_id(b, 2)
            if not m_search or s_search < 9000: continue
            if not self._is_unique_mate(b, m_search, 2): continue
            
            pv = _get_solution(b, m_search, 3)
            ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle (MCTS-Verified).",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_mate_ga(self, tk, dr, core_setup_fn, check_fn=None, depth=2):
        for _ in range(15): 
            core_pieces = core_setup_fn()
            if not core_pieces: continue
            
            def eval_fn(board):
                if not board.is_valid() or board.is_game_over(): return -1000
                m, s = _search_id(board, depth)
                if not m or s < 9000: return -1000
                tb = board.copy(); tb.push(m)
                if not tb.is_checkmate(): return -1000
                if check_fn and not check_fn(board, m): return -1000
                
                # GA Fitness heavily rewards uniqueness
                fitness = 1000 - (len(board.piece_map()) * 15)
                mating_moves = 0
                for move in board.legal_moves:
                    if move == m: continue
                    board.push(move)
                    _, sc = _search_id(board, max(1, depth - 1))
                    board.pop()
                    if -sc >= MATE_SCORE - 100:
                        mating_moves += 1
                if mating_moves == 0: fitness += 500
                else: fitness -= 300 * mating_moves
                return fitness
                
            best_pieces = _ga_optimize_filler(core_pieces, eval_fn, 15, 4)
            b = _eb()
            for sq, pt, c in best_pieces: _p(b, sq, pt, c)
            
            if not b.is_valid() or b.is_game_over(): continue
            m, s = _search_id(b, depth)
            if not m or s < 9000: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            if not self._is_unique_mate(b, m, depth): continue
            
            pv = _get_solution(b, m, 3)
            ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle (GA-Optimized).",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_tactic(self, tk, dr, setup_fn, check_fn=None, min_adv=200, depth=3):
        for _ in range(80):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, s = _search_id(b, depth)
            if not m or s < min_adv: continue
            if check_fn and not check_fn(b, m): continue
            if not self._is_unique_tactic(b, m, min_adv, depth): continue  # Uniqueness validation
            
            pv = _get_solution(b, m, 4)
            ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the best tactical move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} opportunity.", category=ti.category)
        return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # MATE GENERATORS (Using MCTS and GA)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _g_back_rank_mate(self, tk, dr):
        def setup(b):
            kf = random.randint(4, 7)
            bk_sq = chess.square(kf, 7)
            _p(b, bk_sq, chess.KING, chess.BLACK)
            for df in [-1, 0, 1]:
                f = kf + df
                if 0 <= f <= 7 and f != kf: _p(b, chess.square(f, 6), chess.PAWN, chess.BLACK)
            rf = random.randint(0, 7)
            while rf == kf: rf = random.randint(0, 7)
            for r in range(1, 8):
                sq = chess.square(rf, r)
                if b.piece_at(sq): b.remove_piece_at(sq)
            _p(b, chess.square(rf, 1), chess.ROOK, chess.WHITE)
            for _ in range(100):
                wk = chess.square(random.randint(0,7), random.randint(0,3))
                if abs(chess.square_file(wk)-kf) + abs(chess.square_rank(wk)-7) >= 3:
                    _p(b, wk, chess.KING, chess.WHITE); break
            else:
                return False
            _add_filler(b, random.randint(0,1), random.randint(1,3), {b.king(chess.WHITE), b.king(chess.BLACK)})
            return True
        return self._try_mate_mcts(tk, dr, setup, lambda b,m: chess.square_rank(m.to_square)==7, iterations=500)

    def _g_scholar_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.E1, chess.KING, chess.WHITE); _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D1, chess.QUEEN, chess.WHITE); _p(b, chess.F1, chess.BISHOP, chess.WHITE)
            _p(b, chess.E7, chess.PAWN, chess.BLACK); _p(b, chess.F7, chess.PAWN, chess.BLACK)
            _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.D7, chess.PAWN, chess.BLACK)
            _p(b, chess.D8, chess.QUEEN, chess.BLACK); _p(b, chess.C8, chess.BISHOP, chess.BLACK)
            _p(b, chess.B8, chess.KNIGHT, chess.BLACK); _p(b, chess.G8, chess.KNIGHT, chess.BLACK)
            _p(b, chess.F1, chess.BISHOP, chess.WHITE); _p(b, chess.C4, chess.BISHOP, chess.WHITE)
            _p(b, chess.H5, chess.QUEEN, chess.WHITE)
            return True
        return self._try_mate(tk, dr, setup, lambda b,m: m.to_square==chess.F7, depth=2)

    def _g_smothered_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            for s in [chess.F8, chess.H8, chess.F7, chess.H7]:
                if random.random() < 0.7: _p(b, s, random.choice([chess.ROOK, chess.PAWN, chess.QUEEN]), chess.BLACK)
            nsq = random.choice([chess.E5, chess.F6, chess.H6, chess.E7])
            _p(b, nsq, chess.KNIGHT, chess.WHITE)
            wk = chess.square(random.randint(0,4), random.randint(0,3))
            _p(b, wk, chess.KING, chess.WHITE)
            _add_filler(b, 0, 0)
            return True
        return self._try_mate_mcts(tk, dr, setup, iterations=500)

    def _g_arabian_mate(self, tk, dr):
        def setup(b):
            kf = random.choice([7, 6])
            _p(b, chess.square(kf, 7), chess.KING, chess.BLACK)
            _p(b, chess.square(kf, 6), chess.PAWN, chess.BLACK)
            rf = kf - 1
            if rf < 0: return False
            _p(b, chess.square(rf, 0), chess.ROOK, chess.WHITE)
            nsq = chess.square(kf - 1, 6)
            if nsq < 0: return False
            _p(b, nsq, chess.KNIGHT, chess.WHITE)
            wk = chess.square(random.randint(0,3), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_anastasia_mate(self, tk, dr):
        def core_setup():
            return [
                (chess.H7, chess.KING, chess.BLACK),
                (chess.G7, chess.PAWN, chess.BLACK),
                (chess.E7, chess.KNIGHT, chess.WHITE),
                (chess.G1, chess.ROOK, chess.WHITE),
                (chess.B2, chess.KING, chess.WHITE)
            ]
        return self._try_mate_ga(tk, dr, core_setup, depth=2)

    def _g_epaulette_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D8, chess.ROOK, chess.BLACK); _p(b, chess.F8, chess.ROOK, chess.BLACK)
            _p(b, random.choice([chess.A1, chess.B1, chess.C1]), chess.QUEEN, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_greco_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.BISHOP, chess.WHITE); _p(b, chess.D1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_boden_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.C8, chess.KING, chess.BLACK)
            _p(b, chess.B8, chess.ROOK, chess.BLACK); _p(b, chess.D8, chess.ROOK, chess.BLACK)
            _p(b, chess.A6, chess.BISHOP, chess.WHITE); _p(b, chess.F6, chess.BISHOP, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_damiano_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, random.choice([chess.E5, chess.F5, chess.G5]), chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_opera_mate(self, tk, dr):
        def core_setup():
            return [
                (chess.G8, chess.KING, chess.BLACK),
                (chess.F7, chess.PAWN, chess.BLACK),
                (chess.G7, chess.PAWN, chess.BLACK),
                (chess.D3, chess.BISHOP, chess.WHITE),
                (chess.A1, chess.ROOK, chess.WHITE), 
                (chess.A2, chess.KING, chess.WHITE)
            ]
        return self._try_mate_ga(tk, dr, core_setup, depth=2)

    def _g_morphy_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F8, chess.ROOK, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, chess.E1, chess.ROOK, chess.WHITE); _p(b, chess.H5, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_blackburne_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.H7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.BISHOP, chess.WHITE); _p(b, chess.F6, chess.KNIGHT, chess.WHITE); _p(b, chess.G5, chess.KNIGHT, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_legal_mate(self, tk, dr):
        return self._g_smothered_mate(tk, dr)

    # ═══════════════════════════════════════════════════════════════════════════════
    # ATTACK GENERATORS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _g_fork(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            nsq = random.choice([chess.C4, chess.D4, chess.D5, chess.E5, chess.E4, chess.F5, chess.C5, chess.F4])
            atks = _knight_atks(nsq)
            targets = random.sample([s for s in atks if s != chess.H8 and s != nsq], min(2, len(atks)-1))
            if len(targets) < 2: return False
            ttypes = random.sample([chess.ROOK, chess.QUEEN, chess.BISHOP, chess.ROOK], 2)
            for i, t in enumerate(targets): _p(b, t, ttypes[i], chess.BLACK)
            _p(b, nsq, chess.KNIGHT, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(0,2), random.randint(0,2))
            return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=2)

    def _g_pin(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            pin_file = random.randint(1, 6)
            _p(b, chess.square(pin_file, 6), random.choice([chess.ROOK, chess.QUEEN, chess.BISHOP]), chess.BLACK)
            _p(b, chess.E1, chess.BISHOP, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(1,3), random.randint(1,3))
            return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_skewer(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.ROOK, chess.WHITE)
            mid = random.choice([chess.E6, chess.E7])
            _p(b, mid, chess.QUEEN, chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    def _g_discovered_attack(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.BISHOP, chess.WHITE); _p(b, chess.E5, chess.KNIGHT, chess.WHITE)
            _p(b, random.choice([chess.D7, chess.E7]), random.choice([chess.QUEEN, chess.ROOK]), chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, 0, random.randint(1,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_double_check(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.KNIGHT, chess.WHITE); _p(b, chess.B2, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE)
            _add_filler(b, 0, random.randint(0,2)); return True
        return self._try_mate(tk, dr, setup, lambda b,m: len(b.checkers())>=2 if b.gives_check(m) else False, depth=2)

    def _g_xray_attack(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E1, chess.ROOK, chess.WHITE); _p(b, chess.E5, chess.ROOK, chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_windmill(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.F7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.KNIGHT, chess.WHITE); _p(b, chess.B2, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE)
            _p(b, chess.D7, random.choice([chess.QUEEN, chess.ROOK]), chess.BLACK)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=500, depth=3)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SACRIFICE GENERATORS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _g_queen_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, chess.H5, chess.QUEEN, chess.WHITE); _p(b, chess.E1, chess.ROOK, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, lambda b,m: b.piece_type_at(m.from_square)==chess.QUEEN, depth=3)

    def _g_rook_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, chess.E1, chess.ROOK, chess.WHITE); _p(b, chess.F1, chess.BISHOP, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, lambda b,m: b.piece_type_at(m.from_square)==chess.ROOK, depth=3)

    def _g_bishop_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.B4, chess.BISHOP, chess.WHITE); _p(b, chess.F3, chess.KNIGHT, chess.WHITE)
            _p(b, random.choice([chess.D7, chess.D8]), chess.QUEEN, chess.BLACK)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    def _g_knight_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.KNIGHT, chess.WHITE); _p(b, chess.E1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, lambda b,m: b.piece_type_at(m.from_square)==chess.KNIGHT, depth=3)

    def _g_greek_gift(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            _p(b, chess.D3, chess.BISHOP, chess.WHITE); _p(b, chess.F3, chess.KNIGHT, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, lambda b,m: b.piece_type_at(m.from_square)==chess.BISHOP and m.to_square==chess.H7, depth=3)

    def _g_pawn_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E5, chess.PAWN, chess.WHITE); _p(b, chess.F3, chess.KNIGHT, chess.WHITE)
            _p(b, chess.E4, chess.PAWN, chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, 1, random.randint(1,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=150, depth=3)

    def _g_exchange_sacrifice(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.E1, chess.ROOK, chess.WHITE); _p(b, chess.D4, chess.KNIGHT, chess.WHITE)
            _p(b, random.choice([chess.D7, chess.E7]), chess.QUEEN, chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TACTIC GENERATORS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _g_deflection(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.C8, chess.ROOK, chess.BLACK)
            _p(b, chess.A1, chess.ROOK, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(1,2), random.randint(0,1)); return True
        return self._try_mate(tk, dr, setup, depth=3)

    def _g_decoy(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D8, chess.QUEEN, chess.BLACK)
            _p(b, chess.A1, chess.ROOK, chess.WHITE); _p(b, chess.B4, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_clearance(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.BISHOP, chess.WHITE); _p(b, chess.E1, chess.ROOK, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, 0, random.randint(1,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_zwischenzug(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK)
            _p(b, chess.E1, chess.QUEEN, chess.WHITE); _p(b, chess.E5, chess.KNIGHT, chess.WHITE)
            _p(b, chess.D7, random.choice([chess.ROOK, chess.BISHOP]), chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_remove_defender(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F6, chess.KNIGHT, chess.BLACK)
            _p(b, chess.D4, chess.KNIGHT, chess.WHITE); _p(b, chess.D1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_underpromotion(self, tk, dr):
        def setup(b):
            corner = random.choice([chess.A8, chess.H8])
            kf = chess.square_file(corner)
            _p(b, corner, chess.KING, chess.BLACK)
            for df in [-1, 0, 1]:
                f = kf + df
                if 0 <= f <= 7 and f != kf:
                    sq = chess.square(f, 7)
                    if random.random() < 0.7:
                        _p(b, sq, random.choice([chess.ROOK, chess.PAWN, chess.BISHOP]), chess.BLACK)
            for df in [-1, 1]:
                f = kf + df
                if 0 <= f <= 7:
                    sq7 = chess.square(f, 6)
                    if not b.piece_at(sq7) and random.random() < 0.5:
                        _p(b, sq7, random.choice([chess.PAWN, chess.BISHOP]), chess.BLACK)
            pawn_f = (kf - 1) if kf > 0 else (kf + 1)
            _p(b, chess.square(pawn_f, 6), chess.PAWN, chess.WHITE)
            support_files = [f for f in range(8) if f != pawn_f and f != kf]
            if support_files:
                sf = random.choice(support_files)
                _p(b, chess.square(sf, random.randint(3,5)), random.choice([chess.QUEEN, chess.ROOK]), chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            return True
        return self._try_mate(tk, dr, setup, lambda b,m: bool(m.promotion) and m.promotion != chess.QUEEN, depth=3)

    def _g_interference(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D7, chess.BISHOP, chess.BLACK); _p(b, chess.E6, chess.ROOK, chess.BLACK)
            _p(b, chess.D4, chess.KNIGHT, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_trapped_piece(self, tk, dr):
        def setup(b):
            _p(b, chess.A1, chess.KING, chess.BLACK)
            _p(b, chess.H8, chess.ROOK, chess.BLACK)
            _p(b, chess.H1, chess.ROOK, chess.WHITE); _p(b, chess.G1, chess.KNIGHT, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_weak_back_rank(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            rf = random.randint(0, 5)
            _p(b, chess.square(rf, 1), chess.ROOK, chess.WHITE)
            for r in range(2, 8):
                sq = chess.square(rf, r)
                if b.piece_at(sq) and b.piece_type_at(sq) != chess.KING: b.remove_piece_at(sq)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, 0, random.randint(0,2)); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_overloaded_piece(self, tk, dr):
        def setup(b):
            _p(b, chess.A8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.QUEEN, chess.BLACK)
            _p(b, chess.D1, chess.ROOK, chess.WHITE); _p(b, chess.C4, chess.KNIGHT, chess.WHITE)
            _p(b, chess.H1, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_attraction(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D1, chess.QUEEN, chess.WHITE); _p(b, chess.B4, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_line_clearance(self, tk, dr):
        return self._g_clearance(tk, dr)

    def _g_square_vacation(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.KNIGHT, chess.WHITE); _p(b, chess.E1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, 0, random.randint(0,1)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_stalemate_trick(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK); _p(b, chess.G8, chess.PAWN, chess.BLACK)
            _p(b, chess.F6, chess.KING, chess.WHITE); _p(b, chess.E5, chess.QUEEN, chess.WHITE)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=100, depth=2)

    def _g_zugzwang(self, tk, dr):
        def setup(b):
            wk = chess.square(random.randint(3,5), random.randint(4,6))
            bk = chess.square(random.randint(3,5), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK)
            pf = random.randint(2,5); _p(b, chess.square(pf, 5), chess.PAWN, chess.WHITE)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=100, depth=3)

    # ═══════════════════════════════════════════════════════════════════════════════
    # ENDGAME GENERATORS
    # ═══════════════════════════════════════════════════════════════════════════════

    def _g_king_pawn_endgame(self, tk, dr):
        def setup(b):
            wk = chess.square(random.randint(3,5), random.randint(4,6))
            bk = chess.square(random.randint(3,5), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK)
            pf = random.randint(2,5); _p(b, chess.square(pf, random.randint(3,5)), chess.PAWN, chess.WHITE)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=100, depth=4)

    def _g_rook_endgame(self, tk, dr):
        def setup(b):
            wk = chess.square(random.randint(3,5), random.randint(5,7))
            bk = chess.square(random.randint(3,5), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK)
            _p(b, chess.A1, chess.ROOK, chess.WHITE)
            _p(b, chess.square(random.randint(4,7), random.randint(5,7)), chess.PAWN, chess.BLACK)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=4)

    def _g_bishop_endgame(self, tk, dr):
        def setup(b):
            wk = chess.square(random.randint(3,5), random.randint(5,7))
            bk = chess.square(random.randint(3,5), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK)
            _p(b, chess.C1, chess.BISHOP, chess.WHITE)
            _p(b, chess.square(random.randint(3,6), random.randint(4,6)), chess.PAWN, chess.WHITE)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=100, depth=4)

    def _g_knight_endgame(self, tk, dr):
        def setup(b):
            wk = chess.square(random.randint(3,5), random.randint(5,7))
            bk = chess.square(random.randint(3,5), random.randint(0,2))
            _p(b, wk, chess.KING, chess.WHITE); _p(b, bk, chess.KING, chess.BLACK)
            _p(b, chess.D4, chess.KNIGHT, chess.WHITE)
            _p(b, chess.square(random.randint(3,6), random.randint(4,6)), chess.PAWN, chess.WHITE)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=100, depth=4)

    def _g_queen_endgame(self, tk, dr):
        def setup(b):
            bk = chess.square(random.randint(2,6), random.randint(0,2))
            _p(b, bk, chess.KING, chess.BLACK)
            _p(b, chess.D1, chess.QUEEN, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE)
            _p(b, chess.square(random.randint(3,6), random.randint(3,5)), random.choice([chess.PAWN, chess.KNIGHT]), chess.BLACK)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    def _g_general(self, tk, dr):
        for _ in range(100):
            b = _eb()
            if not _place_kings(b, range(0,3), range(6,8)): continue
            nw = random.randint(2, 5); nb = random.randint(2, 5)
            wtypes = [chess.QUEEN] + [chess.ROOK]*2 + [chess.BISHOP]*2 + [chess.KNIGHT]*2
            btypes = [chess.QUEEN] + [chess.ROOK]*2 + [chess.BISHOP]*2 + [chess.KNIGHT]*2
            placed = {b.king(chess.WHITE), b.king(chess.BLACK)}
            for c, cnt, types in [(chess.WHITE, nw, wtypes), (chess.BLACK, nb, btypes)]:
                for _ in range(cnt * 4):
                    if cnt <= 0: break
                    sq = random.randint(0, 63)
                    if sq in placed: continue
                    ranks = range(1,7) if c == chess.WHITE else range(1,7)
                    if chess.square_rank(sq) not in ranks: continue
                    _p(b, sq, random.choice(types), c); placed.add(sq); cnt -= 1
            for _ in range(random.randint(2,6)):
                f = random.randint(0,7); r = random.randint(2,5)
                sq = chess.square(f, r)
                if sq not in placed: _p(b, sq, chess.PAWN, chess.WHITE); placed.add(sq)
            for _ in range(random.randint(2,6)):
                f = random.randint(0,7); r = random.randint(3,6)
                sq = chess.square(f, r)
                if sq not in placed: _p(b, sq, chess.PAWN, chess.BLACK); placed.add(sq)

            if not b.is_valid() or b.is_game_over(): continue
            ti = THEME_REGISTRY.get(tk)
            if not ti: continue
            is_mate_theme = ti.category == "Checkmate"
            m, s = _search_id(b, 3 if not is_mate_theme else 2)
            if not m: continue
            if is_mate_theme and s < 9000: continue
            if not is_mate_theme and s < 150: continue
            
            # Apply uniqueness validation to general puzzles too
            if is_mate_theme:
                if not self._is_unique_mate(b, m, 2): continue
            else:
                if not self._is_unique_tactic(b, m, 150, 3): continue
                
            pv = _get_solution(b, m, 3)
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the best move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} opportunity.", category=ti.category)
        return None

# ═══════════════════════════════════════════════════════════════════════════════════
# PUZZLE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════════

class PuzzleSolver:
    def __init__(self):
        self.solution_moves=[]; self.current_move_index=0; self.board=None
        self.initial_fen=""; self.is_complete=False; self.is_failed=False; self.move_history=[]
    def load_puzzle(self, fen, solution_moves):
        self.initial_fen=fen; self.board=chess.Board(fen); self.solution_moves=list(solution_moves)
        self.current_move_index=0; self.is_complete=False; self.is_failed=False; self.move_history=[]
    def try_move(self, uci_move):
        if self.is_complete or self.is_failed: return False, "Puzzle finished."
        if not self.board: return False, "No puzzle loaded."
        if self.current_move_index >= len(self.solution_moves): return False, "No more moves."
        expected = self.solution_moves[self.current_move_index]; move = chess.Move.from_uci(uci_move)
        if not self.board.is_legal(move): return False, "Illegal move."
        if uci_move == expected:
            self.board.push(move); self.move_history.append((uci_move, True)); self.current_move_index += 1
            if self.current_move_index >= len(self.solution_moves):
                self.is_complete = True
                return True, "Correct! Checkmate!" if self.board.is_checkmate() else "Correct! Solved!"
            else:
                opp = self.solution_moves[self.current_move_index]; om = chess.Move.from_uci(opp)
                if self.board.is_legal(om):
                    self.board.push(om); self.move_history.append((opp, True)); self.current_move_index += 1
                    if self.current_move_index >= len(self.solution_moves):
                        self.is_complete = True
                        return True, "Correct! Checkmate!" if self.board.is_checkmate() else "Correct! Solved!"
                    return True, "Correct! Find the next move."
                self.current_move_index += 1; return True, "Correct!"
        else:
            self.move_history.append((uci_move, False)); self.is_failed = True
            return False, f"Incorrect. Best was {self._mdesc(expected)}."
    def get_hint(self):
        if self.is_complete or self.is_failed or self.current_move_index >= len(self.solution_moves): return None
        e = self.solution_moves[self.current_move_index]
        return f"Best move starts from {chess.square_name(chess.Move.from_uci(e).from_square)}."
    def get_hint_square(self):
        if self.is_complete or self.is_failed or self.current_move_index >= len(self.solution_moves): return None
        return chess.Move.from_uci(self.solution_moves[self.current_move_index]).from_square
    def undo_last_try(self):
        if self.board:
            self.board=chess.Board(self.initial_fen); self.current_move_index=0
            self.is_failed=False; self.is_complete=False; self.move_history=[]
    def get_solution_display(self):
        tb=chess.Board(self.initial_fen); moves=[]
        for u in self.solution_moves:
            try: m=chess.Move.from_uci(u); moves.append(tb.san(m))
            except: moves.append(u)
            if tb.is_legal(m): tb.push(m)
        return "  ".join(moves)
    def _mdesc(self, uci):
        m=chess.Move.from_uci(uci); s=f"{chess.square_name(m.from_square)}{chess.square_name(m.to_square)}"
        if m.promotion: s+=f"={chess.piece_name(m.promotion)[0].upper()}"
        return s

# ═══════════════════════════════════════════════════════════════════════════════════
# CHESS BOARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════════════

PIECE_UNICODE = {
    chess.PAWN:{chess.WHITE:"\u2659",chess.BLACK:"\u265F"}, chess.KNIGHT:{chess.WHITE:"\u2658",chess.BLACK:"\u265E"},
    chess.BISHOP:{chess.WHITE:"\u2657",chess.BLACK:"\u265D"}, chess.ROOK:{chess.WHITE:"\u2656",chess.BLACK:"\u265C"},
    chess.QUEEN:{chess.WHITE:"\u2655",chess.BLACK:"\u265B"}, chess.KING:{chess.WHITE:"\u2654",chess.BLACK:"\u265A"},
}
LIGHT_SQ=QColor(240,217,181); DARK_SQ=QColor(181,136,99); HIGHLIGHT_CLR=QColor(255,255,100,130)
MOVE_DOT=QColor(0,0,0,60); CAPTURE_CLR=QColor(220,50,50,120); LAST_MOVE_CLR=QColor(155,199,0,100)
CHECK_CLR=QColor(255,0,0,150); CORRECT_CLR=QColor(100,200,100,130); WRONG_CLR=QColor(200,50,50,130)
HINT_CLR=QColor(100,149,237,150)

class ChessBoardWidget(QWidget):
    move_made = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board=chess.Board(); self.square_size=72; self.margin=28; self.flipped=False
        self.highlighted_squares=set(); self.last_move=None; self.check_square=None
        self.correct_square=None; self.wrong_square=None; self.hint_square=None
        self.selected_square=None; self.legal_moves_for_selected=set()
        self._dragging=False; self._drag_piece=None; self._drag_pos=QPoint(); self._drag_from=None
        self._piece_font=None; self._label_font=None
        self.setMinimumSize(self.square_size*8+self.margin*2, self.square_size*8+self.margin*2)
        self.setMouseTracking(True)
    def set_board(self,b): self.board=b.copy(); self.clear_highlights(); self.selected_square=None; self.legal_moves_for_selected=set(); self.update()
    def set_position(self,fen): self.board=chess.Board(fen); self.clear_highlights(); self.selected_square=None; self.legal_moves_for_selected=set(); self.update()
    def clear_highlights(self): self.highlighted_squares.clear(); self.last_move=self.check_square=self.correct_square=self.wrong_square=self.hint_square=None
    def highlight_last_move(self,f,t): self.last_move=(f,t); self.update()
    def highlight_check(self,s): self.check_square=s; self.update()
    def highlight_correct(self,s): self.correct_square=s; self.update()
    def highlight_wrong(self,s): self.wrong_square=s; self.update()
    def highlight_hint(self,s): self.hint_square=s; self.update()
    def _fonts(self):
        if not self._piece_font or self._piece_font.pixelSize()!=int(self.square_size*0.75):
            self._piece_font=QFont("Segoe UI Symbol",int(self.square_size*0.7)); self._piece_font.setBold(True)
        if not self._label_font: self._label_font=QFont("Arial",max(8,self.margin//2))
        return self._piece_font, self._label_font
    def _sq2px(self,sq):
        f,r=chess.square_file(sq),chess.square_rank(sq)
        if self.flipped: f,r=7-f,7-r
        return QPoint(self.margin+f*self.square_size, self.margin+(7-r)*self.square_size)
    def _px2sq(self,pos):
        f=(pos.x()-self.margin)//self.square_size; r=7-(pos.y()-self.margin)//self.square_size
        if self.flipped: f,r=7-f,7-r
        return chess.square(f,r) if 0<=f<=7 and 0<=r<=7 else None
    def sizeHint(self): return QSize(self.square_size*8+self.margin*2, self.square_size*8+self.margin*2)
    def resizeEvent(self,e): self.square_size=max(40,(min(self.width(),self.height())-self.margin*2)//8); self._piece_font=None; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); pf,lf=self._fonts()
        p.fillRect(self.rect(),QColor(48,46,43))
        for sq in chess.SQUARES: self._draw_sq(p,sq)
        p.setFont(lf); p.setPen(QColor(200,200,200))
        for i in range(8):
            fl=chess.FILE_NAMES[7-i if self.flipped else i]; rl=chess.RANK_NAMES[i if self.flipped else 7-i]
            p.drawText(self.margin+i*self.square_size+self.square_size-4, self.margin+8*self.square_size-4, fl)
            p.drawText(self.margin-self.margin+4, self.margin+i*self.square_size+self.square_size-6, rl)
        for sq in chess.SQUARES:
            pc=self.board.piece_at(sq)
            if pc and sq!=self._drag_from: self._draw_pc(p,self._sq2px(sq),pc,pf)
        if self._dragging and self._drag_piece: self._draw_pc(p,self._drag_pos,self._drag_piece,pf)
    def _draw_sq(self,p,sq):
        pos=self._sq2px(sq); f,r=chess.square_file(sq),chess.square_rank(sq)
        c=LIGHT_SQ if (f+r)%2==0 else DARK_SQ
        if sq==self.correct_square: c=CORRECT_CLR
        elif sq==self.wrong_square: c=WRONG_CLR
        elif sq==self.hint_square: c=HINT_CLR
        elif self.check_square is not None and sq==self.check_square: c=CHECK_CLR
        elif self.last_move and sq in self.last_move: c=LAST_MOVE_CLR
        elif sq in self.highlighted_squares: c=HIGHLIGHT_CLR
        p.fillRect(QRect(pos.x(),pos.y(),self.square_size,self.square_size),c)
        if sq in self.legal_moves_for_selected:
            ctr=pos+QPoint(self.square_size//2,self.square_size//2)
            if self.board.piece_at(sq):
                p.setPen(QPen(CAPTURE_CLR,3)); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(ctr,self.square_size//2-4,self.square_size//2-4)
            else:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(MOVE_DOT)
                p.drawEllipse(ctr,self.square_size//6,self.square_size//6)
    def _draw_pc(self,p,pos,piece,font):
        t=PIECE_UNICODE.get(piece.piece_type,{}).get(piece.color,"?")
        if t=="?": return
        p.setFont(font); p.setPen(QColor(0,0,0,60))
        p.drawText(pos.x()+2,pos.y()+2,self.square_size,self.square_size,Qt.AlignmentFlag.AlignCenter,t)
        p.setPen(QColor(255,255,255) if piece.color==chess.WHITE else QColor(30,30,30))
        p.drawText(pos.x(),pos.y(),self.square_size,self.square_size,Qt.AlignmentFlag.AlignCenter,t)
    def _try_promo(self,frm,to):
        m=chess.Move(frm,to)
        if m in self.board.legal_moves: return m
        for pt in [chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT]:
            pm=chess.Move(frm,to,promotion=pt)
            if pm in self.board.legal_moves: return pm
        return m
    def mousePressEvent(self,e):
        if e.button()!=Qt.MouseButton.LeftButton: return
        sq=self._px2sq(e.pos())
        if sq is None: return
        pc=self.board.piece_at(sq)
        if pc and pc.color==self.board.turn:
            self.selected_square=sq; self._drag_from=sq; self._drag_piece=pc
            self.legal_moves_for_selected={m.to_square for m in self.board.legal_moves if m.from_square==sq}
            self._dragging=True; self._drag_pos=e.pos(); self.update()
        elif self.selected_square is not None:
            m=self._try_promo(self.selected_square,sq)
            if m in self.board.legal_moves: self.move_made.emit(m.uci())
            self.selected_square=None; self.legal_moves_for_selected=set(); self.update()
    def mouseMoveEvent(self,e):
        if self._dragging: self._drag_pos=e.pos(); self.update()
    def mouseReleaseEvent(self,e):
        if not self._dragging: return
        sq=self._px2sq(e.pos()); self._dragging=False
        if sq is not None and sq!=self._drag_from:
            m=self._try_promo(self._drag_from,sq)
            if m in self.board.legal_moves: self.move_made.emit(m.uci())
        self.selected_square=None; self._drag_from=None; self._drag_piece=None
        self.legal_moves_for_selected=set(); self.update()

# ═══════════════════════════════════════════════════════════════════════════════════
# THEME FILTER PANEL
# ═══════════════════════════════════════════════════════════════════════════════════════════════

class ThemeFilterPanel(QScrollArea):
    theme_changed = Signal()
    def __init__(self,parent=None):
        super().__init__(parent); self._tc={}; self._cc={}
        self.setWidgetResizable(True); self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMaximumWidth(280); self.setMinimumWidth(220); self._build()
    def _build(self):
        c=QWidget(); l=QVBoxLayout(c); l.setSpacing(8); l.setContentsMargins(8,8,8,8)
        cg=QGroupBox("Categories"); cl=QVBoxLayout(cg); cl.setSpacing(4)
        for cat in CATEGORIES:
            cb=QCheckBox(cat); cb.setChecked(True); cb.stateChanged.connect(self._oc); self._cc[cat]=cb; cl.addWidget(cb)
        br=QHBoxLayout(); b1=QPushButton("Select All"); b1.setFixedHeight(28); b1.clicked.connect(lambda:self._sa(True))
        b2=QPushButton("Deselect All"); b2.setFixedHeight(28); b2.clicked.connect(lambda:self._sa(False))
        br.addWidget(b1); br.addWidget(b2); cl.addLayout(br); l.addWidget(cg)
        tg=QGroupBox("Themes"); tl=QVBoxLayout(tg); tl.setSpacing(2)
        for cat in CATEGORIES:
            tl.addWidget(QLabel(f"<b style='color:#89b4fa;'>{cat}</b>"))
            for t in get_themes_by_category(cat):
                cb=QCheckBox(f"  {t.icon_hint}  {t.display_name}"); cb.setChecked(True)
                cb.stateChanged.connect(self._ot); cb.setToolTip(t.description)
                self._tc[t.name]=(cb,t.category); tl.addWidget(cb)
        l.addWidget(tg); l.addStretch(); self.setWidget(c)
    def _oc(self):
        s=self.sender()
        for cat,cb in self._cc.items():
            if cb==s:
                v=cb.isChecked()
                for tk,(tcb,tc) in self._tc.items():
                    if tc==cat: tcb.blockSignals(True); tcb.setChecked(v); tcb.blockSignals(False)
                break
        self.theme_changed.emit()
    def _ot(self): self.theme_changed.emit()
    def _sa(self,v):
        for cb in self._cc.values(): cb.blockSignals(True); cb.setChecked(v); cb.blockSignals(False)
        for _,(cb,_) in self._tc.items(): cb.blockSignals(True); cb.setChecked(v); cb.blockSignals(False)
        self.theme_changed.emit()
    def get_selected_themes(self): return [tk for tk,(cb,_) in self._tc.items() if cb.isChecked()]

# ═══════════════════════════════════════════════════════════════════════════════════
# ASYNC WORKER
# ═══════════════════════════════════════════════════════════════════════════════════

class PuzzleGeneratorWorker(QThread):
    finished = Signal(list)
    
    def __init__(self, generator, difficulty_range, count, themes):
        super().__init__()
        self.generator = generator
        self.difficulty_range = difficulty_range
        self.count = count
        self.themes = themes

    def run(self):
        puzzles = self.generator.generate(difficulty_range=self.difficulty_range, count=self.count)
        valid_puzzles = [p for p in puzzles if p.theme_key in self.themes]
        self.finished.emit(valid_puzzles)

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════════════════════

DARK_STYLESHEET = """
QMainWindow{background-color:#1e1e2e}QWidget{background-color:#1e1e2e;color:#cdd6f4;font-family:'Segoe UI','Arial',sans-serif;font-size:13px}
QGroupBox{border:1px solid #45475a;border-radius:8px;margin-top:12px;padding-top:14px;font-weight:bold;font-size:13px;color:#cdd6f4;background-color:#181825}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px;color:#89b4fa}
QPushButton{background-color:#45475a;border:1px solid #585b70;border-radius:6px;padding:8px 16px;color:#cdd6f4;font-weight:bold;font-size:13px;min-height:20px}
QPushButton:hover{background-color:#585b70;border-color:#89b4fa}QPushButton:pressed{background-color:#313244}
QPushButton:disabled{background-color:#313244;color:#6c7086;border-color:#45475a}
QPushButton#primaryBtn{background-color:#89b4fa;color:#1e1e2e;border:none}QPushButton#primaryBtn:hover{background-color:#b4d0fb}
QPushButton#warningBtn{background-color:#f9e2af;color:#1e1e2e;border:none}QPushButton#warningBtn:hover{background-color:#fcecc7}
QPushButton#dangerBtn{background-color:#f38ba8;color:#1e1e2e;border:none}QPushButton#dangerBtn:hover{background-color:#f5b0c4}
QComboBox{background-color:#313244;border:1px solid #45475a;border-radius:6px;padding:6px 10px;color:#cdd6f4;min-height:20px}
QComboBox:hover{border-color:#89b4fa}QComboBox::drop-down{border:none;width:24px}QComboBox::down-arrow{image:none;border:none}
QComboBox QAbstractItemView{background-color:#313244;border:1px solid #45475a;color:#cdd6f4;selection-background-color:#45475a;selection-color:#89b4fa;border-radius:4px}
QSlider::groove:horizontal{border:1px solid #45475a;height:6px;background:#313244;border-radius:3px}
QSlider::handle:horizontal{background:#89b4fa;border:none;width:18px;height:18px;margin:-6px 0;border-radius:9px}
QSlider::handle:horizontal:hover{background:#b4d0fb}QSlider::sub-page:horizontal{background:#89b4fa;border-radius:3px}
QTextEdit,QTextBrowser{background-color:#181825;border:1px solid #45475a;border-radius:6px;color:#cdd6f4;padding:8px;font-size:12px}
QScrollArea{border:none}
QProgressBar{border:1px solid #45475a;border-radius:4px;text-align:center;background-color:#313244;color:#cdd6f4;font-size:11px;min-height:16px}
QProgressBar::chunk{background-color:#89b4fa;border-radius:3px}
QCheckBox{spacing:6px;color:#cdd6f4}QCheckBox::indicator{width:16px;height:16px;border:1px solid #45475a;border-radius:4px;background-color:#313244}
QCheckBox::indicator:checked{background-color:#89b4fa;border-color:#89b4fa}QCheckBox::indicator:hover{border-color:#89b4fa}
QLabel#statLabel{font-size:14px;font-weight:bold;color:#89b4fa}
QLabel#successMessage{color:#a6e3a1;background-color:rgba(166,227,161,0.1);border:1px solid rgba(166,227,161,0.3);padding:10px 14px;border-radius:6px;font-size:13px}
QLabel#errorMessage{color:#f38ba8;background-color:rgba(243,139,168,0.1);border:1px solid rgba(243,139,168,0.3);padding:10px 14px;border-radius:6px;font-size:13px}
QLabel#hintMessage{color:#f9e2af;background-color:rgba(249,226,175,0.1);border:1px solid rgba(249,226,175,0.3);padding:10px 14px;border-radius:6px;font-size:13px}
QToolBar{background-color:#181825;border-bottom:1px solid #313244;spacing:6px;padding:4px}
QStatusBar{background-color:#181825;color:#a6adc8;border-top:1px solid #313244;font-size:12px}
QMenu{background-color:#313244;border:1px solid #45475a;color:#cdd6f4;border-radius:6px;padding:4px}
QMenu::item{padding:6px 24px;border-radius:4px}QMenu::item:selected{background-color:#45475a}
QMenu::separator{height:1px;background:#45475a;margin:4px 8px}
QToolButton{background-color:transparent;border:1px solid transparent;border-radius:6px;padding:6px;color:#cdd6f4}
QToolButton:hover{background-color:#45475a;border-color:#585b70}
"""

class ChessPuzzleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Puzzle Solver — Procedural Generation"); self.setMinimumSize(1100,750); self.resize(1280,820)
        self.generator=ProceduralGenerator(); self.solver=PuzzleSolver()
        self.current_puzzle=None; self.puzzle_queue=[]; self.puzzle_index=0
        self.stats={"solved":0,"failed":0,"total":0}
        self.worker = None 
        self._auto_advance_timer = None
        
        # UI Timer Setup
        self.puzzle_timer = QTimer(self)
        self.puzzle_timer.setInterval(1000)
        self.puzzle_timer.timeout.connect(self._update_timer)
        self.elapsed_time = 0
        
        self._build_ui(); self._build_toolbar(); self._build_statusbar(); self._load_initial()
        
    def _build_ui(self):
        c=QWidget(); self.setCentralWidget(c); ml=QHBoxLayout(c); ml.setSpacing(0); ml.setContentsMargins(0,0,0,0)
        self.filter_panel=ThemeFilterPanel(); self.filter_panel.theme_changed.connect(self._on_filter); ml.addWidget(self.filter_panel)
        cw=QWidget(); cl=QVBoxLayout(cw); cl.setSpacing(8); cl.setContentsMargins(12,12,12,12)
        ib=QHBoxLayout()
        self.theme_label=QLabel("Theme: --"); self.theme_label.setObjectName("statLabel")
        self.diff_label=QLabel("Difficulty: --"); self.diff_label.setObjectName("statLabel")
        self.cat_label=QLabel("Category: --"); self.cat_label.setObjectName("statLabel")
        self.prog_label=QLabel("Puzzle 0/0")
        self.time_label=QLabel("⏱ 00:00"); self.time_label.setObjectName("statLabel")
        ib.addWidget(self.theme_label); ib.addSpacing(20); ib.addWidget(self.cat_label)
        ib.addSpacing(20); ib.addWidget(self.diff_label); ib.addStretch(); ib.addWidget(self.prog_label)
        ib.addSpacing(20); ib.addWidget(self.time_label)
        cl.addLayout(ib)
        self.pbar=QProgressBar(); self.pbar.setFixedHeight(6); self.pbar.setTextVisible(False); cl.addWidget(self.pbar)
        bf=QFrame(); bf.setStyleSheet("QFrame{background-color:#181825;border-radius:8px;padding:8px}")
        bl=QVBoxLayout(bf); bl.setContentsMargins(8,8,8,8)
        self.board=ChessBoardWidget(); self.board.move_made.connect(self._on_move)
        bl.addWidget(self.board,alignment=Qt.AlignmentFlag.AlignCenter); cl.addWidget(bf,stretch=1)
        self.msg=QLabel("Click 'New Puzzle' to generate procedural puzzles!"); self.msg.setWordWrap(True); self.msg.setMinimumHeight(36); cl.addWidget(self.msg)
        ctrls=QHBoxLayout()
        self.prev_btn=QPushButton("  ◀  Previous  "); self.prev_btn.clicked.connect(self._prev); self.prev_btn.setEnabled(False)
        self.next_btn=QPushButton("  Next  ▶  "); self.next_btn.clicked.connect(self._next); self.next_btn.setEnabled(False)
        self.new_btn=QPushButton("  New Puzzle  "); self.new_btn.setObjectName("primaryBtn"); self.new_btn.clicked.connect(self._new)
        self.hint_btn=QPushButton("  Hint  "); self.hint_btn.setObjectName("warningBtn"); self.hint_btn.clicked.connect(self._hint); self.hint_btn.setEnabled(False)
        self.reset_btn=QPushButton("  Reset  "); self.reset_btn.clicked.connect(self._reset); self.reset_btn.setEnabled(False)
        self.sol_btn=QPushButton("  Show Solution  "); self.sol_btn.setObjectName("dangerBtn"); self.sol_btn.clicked.connect(self._show_sol); self.sol_btn.setEnabled(False)
        for b in [self.prev_btn,self.reset_btn,self.hint_btn,self.new_btn,self.sol_btn,self.next_btn]: ctrls.addWidget(b)
        cl.addLayout(ctrls)
        dr=QHBoxLayout(); dr.addWidget(QLabel("Difficulty Range:"))
        self.dmin_l=QLabel("1"); self.dmin_l.setFixedWidth(16); self.dmax_l=QLabel("10"); self.dmax_l.setFixedWidth(20)
        self.ds1=QSlider(Qt.Orientation.Horizontal); self.ds1.setRange(1,10); self.ds1.setValue(1)
        self.ds1.setTickPosition(QSlider.TickPosition.TicksBelow); self.ds1.setTickInterval(1); self.ds1.valueChanged.connect(self._ddiff); self._dmin=1
        self.ds2=QSlider(Qt.Orientation.Horizontal); self.ds2.setRange(1,10); self.ds2.setValue(10)
        self.ds2.setTickPosition(QSlider.TickPosition.TicksBelow); self.ds2.setTickInterval(1); self.ds2.valueChanged.connect(self._ddiff); self._dmax=10
        dr.addWidget(self.dmin_l); dr.addWidget(self.ds1); dr.addWidget(QLabel("-")); dr.addWidget(self.ds2); dr.addWidget(self.dmax_l)
        cl.addLayout(dr); ml.addWidget(cw,stretch=1)
        rp=QWidget(); rp.setMaximumWidth(300); rp.setMinimumWidth(240); rl=QVBoxLayout(rp); rl.setSpacing(8); rl.setContentsMargins(8,12,8,12)
        dg=QGroupBox("Puzzle Info"); dl=QVBoxLayout(dg)
        self.desc=QTextEdit(); self.desc.setReadOnly(True); self.desc.setMaximumHeight(160); self.desc.setPlaceholderText("No puzzle loaded..."); dl.addWidget(self.desc)
        sg=QGroupBox("Statistics"); sl=QGridLayout(sg)
        self.sv_l=QLabel("0"); self.sv_l.setObjectName("statLabel"); self.fl=QLabel("0"); self.fl.setObjectName("statLabel")
        self.tl=QLabel("0"); self.tl.setObjectName("statLabel"); self.al=QLabel("--"); self.al.setObjectName("statLabel")
        for i,(t,w) in enumerate([("Solved:",self.sv_l),("Failed:",self.fl),("Total:",self.tl),("Accuracy:",self.al)]): sl.addWidget(QLabel(t),i,0); sl.addWidget(w,i,1)
        dl.addWidget(sg); rl.addWidget(dg)
        hg=QGroupBox("Move History"); hl=QVBoxLayout(hg)
        self.hist=QTextEdit(); self.hist.setReadOnly(True); self.hist.setMaximumHeight(180); self.hist.setPlaceholderText("Moves will appear here..."); hl.addWidget(self.hist)
        rl.addWidget(hg); rl.addStretch(); ml.addWidget(rp)
        
    def _build_toolbar(self):
        tb=QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        a=QAction("New Puzzle",self); a.setShortcut("Ctrl+N"); a.triggered.connect(self._new); tb.addAction(a); tb.addSeparator()
        a=QAction("Hint",self); a.setShortcut("Ctrl+H"); a.triggered.connect(self._hint); tb.addAction(a)
        a=QAction("Reset",self); a.setShortcut("Ctrl+R"); a.triggered.connect(self._reset); tb.addAction(a); tb.addSeparator()
        a=QAction("Random Puzzle",self); a.setShortcut("Ctrl+Shift+N"); a.triggered.connect(self._rand); tb.addAction(a); tb.addSeparator()
        a=QAction("Flip Board",self); a.setShortcut("Ctrl+F"); a.triggered.connect(self._flip); tb.addAction(a)
        
    def _build_statusbar(self): self.statusBar().showMessage("Ready. Puzzles generated algorithmically on demand!")
    
    def _ddiff(self):
        self._dmin=min(self.ds1.value(),self.ds2.value()); self._dmax=max(self.ds1.value(),self.ds2.value())
        self.dmin_l.setText(str(self._dmin)); self.dmax_l.setText(str(self._dmax))
        
    def _on_filter(self): pass
    
    def _load_initial(self): self._gen()
    
    def _update_timer(self):
        self.elapsed_time += 1
        mins = self.elapsed_time // 60
        secs = self.elapsed_time % 60
        self.time_label.setText(f"⏱ {mins:02d}:{secs:02d}")
        
    def _gen(self, cnt=20):
        themes=self.filter_panel.get_selected_themes()
        if not themes:
            self.msg.setText("⚠ No themes selected!"); self.msg.setObjectName("errorMessage"); self.msg.setStyleSheet(""); return
            
        self.msg.setText("⏳ Generating puzzles in background..."); self.msg.setStyleSheet("")
        self.new_btn.setEnabled(False)  
        
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        self.worker = PuzzleGeneratorWorker(self.generator, (self._dmin, self._dmax), cnt, themes)
        self.worker.finished.connect(self._on_gen_finished)
        self.worker.start()

    def _on_gen_finished(self, puzzle_queue):
        self.new_btn.setEnabled(True)
        if not puzzle_queue:
            self.msg.setText("⚠ No puzzles generated. Try broadening filters."); self.msg.setObjectName("errorMessage"); self.msg.setStyleSheet(""); return
        self.puzzle_queue = puzzle_queue
        self.puzzle_index=0
        self._load(0)
        
    def _load(self,idx):
        if 0<=idx<len(self.puzzle_queue):
            self.current_puzzle=self.puzzle_queue[idx]
            self.board.set_position(self.current_puzzle.fen)
            self.solver.load_puzzle(self.current_puzzle.fen,self.current_puzzle.moves)
            self.board.clear_highlights(); self._upd_info(); self._upd_nav()
            self._setmsg(f"Find the best move! Theme: {self.current_puzzle.theme_name}")
            self.hint_btn.setEnabled(True); self.reset_btn.setEnabled(True); self.sol_btn.setEnabled(True); self.hist.clear()
            self.statusBar().showMessage(f"Puzzle {idx+1}/{len(self.puzzle_queue)} | Theme: {self.current_puzzle.theme_name} | Difficulty: {self.current_puzzle.difficulty}/10 | Procedurally generated")
            
            # Reset and start timer
            self.elapsed_time = 0
            self.time_label.setText("⏱ 00:00")
            self.puzzle_timer.start()
            
    def _upd_info(self):
        if not self.current_puzzle: return
        p=self.current_puzzle; ti=THEME_REGISTRY.get(p.theme_key)
        self.theme_label.setText(f"Theme: {p.theme_name}")
        self.diff_label.setText(f"Difficulty: {FILLED_STAR*p.difficulty}{EMPTY_STAR*(10-p.difficulty)}")
        self.cat_label.setText(f"Category: {p.category}")
        self.prog_label.setText(f"Puzzle {self.puzzle_index+1}/{len(self.puzzle_queue)}")
        self.pbar.setMaximum(len(self.puzzle_queue)); self.pbar.setValue(self.puzzle_index+1)
        h=f"<h3 style='color:#89b4fa;margin:0 0 6px 0;'>{p.theme_name}</h3><p style='margin:2px 0;'>{p.description}</p>"
        if ti: h+=f"<p style='margin:2px 0;color:#a6adc8;'>{ti.description}</p>"
        h+=f"<p style='margin:6px 0 0 0;color:#a6adc8;font-size:11px;'>Difficulty: {p.difficulty}/10 | Category: {p.category}</p>"
        self.desc.setHtml(h)
        
    def _upd_nav(self): self.prev_btn.setEnabled(self.puzzle_index>0); self.next_btn.setEnabled(self.puzzle_index<len(self.puzzle_queue)-1)
    
    def _upd_stats(self):
        self.stats["total"]=self.stats["solved"]+self.stats["failed"]
        self.sv_l.setText(str(self.stats["solved"])); self.fl.setText(str(self.stats["failed"]))
        self.tl.setText(str(self.stats["total"]))
        self.al.setText(f"{(self.stats['solved']/self.stats['total']*100):.0f}%" if self.stats["total"] else "--")
        
    def _setmsg(self,t): self.msg.setText(t); self.msg.setObjectName(""); self.msg.setStyleSheet("")
    
    def _on_move(self,uci):
        if not self.current_puzzle: return
        ok,msg=self.solver.try_move(uci)
        if ok:
            self.board.set_board(self.solver.board); m=chess.Move.from_uci(uci)
            self.board.highlight_last_move(m.from_square,m.to_square)
            if self.solver.board.is_check():
                ks=self.solver.board.king(self.solver.board.turn)
                if ks is not None: self.board.highlight_check(ks)
            san=self.solver.board.san(chess.Move.from_uci(uci))
            self.hist.append(f"<span style='color:#a6e3a1;'>✓ {san}</span>")
            if self.solver.is_complete:
                self._setmsg("🎉 "+msg); self.msg.setObjectName("successMessage"); self.msg.setStyleSheet("")
                self.stats["solved"]+=1; self._upd_stats(); self.hint_btn.setEnabled(False)
                self.puzzle_timer.stop()
                # Auto-advance after 1.5 seconds if not the last puzzle
                if self.puzzle_index < len(self.puzzle_queue) - 1:
                    self._auto_advance_timer = QTimer.singleShot(1500, self._safe_next)
            else:
                self._setmsg(msg); self.msg.setObjectName("successMessage"); self.msg.setStyleSheet("")
        else:
            m=chess.Move.from_uci(uci); self.board.highlight_wrong(m.to_square)
            self._setmsg("❌ "+msg); self.msg.setObjectName("errorMessage"); self.msg.setStyleSheet("")
            self.stats["failed"]+=1; self._upd_stats(); self.hint_btn.setEnabled(False)
            self.puzzle_timer.stop()
            QTimer.singleShot(2000,self._reset)
        self.hist.verticalScrollBar().setValue(self.hist.verticalScrollBar().maximum())
        
    def _safe_next(self):
        """Ensures auto-advance only happens if the puzzle is still complete and hasn't been reset."""
        if self.solver.is_complete and not self.solver.is_failed:
            self._next()
        
    def _new(self): 
        self._gen(); self.stats={"solved":0,"failed":0,"total":0}; self._upd_stats()
    
    def _rand(self):
        p=self.generator.get_random_puzzle(difficulty_range=(self._dmin,self._dmax))
        if p:
            self.puzzle_queue=[p]; self.puzzle_index=0; self._load(0)
            self.statusBar().showMessage(f"Random: {p.theme_name} (Difficulty {p.difficulty}/10)")
            
    def _next(self):
        if self.puzzle_index < len(self.puzzle_queue) - 1:
            self.puzzle_index += 1
            self._load(self.puzzle_index)
        else:
            # Queue exhausted, generate more
            self.statusBar().showMessage("Queue finished. Generating new puzzles...")
            self._gen()
        
    def _prev(self):
        if self.puzzle_index>0: self.puzzle_index-=1; self._load(self.puzzle_index)
        
    def _reset(self):
        if self.current_puzzle:
            self.solver.undo_last_try(); self.board.set_board(self.solver.board); self.board.clear_highlights()
            self.hint_btn.setEnabled(True); self.hist.clear()
            self._setmsg("Puzzle reset. Try again!"); self.msg.setStyleSheet("")
            self.puzzle_timer.start() # Resume timer
            
    def _hint(self):
        if not self.current_puzzle or self.solver.is_complete or self.solver.is_failed: return
        h=self.solver.get_hint(); hs=self.solver.get_hint_square()
        if h and hs is not None: self.board.highlight_hint(hs)
        self._setmsg(f"💡 {h}  |  {self.current_puzzle.hint}")
        self.msg.setObjectName("hintMessage"); self.msg.setStyleSheet("")
        
    def _show_sol(self):
        if not self.current_puzzle: return
        sol=self.solver.get_solution_display()
        r=QMessageBox.question(self,"Show Solution",f"See solution? (Counts as failed)\n\n{sol}",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        if r==QMessageBox.StandardButton.Yes:
            self.stats["failed"]+=1; self._upd_stats(); self._play_sol(0)
            self.puzzle_timer.stop()
            
    def _play_sol(self,idx):
        if idx>=len(self.current_puzzle.moves): return
        uci=self.current_puzzle.moves[idx]; m=chess.Move.from_uci(uci)
        if self.solver.board.is_legal(m):
            self.solver.board.push(m); self.board.set_board(self.solver.board)
            self.board.highlight_last_move(m.from_square,m.to_square)
            self.hist.append(f"<span style='color:#89b4fa;'>→ {self.solver.board.san(m)}</span>")
            self.hist.verticalScrollBar().setValue(self.hist.verticalScrollBar().maximum())
            QTimer.singleShot(800,lambda:self._play_sol(idx+1))
            
    def _flip(self): self.board.flipped=not self.board.flipped; self.board.update()

def main():
    app=QApplication(sys.argv); app.setStyle("Fusion")
    pal=QPalette()
    pal.setColor(QPalette.ColorRole.Window,QColor(30,30,46)); pal.setColor(QPalette.ColorRole.WindowText,QColor(205,214,244))
    pal.setColor(QPalette.ColorRole.Base,QColor(24,24,37)); pal.setColor(QPalette.ColorRole.AlternateBase,QColor(30,30,46))
    pal.setColor(QPalette.ColorRole.ToolTipBase,QColor(49,50,68)); pal.setColor(QPalette.ColorRole.ToolTipText,QColor(205,214,244))
    pal.setColor(QPalette.ColorRole.Text,QColor(205,214,244)); pal.setColor(QPalette.ColorRole.Button,QColor(49,50,68))
    pal.setColor(QPalette.ColorRole.ButtonText,QColor(205,214,244)); pal.setColor(QPalette.ColorRole.Highlight,QColor(137,180,250))
    pal.setColor(QPalette.ColorRole.HighlightedText,QColor(30,30,46))
    app.setPalette(pal); app.setStyleSheet(DARK_STYLESHEET)
    w=ChessPuzzleApp(); w.show(); sys.exit(app.exec())

if __name__=="__main__": main()