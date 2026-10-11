#!/usr/bin/env python3
"""
Chess Puzzle Solver — Procedural Generation, YouTube Shorts Creator, & Numba Optimizations
Includes: Negamax + AB Pruning, MCTS, GA, Numba JIT Evaluation, Advanced Video Rendering, 
Analysis Mode, Smooth Preview Animation, Procedural Audio Synth, BGM Ducking, PGN Export, Settings Persistence.
"""

import sys, os, random, math, chess, subprocess, logging, json, wave
from contextlib import contextmanager
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    filename='chess_puzzle.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("=== Chess Puzzle Solver Logger Initialized ===")

try:
    import cv2
    VIDEO_EXPORT_ENABLED = True
    logger.info("Video export dependencies (cv2, numpy) loaded successfully.")
except ImportError:
    VIDEO_EXPORT_ENABLED = False
    logger.warning("Video export dependencies (cv2, numpy) not available.")

try:
    from numba import njit
    NUMBA_ENABLED = True
    logger.info("Numba JIT enabled.")
except ImportError:
    NUMBA_ENABLED = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    logger.warning("Numba not available — using fallback (slower).")

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTextEdit, QFrame, QSplitter,
    QScrollArea, QCheckBox, QGroupBox, QSizePolicy, QToolBar, QStatusBar,
    QProgressBar, QSpacerItem, QToolButton, QMenu, QMessageBox, QDialog, 
    QSpinBox, QFileDialog, QProgressDialog, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QPoint, QPointF, QRect, QRectF, QThread, QDateTime
from PySide6.QtGui import (
    QIcon, QFont, QColor, QPalette, QAction, QCursor, QPixmap, QPainter, QBrush, QPen,
    QImage, QPolygon, QPolygonF, QRadialGradient, QLinearGradient
)

# ═══════════════════════════════════════════════════════════════════════════════════
# THEME REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class ThemeInfo:
    name: str; display_name: str; category: str; base_difficulty: int; description: str; icon_hint: str

THEME_REGISTRY: Dict[str, ThemeInfo] = {
    "fork":             ThemeInfo("fork",             "Fork / Double Attack",      "Attack",     3, "Attack two or more pieces simultaneously with a single move.", "\u265e"),
    "pin":              ThemeInfo("pin",              "Pin",                       "Attack",     2, "Restrict an opponent's piece from moving because it would expose a more valuable piece behind it.", "\u2659"),
    "skewer":           ThemeInfo("skewer",           "Skewer",                    "Attack",     4, "Force a valuable piece to move, revealing a less valuable piece behind it.", "\u265c"),
    "discovered_attack":ThemeInfo("discovered_attack","Discovered Attack",         "Attack",     4, "Move one piece to reveal an attack by another piece behind it.", "\u2657"),
    "double_check":     ThemeInfo("double_check",     "Double Check",             "Attack",     6, "Deliver check with two pieces simultaneously, forcing the king to move.", "\u265b"),
    "xray_attack":      ThemeInfo("xray_attack",      "X-Ray Attack",             "Attack",     5, "Attack through an enemy piece to target a piece behind it.", "\u265c"),
    "windmill":         ThemeInfo("windmill",         "Windmill",                 "Attack",     8, "A discovered check pattern where the moving piece repeatedly checks and captures.", "\u265e"),
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
# NUMBA OPTIMIZED HELPERS & PSTs
# ═══════════════════════════════════════════════════════════════════════════════════

PAWN_PST = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0,  0,  0,  0,  0,  0,  0,  0
], dtype=np.int32)

KNIGHT_PST = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
], dtype=np.int32)

KING_MG_PST = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
], dtype=np.int32)

KING_EG_PST = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
], dtype=np.int32)

@njit(cache=True)
def _popcount_numba(x):
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
    h  = np.uint64(0x0101010101010101)
    
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return int((x * h) >> np.uint64(56))

@njit(cache=True)
def _eval_bitboards_numba(wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk):
    score = 0
    score += 100 * _popcount_numba(wp) - 100 * _popcount_numba(bp)
    score += 320 * _popcount_numba(wn) - 320 * _popcount_numba(bn)
    score += 330 * _popcount_numba(wb) - 330 * _popcount_numba(bb)
    score += 500 * _popcount_numba(wr) - 500 * _popcount_numba(br)
    score += 900 * _popcount_numba(wq) - 900 * _popcount_numba(bq)
    
    if _popcount_numba(wb) >= 2: score += 30
    if _popcount_numba(bb) >= 2: score -= 30
    
    phase = 24
    phase -= 4 * _popcount_numba(wq) + 4 * _popcount_numba(bq)
    phase -= 2 * _popcount_numba(wr) + 2 * _popcount_numba(br)
    phase -= _popcount_numba(wn) + _popcount_numba(bn)
    phase -= _popcount_numba(wb) + _popcount_numba(bb)
    phase = max(0, phase)
    mg = phase
    eg = 24 - phase
    
    for i in range(64):
        if (wp >> np.uint64(i)) & 1:
            score += 10 + PAWN_PST[i]
            f = i % 8; r = i // 8
            is_passed = True
            for j in range(64):
                if (bp >> np.uint64(j)) & 1:
                    bf = j % 8; br = j // 8
                    if abs(bf - f) <= 1 and br > r:
                        is_passed = False; break
            if is_passed: score += 10 * r
            
        if (bp >> np.uint64(i)) & 1:
            score -= 10 + PAWN_PST[63 - i]
            f = i % 8; r = i // 8
            is_passed = True
            for j in range(64):
                if (wp >> np.uint64(j)) & 1:
                    wf = j % 8; wr = j // 8
                    if abs(wf - f) <= 1 and wr < r:
                        is_passed = False; break
            if is_passed: score -= 10 * (7 - r)
            
        if (wn >> np.uint64(i)) & 1: score += KNIGHT_PST[i]
        if (bn >> np.uint64(i)) & 1: score -= KNIGHT_PST[63 - i]
        
        if (wk >> np.uint64(i)) & 1: score += (KING_MG_PST[i] * mg + KING_EG_PST[i] * eg) // 24
        if (bk >> np.uint64(i)) & 1: score -= (KING_MG_PST[63 - i] * mg + KING_EG_PST[63 - i] * eg) // 24
            
    return score

@njit(cache=True)
def _ga_crossover_mutate_numba(p1, p2, core_len, mutation_rate):
    child = np.empty((16, 3), dtype=np.int32)
    child_len = 0
    for i in range(core_len):
        child[i, 0] = p1[i, 0]; child[i, 1] = p1[i, 1]; child[i, 2] = p1[i, 2]
    child_len = core_len
    p1_fillers = p1[core_len:]
    p2_fillers = p2[core_len:]
    half1 = len(p1_fillers) // 2
    half2 = len(p2_fillers) // 2
    for i in range(half1):
        sq = p1_fillers[i, 0]
        valid = True
        for j in range(child_len):
            if child[j, 0] == sq: valid = False; break
        if valid:
            child[child_len, 0] = sq; child[child_len, 1] = p1_fillers[i, 1]; child[child_len, 2] = p1_fillers[i, 2]
            child_len += 1
    for i in range(half2, len(p2_fillers)):
        sq = p2_fillers[i, 0]
        valid = True
        for j in range(child_len):
            if child[j, 0] == sq: valid = False; break
        if valid:
            child[child_len, 0] = sq; child[child_len, 1] = p2_fillers[i, 1]; child[child_len, 2] = p2_fillers[i, 2]
            child_len += 1
    if np.random.rand() < mutation_rate:
        if np.random.rand() < 0.5 and child_len > core_len:
            rem_idx = np.random.randint(core_len, child_len)
            for i in range(rem_idx, child_len - 1):
                child[i, 0] = child[i+1, 0]; child[i, 1] = child[i+1, 1]; child[i, 2] = child[i+1, 2]
            child_len -= 1
        else:
            sq = np.random.randint(0, 64)
            valid = True
            for j in range(child_len):
                if child[j, 0] == sq: valid = False; break
            if valid:
                child[child_len, 0] = sq
                child[child_len, 1] = np.random.randint(1, 5)
                child[child_len, 2] = np.random.randint(0, 2)
                child_len += 1
    return child[:child_len].copy()

@njit(cache=True)
def _ga_random_individual_numba(core_pieces, max_fillers):
    core_len = len(core_pieces)
    pieces = np.empty((core_len + max_fillers, 3), dtype=np.int32)
    for i in range(core_len):
        pieces[i, 0] = core_pieces[i, 0]; pieces[i, 1] = core_pieces[i, 1]; pieces[i, 2] = core_pieces[i, 2]
    num_fillers = np.random.randint(0, max_fillers + 1)
    for _ in range(num_fillers):
        sq = np.random.randint(0, 64)
        valid = True
        for i in range(core_len):
            if pieces[i, 0] == sq: valid = False; break
        if valid:
            pieces[core_len, 0] = sq
            pieces[core_len, 1] = np.random.randint(1, 5)
            pieces[core_len, 2] = np.random.randint(0, 2)
            core_len += 1
    return pieces[:core_len].copy()

# ═══════════════════════════════════════════════════════════════════════════════════
# EXPERT SEARCH ENGINES
# ═══════════════════════════════════════════════════════════════════════════════════

PV = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
MATE_SCORE = 100000
TT = {}
KILLER_MOVES = [[None, None] for _ in range(64)]
HISTORY = {}
COUNTER_MOVES = {}

def _reset_search_state():
    global TT, KILLER_MOVES, HISTORY, COUNTER_MOVES
    TT.clear()
    KILLER_MOVES = [[None, None] for _ in range(64)]
    HISTORY = {}
    COUNTER_MOVES = {}

def _eval_negamax(board):
    if board.is_checkmate(): return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return 0
    if NUMBA_ENABLED:
        w_occ = board.occupied_co[chess.WHITE]
        b_occ = board.occupied_co[chess.BLACK]
        s = _eval_bitboards_numba(
            np.uint64(board.pawns & w_occ), np.uint64(board.knights & w_occ), np.uint64(board.bishops & w_occ),
            np.uint64(board.rooks & w_occ), np.uint64(board.queens & w_occ), np.uint64(board.kings & w_occ),
            np.uint64(board.pawns & b_occ), np.uint64(board.knights & b_occ), np.uint64(board.bishops & b_occ),
            np.uint64(board.rooks & b_occ), np.uint64(board.queens & b_occ), np.uint64(board.kings & b_occ)
        )
    else:
        s = 0
        for sq, p in board.piece_map().items():
            v = PV[p.piece_type]
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                f, r = chess.square_file(sq), chess.square_rank(sq)
                v += max(0, int((7 - abs(3.5 - f) - abs(3.5 - r)) * 5))
            s += v if p.color == board.turn else -v
    return s if board.turn == chess.WHITE else -s

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
        if not board.is_check():
            victim = board.piece_type_at(move.to_square)
            v_val = PV.get(victim, 100) if victim else 0
            if move.promotion: v_val += PV.get(move.promotion, 0)
            if stand_pat + v_val + 200 < alpha:
                continue
                
        board.push(move)
        score = -_quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta: return beta
        if score > alpha: alpha = score
    return alpha

def _search(board, depth, alpha, beta, ply=0, last_move=None):
    if board.is_checkmate(): return None, -MATE_SCORE - depth, None, -MATE_SCORE - depth
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return None, 0, None, 0
    
    if board.is_check(): depth += 1
    if depth == 0: 
        q = _quiesce(board, alpha, beta)
        return None, q, None, q
    
    if depth >= 3 and not board.is_check() and ply > 0 and board.has_legal_moves():
        board.push(chess.Move.null())
        _, sc, _, _ = _search(board, depth - 3, -beta, -beta + 1, ply + 1)
        sc = -sc
        board.pop()
        if sc >= beta:
            return None, beta, None, beta
            
    key = board._transposition_key()
    alpha_orig = alpha
    tt_move = None
    if key in TT:
        tt_depth, tt_score, tt_flag, tt_move = TT[key]
        if tt_depth >= depth:
            if tt_flag == 0: return tt_move, tt_score, tt_move, tt_score
            elif tt_flag == 1: alpha = max(alpha, tt_score)
            elif tt_flag == 2: beta = min(beta, tt_score)
            if alpha >= beta: return tt_move, tt_score, tt_move, tt_score
            
    static_eval = _eval_negamax(board) if not board.is_check() else 0
    
    if depth == 1 and not board.is_check():
        if static_eval + 200 <= alpha:
            q = _quiesce(board, alpha, beta)
            return None, q, None, q
            
    moves = list(board.legal_moves)
    def _ms(m):
        s = 0
        if m == tt_move: s += 1000000
        if m.promotion: s += 100000 + PV.get(m.promotion, 0)
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square)
            if victim is None: victim = chess.PAWN
            attacker = board.piece_type_at(m.from_square)
            s += 10000 + (PV.get(victim, 0) * 10) - PV.get(attacker, 0)
        if board.gives_check(m): s += 5000
        if depth < len(KILLER_MOVES) and m in KILLER_MOVES[depth]: s += 9000
        if last_move and COUNTER_MOVES.get((last_move.from_square, last_move.to_square)) == m: s += 8000
        s += HISTORY.get((board.turn, m.from_square, m.to_square), 0)
        return -s
    moves.sort(key=_ms)
    
    bm, bs = None, -float('inf')
    bm2, bs2 = None, -float('inf')
    
    for i, m in enumerate(moves):
        is_cap = board.is_capture(m)
        gives_check = board.gives_check(m)
        
        if depth <= 3 and not board.is_check() and not is_cap and not m.promotion and not gives_check:
            if static_eval + depth * 150 <= alpha:
                continue
            if i > 3 + depth * depth:
                continue
                
        board.push(m)
        sc = 0
        
        if i == 0:
            _, sc, _, _ = _search(board, depth - 1, -beta, -alpha, ply + 1, last_move=m)
            sc = -sc
        else:
            reduction = 0
            if depth > 2 and i > 3 and not is_cap and not m.promotion and not gives_check:
                reduction = 1
                if depth < len(KILLER_MOVES) and m in KILLER_MOVES[depth]:
                    reduction = 0
            
            _, sc, _, _ = _search(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, last_move=m)
            sc = -sc
            
            if sc > alpha and sc < beta:
                _, sc, _, _ = _search(board, depth - 1, -beta, -alpha, ply + 1, last_move=m)
                sc = -sc
                
        board.pop()
        
        if sc > bs:
            bs2, bm2 = bs, bm
            bs, bm = sc, m
        elif sc > bs2:
            bs2, bm2 = sc, m
            
        if bs > alpha: alpha = bs
        if alpha >= beta:
            if not is_cap:
                if depth < len(KILLER_MOVES):
                    if KILLER_MOVES[depth][0] != m:
                        KILLER_MOVES[depth][1] = KILLER_MOVES[depth][0]
                        KILLER_MOVES[depth][0] = m
                HISTORY[(board.turn, m.from_square, m.to_square)] = HISTORY.get((board.turn, m.from_square, m.to_square), 0) + depth * depth
                if last_move:
                    COUNTER_MOVES[(last_move.from_square, last_move.to_square)] = m
            break
            
    flag = 0 if bs <= alpha_orig else (2 if bs >= beta else 1)
    TT[key] = (depth, bs, flag, bm)
    return bm, bs, bm2, bs2

def _search_id(board, max_depth=3):
    bm, bs, bm2, bs2 = None, 0, None, 0
    for d in range(1, max_depth + 1):
        if d >= 3:
            alpha, beta = bs - 50, bs + 50
            bm, bs, bm2, bs2 = _search(board, d, alpha, beta, 0)
            if bs <= alpha or bs >= beta:
                bm, bs, bm2, bs2 = _search(board, d, -float('inf'), float('inf'), 0)
        else:
            bm, bs, bm2, bs2 = _search(board, d, -float('inf'), float('inf'), 0)
            
        if bs > MATE_SCORE - 100: break
    return bm, bs, bm2, bs2

def _get_solution(board, first_move, max_depth=4):
    moves = [first_move.uci()]
    b = board.copy(); b.push(first_move)
    for _ in range(max_depth):
        if b.is_game_over(): break
        m, _, _, _ = _search_id(b, 2)
        if m is None: break
        moves.append(m.uci()); b.push(m)
    return moves

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board; self.parent = parent; self.move = move
        self.children = []; self.visits = 0; self.wins = 0.0
        self.untried_moves = list(board.legal_moves)
    def is_fully_expanded(self): return len(self.untried_moves) == 0
    def best_uct_child(self, c_param=1.414):
        log_n = math.log(self.visits) if self.visits > 0 else 0
        return max(self.children, key=lambda c: (c.wins / c.visits) + c_param * math.sqrt(log_n / c.visits))

def _mcts_search(board, iterations=800):
    root = MCTSNode(board.copy())
    for _ in range(iterations):
        node = root; sim_board = board.copy()
        while node.is_fully_expanded() and node.children:
            node = node.best_uct_child(); sim_board.push(node.move)
        if not sim_board.is_game_over() and node.untried_moves:
            move = random.choice(node.untried_moves); node.untried_moves.remove(move)
            sim_board.push(move)
            child_node = MCTSNode(sim_board.copy(), parent=node, move=move)
            node.children.append(child_node); node = child_node
            
        score = 0
        if sim_board.is_checkmate(): 
            score = 1 if sim_board.turn != board.turn else -1
        else:
            curr = sim_board.copy()
            for _ in range(8):
                if curr.is_game_over(): break
                moves = list(curr.legal_moves)
                best_ev = -float('inf')
                best_move = random.choice(moves)
                for m in moves:
                    curr.push(m)
                    ev = _eval_negamax(curr)
                    if curr.turn == chess.BLACK: ev = -ev
                    if ev > best_ev:
                        best_ev = ev
                        best_move = m
                    curr.pop()
                curr.push(best_move)
                
            if curr.is_checkmate(): 
                score = 1 if curr.turn != board.turn else -1
            else:
                ev = _eval_negamax(curr)
                score = max(-1.0, min(1.0, ev / 2000.0))
                if board.turn != curr.turn: score = -score
                
        while node is not None:
            node.visits += 1; node.wins += score; score = -score; node = node.parent
            
    if not root.children: return None, 0
    best = max(root.children, key=lambda c: c.visits)
    return best.move, best.wins / best.visits

@dataclass
class Individual:
    pieces: List[Tuple[int, int, int]]
    fitness: float = 0.0

def _ga_optimize_filler(core_pieces, eval_fn, population_size=15, generations=5):
    if NUMBA_ENABLED:
        core_arr = np.array(core_pieces, dtype=np.int32)
        core_len = len(core_arr)
        population = [_ga_random_individual_numba(core_arr, 4) for _ in range(population_size)]
        fitnesses = [0.0] * population_size
        for gen in range(generations):
            for i in range(population_size):
                b = _eb()
                ind = population[i]
                for row in range(ind.shape[0]):
                    _p(b, int(ind[row, 0]), int(ind[row, 1]), int(ind[row, 2]))
                fitnesses[i] = eval_fn(b)
            indices = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
            next_gen = [population[i] for i in indices[:max(2, population_size // 5)]]
            next_fit = [fitnesses[i] for i in indices[:max(2, population_size // 5)]]
            while len(next_gen) < population_size:
                p1_idx = indices[np.random.randint(0, max(2, population_size//2))]
                p2_idx = indices[np.random.randint(0, max(2, population_size//2))]
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                child = _ga_crossover_mutate_numba(p1, p2, core_len, 0.4)
                next_gen.append(child)
                next_fit.append(0.0)
            population = next_gen
            fitnesses = next_fit
        for i in range(population_size):
            b = _eb()
            ind = population[i]
            for row in range(ind.shape[0]):
                _p(b, int(ind[row, 0]), int(ind[row, 1]), int(ind[row, 2]))
            fitnesses[i] = eval_fn(b)
        best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        best_ind = population[best_idx]
        return [(int(best_ind[i, 0]), int(best_ind[i, 1]), int(best_ind[i, 2])) for i in range(best_ind.shape[0])]
    else:
        def random_individual():
            pieces = list(core_pieces); num_fillers = random.randint(0, 4)
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
                core_set = set(core_pieces); child_pieces = list(core_set)
                p1_fillers = [p for p in p1.pieces if p not in core_set]
                p2_fillers = [p for p in p2.pieces if p not in core_set]
                for p in p1_fillers[:len(p1_fillers)//2] + p2_fillers[len(p2_fillers)//2:]:
                    if not any(cp[0] == p[0] for cp in child_pieces): child_pieces.append(p)
                if random.random() < 0.4:
                    child_fillers = [p for p in child_pieces if p not in core_set]
                    if child_fillers and random.random() < 0.5: child_pieces.remove(random.choice(child_fillers))
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

def _eb():
    b = chess.Board.empty(); b.turn = chess.WHITE; b.castling_rights = chess.BB_EMPTY; b.ep_square = None
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

@dataclass
class Puzzle:
    fen: str; theme_key: str; theme_name: str; difficulty: int
    moves: List[str]; description: str; hint: str; category: str

class ProceduralGenerator:
    def generate(self, theme_key=None, category=None, theme_keys=None, difficulty=None, difficulty_range=(1,10), count=10, avoid_fens=None, progress_callback=None, cancel_check=None):
        _reset_search_state()
        
        if theme_keys:
            themes = theme_keys
        elif theme_key:
            themes = [theme_key]
        elif category:
            themes = [t.name for t in THEME_REGISTRY.values() if t.category == category]
        else:
            themes = list(THEME_REGISTRY.keys())
            
        if not themes:
            return []
            
        puzzles = []
        max_attempts = count * 100
        attempts = 0
        
        while len(puzzles) < count and attempts < max_attempts:
            if cancel_check and cancel_check():
                logger.info("Generation cancelled by user.")
                break
                
            attempts += 1
            tk = random.choice(themes)
            
            gen = getattr(self, f"_g_{tk}", None)
            if not gen: gen = self._g_general
            
            try:
                p = gen(tk, difficulty_range)
                if p:
                    if difficulty and p.difficulty != difficulty: 
                        continue
                    if avoid_fens and p.fen in avoid_fens:
                        continue
                    puzzles.append(p)
                    if avoid_fens is not None:
                        avoid_fens.add(p.fen)
                    logger.info(f"Generated puzzle {len(puzzles)}/{count}: theme={p.theme_name} (attempt {attempts})")
                    if progress_callback:
                        progress_callback(len(puzzles), count, f"Generated {len(puzzles)}/{count} puzzles...")
            except Exception as e:
                logger.error(f"Error generating puzzle {tk} (attempt {attempts}): {e}")
                
        random.shuffle(puzzles)
        return puzzles[:count]

    def get_random_puzzle(self, difficulty_range=(1,10)):
        tk = random.choice(list(THEME_REGISTRY.keys()))
        p = self.generate(theme_key=tk, difficulty_range=difficulty_range, count=1)
        return p[0] if p else None

    def _try_mate(self, tk, dr, setup_fn, check_fn=None, depth=2):
        for _ in range(150):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, s, m2, s2 = _search_id(b, depth)
            if not m or s < 9000: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            if m2 and s2 > 9000 - 100: continue 
            pv = _get_solution(b, m, 3); ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_mate_mcts(self, tk, dr, setup_fn, check_fn=None, iterations=600):
        for _ in range(80):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, wr = _mcts_search(b, iterations)
            if not m: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            m_search, s_search, m2, s2 = _search_id(b, 2)
            if not m_search or s_search < 9000: continue
            if m2 and s2 > 9000 - 100: continue
            pv = _get_solution(b, m_search, 3); ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle (MCTS-Verified).",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_mate_ga(self, tk, dr, core_setup_fn, check_fn=None, depth=2):
        for _ in range(30):
            core_pieces = core_setup_fn()
            if not core_pieces: continue
            def eval_fn(board):
                if not board.is_valid() or board.is_game_over(): return -1000
                m, s, m2, s2 = _search_id(board, depth)
                if not m or s < 9000: return -1000
                tb = board.copy(); tb.push(m)
                if not tb.is_checkmate(): return -1000
                if check_fn and not check_fn(board, m): return -1000
                fitness = 1000 - (len(board.piece_map()) * 15)
                if m2 and s2 > 9000 - 100: fitness -= 500
                else: fitness += 500
                return fitness
            best_pieces = _ga_optimize_filler(core_pieces, eval_fn, 15, 4)
            b = _eb()
            for sq, pt, c in best_pieces: _p(b, sq, pt, c)
            if not b.is_valid() or b.is_game_over(): continue
            m, s, m2, s2 = _search_id(b, depth)
            if not m or s < 9000: continue
            tb = b.copy(); tb.push(m)
            if not tb.is_checkmate(): continue
            if check_fn and not check_fn(b, m): continue
            if m2 and s2 > 9000 - 100: continue
            pv = _get_solution(b, m, 3); ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the winning move in this {ti.display_name} puzzle (GA-Optimized).",
                hint=f"Look for a {ti.display_name.lower()} pattern.", category=ti.category)
        return None

    def _try_tactic(self, tk, dr, setup_fn, check_fn=None, min_adv=200, depth=3):
        for _ in range(150):
            b = _eb()
            if not setup_fn(b): continue
            if not b.is_valid() or b.is_game_over(): continue
            m, s, m2, s2 = _search_id(b, depth)
            if not m or s < min_adv: continue
            if check_fn and not check_fn(b, m): continue
            if m2 and s2 > s - (min_adv // 2): continue
            pv = _get_solution(b, m, 4); ti = THEME_REGISTRY[tk]
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the best tactical move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} opportunity.", category=ti.category)
        return None

    def _g_back_rank_mate(self, tk, dr):
        def setup(b):
            kf = random.randint(4, 7); bk_sq = chess.square(kf, 7)
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
            else: return False
            _add_filler(b, random.randint(0,1), random.randint(1,3), {b.king(chess.WHITE), b.king(chess.BLACK)})
            return True
        return self._try_mate(tk, dr, setup, lambda b,m: chess.square_rank(m.to_square)==7, depth=2)

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
            _p(b, wk, chess.KING, chess.WHITE); _add_filler(b, 0, 0); return True
        return self._try_mate(tk, dr, setup, depth=3)

    def _g_arabian_mate(self, tk, dr):
        def setup(b):
            kf = random.choice([7, 6]); _p(b, chess.square(kf, 7), chess.KING, chess.BLACK)
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
            return [ (chess.H7, chess.KING, chess.BLACK), (chess.G7, chess.PAWN, chess.BLACK),
                     (chess.E7, chess.KNIGHT, chess.WHITE), (chess.G1, chess.ROOK, chess.WHITE),
                     (chess.B2, chess.KING, chess.WHITE) ]
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
            return [ (chess.G8, chess.KING, chess.BLACK), (chess.F7, chess.PAWN, chess.BLACK),
                     (chess.G7, chess.PAWN, chess.BLACK), (chess.D3, chess.BISHOP, chess.WHITE),
                     (chess.A1, chess.ROOK, chess.WHITE), (chess.A2, chess.KING, chess.WHITE) ]
        return self._try_mate_ga(tk, dr, core_setup, depth=2)

    def _g_morphy_mate(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK); _p(b, chess.F8, chess.ROOK, chess.BLACK)
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

    def _g_legal_mate(self, tk, dr): return self._g_smothered_mate(tk, dr)

    def _g_fork(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            nsq = random.choice([chess.C4, chess.D4, chess.D5, chess.E5, chess.E4, chess.F5, chess.C5, chess.F4])
            atks = _knight_atks(nsq)
            targets = random.sample([s for s in atks if s != chess.H8 and s != nsq], min(2, len(atks)-1))
            if len(targets) < 2: return False
            ttypes = random.sample([chess.ROOK, chess.QUEEN, chess.BISHOP, chess.ROOK], 2)
            for i, t in enumerate(targets): _p(b, t, ttypes[i], chess.BLACK)
            _p(b, nsq, chess.KNIGHT, chess.WHITE); _p(b, chess.A2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(0,2), random.randint(0,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=2)

    def _g_pin(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            pin_file = random.randint(1, 6)
            _p(b, chess.square(pin_file, 6), random.choice([chess.ROOK, chess.QUEEN, chess.BISHOP]), chess.BLACK)
            _p(b, chess.E1, chess.BISHOP, chess.WHITE); _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(1,3), random.randint(1,3)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_skewer(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.ROOK, chess.WHITE); mid = random.choice([chess.E6, chess.E7])
            _p(b, mid, chess.QUEEN, chess.BLACK); _p(b, chess.B2, chess.KING, chess.BLACK); return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    def _g_discovered_attack(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.BISHOP, chess.WHITE); _p(b, chess.E5, chess.KNIGHT, chess.WHITE)
            _p(b, random.choice([chess.D7, chess.E7]), random.choice([chess.QUEEN, chess.ROOK]), chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); _add_filler(b, 0, random.randint(1,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_double_check(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK)
            _p(b, chess.E5, chess.KNIGHT, chess.WHITE); _p(b, chess.B2, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); _add_filler(b, 0, random.randint(0,2)); return True
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
            _p(b, chess.D7, random.choice([chess.QUEEN, chess.ROOK]), chess.BLACK); return True
        return self._try_tactic(tk, dr, setup, min_adv=500, depth=3)

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
            _p(b, chess.E4, chess.PAWN, chess.BLACK); _p(b, chess.B2, chess.KING, chess.WHITE)
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

    def _g_deflection(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK); _p(b, chess.C8, chess.ROOK, chess.BLACK)
            _p(b, chess.A1, chess.ROOK, chess.WHITE); _p(b, chess.B2, chess.KING, chess.WHITE)
            _add_filler(b, random.randint(1,2), random.randint(0,1)); return True
        return self._try_mate(tk, dr, setup, depth=3)

    def _g_decoy(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK); _p(b, chess.D8, chess.QUEEN, chess.BLACK)
            _p(b, chess.A1, chess.ROOK, chess.WHITE); _p(b, chess.B4, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_clearance(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.BISHOP, chess.WHITE); _p(b, chess.E1, chess.ROOK, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); _add_filler(b, 0, random.randint(1,2)); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_zwischenzug(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK); _p(b, chess.F7, chess.PAWN, chess.BLACK)
            _p(b, chess.E1, chess.QUEEN, chess.WHITE); _p(b, chess.E5, chess.KNIGHT, chess.WHITE)
            _p(b, chess.D7, random.choice([chess.ROOK, chess.BISHOP]), chess.BLACK)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_remove_defender(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK); _p(b, chess.F6, chess.KNIGHT, chess.BLACK)
            _p(b, chess.D4, chess.KNIGHT, chess.WHITE); _p(b, chess.D1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_underpromotion(self, tk, dr):
        def setup(b):
            corner = random.choice([chess.A8, chess.H8]); kf = chess.square_file(corner)
            _p(b, corner, chess.KING, chess.BLACK)
            for df in [-1, 0, 1]:
                f = kf + df
                if 0 <= f <= 7 and f != kf:
                    sq = chess.square(f, 7)
                    if random.random() < 0.7: _p(b, sq, random.choice([chess.ROOK, chess.PAWN, chess.BISHOP]), chess.BLACK)
            for df in [-1, 1]:
                f = kf + df
                if 0 <= f <= 7:
                    sq7 = chess.square(f, 6)
                    if not b.piece_at(sq7) and random.random() < 0.5: _p(b, sq7, random.choice([chess.PAWN, chess.BISHOP]), chess.BLACK)
            pawn_f = (kf - 1) if kf > 0 else (kf + 1)
            _p(b, chess.square(pawn_f, 6), chess.PAWN, chess.WHITE)
            support_files = [f for f in range(8) if f != pawn_f and f != kf]
            if support_files:
                sf = random.choice(support_files)
                _p(b, chess.square(sf, random.randint(3,5)), random.choice([chess.QUEEN, chess.ROOK]), chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_mate(tk, dr, setup, lambda b,m: bool(m.promotion) and m.promotion != chess.QUEEN, depth=3)

    def _g_interference(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D7, chess.BISHOP, chess.BLACK); _p(b, chess.E6, chess.ROOK, chess.BLACK)
            _p(b, chess.D4, chess.KNIGHT, chess.WHITE); _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_trapped_piece(self, tk, dr):
        def setup(b):
            _p(b, chess.A1, chess.KING, chess.BLACK); _p(b, chess.H8, chess.ROOK, chess.BLACK)
            _p(b, chess.H1, chess.ROOK, chess.WHITE); _p(b, chess.G1, chess.KNIGHT, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_weak_back_rank(self, tk, dr):
        def setup(b):
            _p(b, chess.G8, chess.KING, chess.BLACK)
            _p(b, chess.F7, chess.PAWN, chess.BLACK); _p(b, chess.G7, chess.PAWN, chess.BLACK); _p(b, chess.H7, chess.PAWN, chess.BLACK)
            rf = random.randint(0, 5); _p(b, chess.square(rf, 1), chess.ROOK, chess.WHITE)
            for r in range(2, 8):
                sq = chess.square(rf, r)
                if b.piece_at(sq) and b.piece_type_at(sq) != chess.KING: b.remove_piece_at(sq)
            _p(b, chess.B2, chess.KING, chess.WHITE); _add_filler(b, 0, random.randint(0,2)); return True
        return self._try_mate(tk, dr, setup, depth=2)

    def _g_overloaded_piece(self, tk, dr):
        def setup(b):
            _p(b, chess.A8, chess.KING, chess.BLACK); _p(b, chess.E4, chess.QUEEN, chess.BLACK)
            _p(b, chess.D1, chess.ROOK, chess.WHITE); _p(b, chess.C4, chess.KNIGHT, chess.WHITE)
            _p(b, chess.H1, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=250, depth=3)

    def _g_attraction(self, tk, dr):
        def setup(b):
            _p(b, chess.E8, chess.KING, chess.BLACK)
            _p(b, chess.D1, chess.QUEEN, chess.WHITE); _p(b, chess.B4, chess.BISHOP, chess.WHITE)
            _p(b, chess.A2, chess.KING, chess.WHITE); return True
        return self._try_tactic(tk, dr, setup, min_adv=200, depth=3)

    def _g_line_clearance(self, tk, dr): return self._g_clearance(tk, dr)

    def _g_square_vacation(self, tk, dr):
        def setup(b):
            _p(b, chess.H8, chess.KING, chess.BLACK)
            _p(b, chess.E4, chess.KNIGHT, chess.WHITE); _p(b, chess.E1, chess.QUEEN, chess.WHITE)
            _p(b, chess.B2, chess.KING, chess.WHITE); _add_filler(b, 0, random.randint(0,1)); return True
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
            _p(b, chess.D1, chess.QUEEN, chess.WHITE); _p(b, chess.A2, chess.KING, chess.WHITE)
            _p(b, chess.square(random.randint(3,6), random.randint(3,5)), random.choice([chess.PAWN, chess.KNIGHT]), chess.BLACK)
            return True
        return self._try_tactic(tk, dr, setup, min_adv=300, depth=3)

    def _g_general(self, tk, dr):
        for _ in range(200):
            b = _eb()
            if not _place_kings(b, range(0,3), range(6,8)): continue
            nw = random.randint(2, 5); nb = random.randint(2, 5)
            wtypes = [chess.QUEEN] + [chess.ROOK]*2 + [chess.BISHOP]*2 + [chess.KNIGHT]*2
            btypes = [chess.QUEEN] + [chess.ROOK]*2 + [chess.BISHOP]*2 + [chess.KNIGHT]*2
            placed = {b.king(chess.WHITE), b.king(chess.BLACK)}
            
            for _ in range(nw * 4):
                if nw <= 0: break
                sq = random.randint(0, 63)
                if sq in placed: continue
                if chess.square_rank(sq) not in range(1,7): continue
                _p(b, sq, random.choice(wtypes), chess.WHITE); placed.add(sq); nw -= 1
                
            for _ in range(nb * 4):
                if nb <= 0: break
                sq = random.randint(0, 63)
                if sq in placed: continue
                if chess.square_rank(sq) not in range(1,7): continue
                _p(b, sq, random.choice(btypes), chess.BLACK); placed.add(sq); nb -= 1
                
            pawn_files_w = random.sample(range(8), random.randint(3, 5))
            for f in pawn_files_w:
                r = random.choice([1, 2, 3])
                sq = chess.square(f, r)
                if sq not in placed: 
                    _p(b, sq, chess.PAWN, chess.WHITE); placed.add(sq)
                    
            pawn_files_b = random.sample(range(8), random.randint(3, 5))
            for f in pawn_files_b:
                r = random.choice([4, 5, 6])
                sq = chess.square(f, r)
                if sq not in placed:
                    _p(b, sq, chess.PAWN, chess.BLACK); placed.add(sq)
                    
            if not b.is_valid() or b.is_game_over(): continue
            ti = THEME_REGISTRY.get(tk)
            if not ti: continue
            is_mate_theme = ti.category == "Checkmate"
            m, s, m2, s2 = _search_id(b, 3 if not is_mate_theme else 2)
            if not m: continue
            if is_mate_theme and s < 9000: continue
            if not is_mate_theme and s < 150: continue
            if is_mate_theme:
                if m2 and s2 > 9000 - 100: continue
            else:
                if m2 and s2 > s - 75: continue
            pv = _get_solution(b, m, 3)
            return Puzzle(fen=b.fen(), theme_key=tk, theme_name=ti.display_name, difficulty=_rand_diff(dr),
                moves=pv, description=f"Find the best move in this {ti.display_name} puzzle.",
                hint=f"Look for a {ti.display_name.lower()} opportunity.", category=ti.category)
        return None

# ═══════════════════════════════════════════════════════════════════════════════════
# AUDIO SYNTHESIZER (PROCEDURAL SFX)
# ═══════════════════════════════════════════════════════════════════════════════════

class AudioSynth:
    """Procedurally generates high-quality chess sound effects using numpy."""
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def _envelope(self, length, attack=0.01, decay=0.1, sustain=0.0, release=0.1):
        a = int(self.sr * attack)
        d = int(self.sr * decay)
        r = int(self.sr * release)
        s = length - a - d - r
        if s < 0: s = 0
        env = np.concatenate([
            np.linspace(0, 1, a, endpoint=False),
            np.linspace(1, sustain, d, endpoint=False),
            np.full(s, sustain),
            np.linspace(sustain, 0, r, endpoint=False)
        ])
        if len(env) < length:
            env = np.pad(env, (0, length - len(env)))
        return env[:length]

    def _to_int16(self, signal):
        signal = np.clip(signal, -1.0, 1.0)
        return (signal * 32767).astype(np.int16)

    def synth_move(self):
        dur = 0.15
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        wave_data = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)
        env = self._envelope(len(t), attack=0.005, decay=0.05, sustain=0.0, release=0.1)
        return wave_data * env * 0.5

    def synth_capture(self):
        dur = 0.2
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        tone = np.sin(2 * np.pi * 150 * t)
        noise = np.random.uniform(-1, 1, len(t))
        env = self._envelope(len(t), attack=0.002, decay=0.1, sustain=0.0, release=0.1)
        return (tone * 0.6 + noise * 0.4) * env * 0.7

    def synth_check(self):
        dur = 0.4
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        wave_data = np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)
        env = self._envelope(len(t), attack=0.01, decay=0.2, sustain=0.1, release=0.2)
        return wave_data * env * 0.6

    def synth_mate(self):
        dur = 1.0
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        freqs = [523.25, 659.25, 783.99, 1046.50]
        signal = np.zeros_like(t)
        note_dur = 0.2
        for i, f in enumerate(freqs):
            start = int(i * note_dur * self.sr)
            end = start + int(note_dur * self.sr * 2)
            if start >= len(t): break
            t_note = np.linspace(0, note_dur*2, len(t[start:end]), endpoint=False)
            wave_data = np.sin(2 * np.pi * f * t_note)
            env = self._envelope(len(t_note), attack=0.01, decay=0.1, sustain=0.3, release=0.2)
            signal[start:end] += wave_data * env * 0.5
        return signal

    def create_sfx_track(self, events, total_duration):
        total_samples = int(self.sr * total_duration)
        track = np.zeros(total_samples)
        for ev in events:
            time = ev['time']
            sound_type = ev['type']
            if sound_type == 'move': sfx = self.synth_move()
            elif sound_type == 'capture': sfx = self.synth_capture()
            elif sound_type == 'check': sfx = self.synth_check()
            elif sound_type == 'mate': sfx = self.synth_mate()
            else: continue
            start_idx = int(time * self.sr)
            end_idx = start_idx + len(sfx)
            if end_idx > len(track):
                sfx = sfx[:len(track) - start_idx]
            track[start_idx:end_idx] += sfx
        return self._to_int16(track)

    def save_wav(self, filename, signal_int16):
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(signal_int16.tobytes())


# ═══════════════════════════════════════════════════════════════════════════════════
# YOUTUBE SHORTS VIDEO RENDERER
# ═══════════════════════════════════════════════════════════════════════════════════

PIECE_UNICODE = {
    chess.PAWN:{chess.WHITE:"\u2659",chess.BLACK:"\u265F"}, chess.KNIGHT:{chess.WHITE:"\u2658",chess.BLACK:"\u265E"},
    chess.BISHOP:{chess.WHITE:"\u2657",chess.BLACK:"\u265D"}, chess.ROOK:{chess.WHITE:"\u2656",chess.BLACK:"\u265C"},
    chess.QUEEN:{chess.WHITE:"\u2655",chess.BLACK:"\u265B"}, chess.KING:{chess.WHITE:"\u2654",chess.BLACK:"\u265A"},
}

class VideoRenderer:
    def __init__(self, output_path, fps=30):
        self.output_path = output_path
        self.fps = fps
        self.width = 1080
        self.height = 1920
        self.board_size = 800
        self.board_x = 140
        self.board_y = 450
        self.bar_x = 50
        self.bar_w = 40
        self.square_size = self.board_size // 8
        self._bg_image = None
        self._piece_imgs = {}

    def _get_eval(self, b):
        _, s, _, _ = _search_id(b, 2)
        return s if b.turn == chess.WHITE else -s

    def _eval_to_text(self, score):
        if score >= MATE_SCORE - 1000:
            dist = score - MATE_SCORE
            return f"M{int(dist / 2) + 1}"
        if score <= -MATE_SCORE + 1000:
            dist = -MATE_SCORE - score
            return f"-M{int(dist / 2) + 1}"
        return f"{score / 100.0:+.1f}"

    def _precompute_assets(self, puzzle):
        self._bg_image = QImage(self.width, self.height, QImage.Format.Format_RGBA8888)
        self._bg_image.fill(QColor(18, 18, 24))
        p = QPainter(self._bg_image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_top_ui(p, puzzle, "intro")
        
        light_sq = QColor(235, 236, 208)
        dark_sq = QColor(115, 149, 82)
        for f in range(8):
            for r in range(8):
                x = self.board_x + f * self.square_size
                y = self.board_y + (7 - r) * self.square_size
                color = light_sq if (f + r) % 2 == 0 else dark_sq
                p.fillRect(QRect(x, y, self.square_size, self.square_size), color)
        p.end()

        self._piece_imgs = {}
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for cl in [chess.WHITE, chess.BLACK]:
                img = QImage(self.square_size, self.square_size, QImage.Format.Format_RGBA8888)
                img.fill(Qt.transparent)
                pp = QPainter(img)
                pp.setRenderHint(QPainter.RenderHint.Antialiasing)
                self._draw_piece_text(pp, 0, 0, chess.Piece(pt, cl))
                pp.end()
                self._piece_imgs[(pt, cl)] = img

    def render(self, puzzle, assets_dir="assets", enable_sfx=True, bgm_path=None):
        self._precompute_assets(puzzle)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        first_move = chess.Move.from_uci(puzzle.moves[0])
        
        audio_events = []
        current_time = 0.0
        
        e1 = self._get_eval(chess.Board(puzzle.fen))
        
        for i in range(int(5 * self.fps)):
            frame = self._draw_frame(puzzle, chess.Board(puzzle.fen), phase="intro", highlight_sq=first_move.from_square, frame_idx=i, eval_start=e1, eval_end=e1)
            out.write(frame)
        current_time += 5.0
            
        for i in range(int(3 * self.fps)):
            frame = self._draw_frame(puzzle, chess.Board(puzzle.fen), phase="think", frame_idx=i, eval_start=e1, eval_end=e1)
            out.write(frame)
        current_time += 3.0
            
        board = chess.Board(puzzle.fen)
        move_duration = 0.5
        pause_duration = 0.5
        
        for i, uci in enumerate(puzzle.moves):
            move = chess.Move.from_uci(uci)
            is_capture = board.is_capture(move)
            
            e_hl = self._get_eval(board)
            for f_idx in range(int(0.5 * self.fps)):
                frame = self._draw_frame(puzzle, board, phase="reveal_highlight", highlight_sq=move.from_square, frame_idx=f_idx, eval_start=e_hl, eval_end=e_hl)
                out.write(frame)
            current_time += 0.5
                
            e_start = self._get_eval(board)
            board.push(move)
            e_end = self._get_eval(board)
            board.pop()
            
            if is_capture:
                audio_events.append({'time': current_time, 'type': 'capture'})
            else:
                audio_events.append({'time': current_time, 'type': 'move'})
                
            anim_frames = int(move_duration * self.fps)
            for f_idx in range(anim_frames):
                progress = (f_idx + 1) / anim_frames
                frame = self._draw_frame(puzzle, board, phase="reveal_move", anim_move=move, progress=progress, capture_sq=move.to_square if is_capture else None, frame_idx=f_idx, eval_start=e_start, eval_end=e_end)
                out.write(frame)
            current_time += move_duration
                
            board.push(move)
            
            if board.is_checkmate():
                audio_events.append({'time': current_time, 'type': 'mate'})
            elif board.is_check():
                audio_events.append({'time': current_time + 0.05, 'type': 'check'})
                
            e_pause = self._get_eval(board)
            for f_idx in range(int(pause_duration * self.fps)):
                frame = self._draw_frame(puzzle, board, phase="reveal_pause", highlight_sq=move.to_square, frame_idx=f_idx, eval_start=e_pause, eval_end=e_pause)
                out.write(frame)
            current_time += pause_duration
                
        e_outro = self._get_eval(board)
        for i in range(int(4 * self.fps)):
            frame = self._draw_frame(puzzle, board, phase="outro", frame_idx=i, eval_start=e_outro, eval_end=e_outro)
            out.write(frame)
        current_time += 4.0
            
        out.release()
        
        self._mix_audio(audio_events, current_time, enable_sfx, bgm_path)

    def _mix_audio(self, audio_events, total_duration, enable_sfx, bgm_path):
        cmd = ['ffmpeg', '-y']
        inputs = ['-i', self.output_path]
        filters = []
        map_audio = '1:a'
        
        has_sfx = False
        sfx_path = os.path.join(os.path.dirname(self.output_path), "temp_sfx.wav")
        
        if enable_sfx:
            try:
                synth = AudioSynth()
                sfx_signal = synth.create_sfx_track(audio_events, total_duration)
                synth.save_wav(sfx_path, sfx_signal)
                inputs += ['-i', sfx_path]
                has_sfx = True
            except Exception as e:
                logger.error(f"SFX generation failed: {e}")
                
        has_bgm = False
        if bgm_path and os.path.exists(bgm_path):
            inputs += ['-i', bgm_path]
            bgm_idx = 2 if has_sfx else 1
            filters.append(f'[{bgm_idx}:a]aloop=loop=-1:size=2e9,volume=0.25[bgm]')
            has_bgm = True
            
        temp_path = self.output_path + ".tmp.mp4"
        
        if has_sfx and has_bgm:
            filters.append('[bgm][1:a]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=50[aout]')
            filter_str = ';'.join(filters)
            cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_str, '-map', '0:v', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', temp_path]
        elif has_sfx:
            cmd = ['ffmpeg', '-y'] + inputs + ['-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', temp_path]
        elif has_bgm:
            filter_str = ';'.join(filters)
            cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_str, '-map', '0:v', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', temp_path]
        else:
            return
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            os.replace(temp_path, self.output_path)
            logger.info("Audio mixed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio mixing failed (ffmpeg): {e.stderr}")
        except Exception as e:
            logger.error(f"Audio mixing general error: {e}")
        finally:
            if os.path.exists(sfx_path):
                try: os.remove(sfx_path)
                except: pass

    def _draw_frame(self, puzzle, board, phase="intro", anim_move=None, progress=0.0, highlight_sq=None, capture_sq=None, frame_idx=0, eval_start=0, eval_end=0):
        img = self._bg_image.copy()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self._draw_eval_bar(p, eval_start, eval_end, progress)
        self._draw_dynamic_board(p, board, anim_move, progress, highlight_sq, capture_sq, frame_idx)
        self._draw_bottom_ui(p, phase, frame_idx)
        
        if phase == "outro":
            fade_frames = int(1.5 * self.fps)
            if frame_idx > int(2.5 * self.fps):
                fade_progress = min(1.0, (frame_idx - int(2.5 * self.fps)) / fade_frames)
                p.fillRect(QRect(0, 0, self.width, self.height), QColor(0, 0, 0, int(255 * fade_progress)))
                
        p.end()
        
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        arr = np.array(ptr).reshape(self.height, self.width, 4)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return bgr

    def _draw_top_ui(self, p, puzzle, phase):
        p.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 40, QFont.Bold)
        p.setFont(font)
        p.drawText(QRect(0, 150, self.width, 60), Qt.AlignmentFlag.AlignCenter, puzzle.theme_name)
        
        p.setPen(QColor(255, 215, 0))
        font.setPointSize(28)
        p.setFont(font)
        diff_str = "\u2605" * puzzle.difficulty + "\u2606" * (10 - puzzle.difficulty)
        p.drawText(QRect(0, 230, self.width, 50), Qt.AlignmentFlag.AlignCenter, diff_str)
        
        p.setPen(QColor(200, 200, 200))
        font.setPointSize(24)
        font.setBold(False)
        p.setFont(font)
        turn_str = "White to move" if chess.Board(puzzle.fen).turn == chess.WHITE else "Black to move"
        hook_text = f"{turn_str} and " + ("win material." if puzzle.category != "Checkmate" else "checkmate.")
        p.drawText(QRect(0, 300, self.width, 50), Qt.AlignmentFlag.AlignCenter, hook_text)

    def _draw_eval_bar(self, p, eval_start, eval_end, progress):
        bar_y = self.board_y
        bar_h = self.board_size
        progress_eased = 1 - (1 - progress) ** 3
        current_eval = eval_start + (eval_end - eval_start) * progress_eased
        
        if current_eval >= MATE_SCORE - 1000:
            pct = 1.0
        elif current_eval <= -MATE_SCORE + 1000:
            pct = 0.0
        else:
            clamped = max(-1500, min(1500, current_eval))
            pct = 0.5 + (clamped / 3000.0)
            
        white_h = int(bar_h * pct)
        
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 40))
        p.drawRect(QRect(self.bar_x, bar_y, self.bar_w, bar_h))
        
        p.setBrush(QColor(230, 230, 230))
        p.drawRect(QRect(self.bar_x, bar_y + bar_h - white_h, self.bar_w, white_h))
        
        if 0 < progress < 1 and eval_start != eval_end:
            glow_y = bar_y + bar_h - white_h
            p.setBrush(QColor(255, 215, 0, 200))
            p.drawRect(QRect(self.bar_x - 2, glow_y - 3, self.bar_w + 4, 6))
            
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(100, 100, 100), 2))
        p.drawRect(QRect(self.bar_x, bar_y, self.bar_w, bar_h))
        
        font = QFont("Arial", 14, QFont.Bold)
        p.setFont(font)
        eval_text = self._eval_to_text(current_eval)
        
        if pct > 0.5:
            text_y = bar_y + 10
            text_color = QColor(255, 255, 255)
            outline_color = QColor(0, 0, 0)
        else:
            text_y = bar_y + bar_h - 30
            text_color = QColor(0, 0, 0)
            outline_color = QColor(255, 255, 255)
            
        text_rect = QRect(self.bar_x - 10, text_y, self.bar_w + 20, 20)
        
        p.setPen(outline_color)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p.drawText(text_rect.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, eval_text)
            
        p.setPen(text_color)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, eval_text)

    def _draw_dynamic_board(self, p, board, anim_move, progress, highlight_sq, capture_sq, frame_idx):
        highlight_clr = QColor(255, 255, 0, 150)
        
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x = self.board_x + f * self.square_size
            y = self.board_y + (7 - r) * self.square_size
            
            if highlight_sq is not None and sq == highlight_sq:
                p.fillRect(QRect(x, y, self.square_size, self.square_size), highlight_clr)
            if capture_sq is not None and sq == capture_sq:
                pulse = (math.sin(frame_idx * 0.5) + 1) / 2
                red_alpha = int(100 + 100 * pulse)
                p.fillRect(QRect(x, y, self.square_size, self.square_size), QColor(255, 0, 0, red_alpha))
                
            if anim_move and sq == anim_move.from_square:
                continue
                
            piece = board.piece_at(sq)
            if piece:
                self._draw_piece(p, x, y, piece)
                
        if anim_move:
            f1, r1 = chess.square_file(anim_move.from_square), chess.square_rank(anim_move.from_square)
            f2, r2 = chess.square_file(anim_move.to_square), chess.square_rank(anim_move.to_square)
            
            x1 = self.board_x + f1 * self.square_size
            y1 = self.board_y + (7 - r1) * self.square_size
            x2 = self.board_x + f2 * self.square_size
            y2 = self.board_y + (7 - r2) * self.square_size
            
            ax = x1 + (x2 - x1) * progress
            ay = y1 + (y2 - y1) * progress
            
            piece = board.piece_at(anim_move.from_square)
            if piece:
                self._draw_piece(p, ax, ay, piece)

    def _draw_piece_text(self, p, x, y, piece):
        font = QFont("Segoe UI Symbol", 50, QFont.Bold)
        p.setFont(font)
        t = PIECE_UNICODE.get(piece.piece_type, {}).get(piece.color, "?")
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(QRect(int(x+3), int(y+3), self.square_size, self.square_size), Qt.AlignmentFlag.AlignCenter, t)
        p.setPen(QColor(255, 255, 255) if piece.color == chess.WHITE else QColor(20, 20, 20))
        p.drawText(QRect(int(x), int(y), self.square_size, self.square_size), Qt.AlignmentFlag.AlignCenter, t)

    def _draw_piece(self, p, x, y, piece):
        img = self._piece_imgs.get((piece.piece_type, piece.color))
        if img:
            p.drawImage(int(x), int(y), img)

    def _draw_bottom_ui(self, p, phase, frame_idx):
        p.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 36, QFont.Bold)
        p.setFont(font)
        text = ""; sub_text = ""
        if phase == "intro": text = "Can you solve this?"
        elif phase == "think":
            sec = int(frame_idx / self.fps)
            text = str(3 - sec); sub_text = "Think carefully..."
        elif phase == "reveal_highlight": text = "The Solution is..."
        elif phase == "reveal_move" or phase == "reveal_pause": text = "Solution revealed below!"
        elif phase == "outro":
            text = "Subscribe for daily puzzles!"; sub_text = "Like and Comment your guess!"
        p.drawText(QRect(0, self.board_y + self.board_size + 80, self.width, 60), Qt.AlignmentFlag.AlignCenter, text)
        if sub_text:
            font.setPointSize(24); font.setBold(False); p.setFont(font)
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRect(0, self.board_y + self.board_size + 150, self.width, 50), Qt.AlignmentFlag.AlignCenter, sub_text)


# ═══════════════════════════════════════════════════════════════════════════════════
# MANIFEST MANAGER & BATCH PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════════

class ManifestManager:
    def __init__(self, output_dir):
        self.path = os.path.join(output_dir, "manifest.json")
        self.fens = set()
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.fens = set(data.get("fens", []))
                logger.info(f"Manifest loaded: {len(self.fens)} existing puzzles found.")
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
                self.fens = set()

    def has(self, fen):
        return fen in self.fens

    def add(self, fen):
        self.fens.add(fen)
        self.save()

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"fens": list(self.fens)}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest: {e}")

class BatchProcessor(QThread):
    progress = Signal(int, int, str)
    finished_batch = Signal(str)

    def __init__(self, generator, settings, selected_themes, diff_range):
        super().__init__()
        self.generator = generator
        self.settings = settings
        self.selected_themes = selected_themes
        self.diff_range = diff_range

    def run(self):
        out_dir = self.settings["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        manifest = ManifestManager(out_dir)
        
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        count = self.settings["count"]
        min_moves = self.settings["min_moves"]
        logger.info(f"Batch Shorts generation started: count={count}, output={out_dir}")
        self.progress.emit(0, count, "Generating puzzles...")
        
        puzzles = []
        attempts = 0
        while len(puzzles) < count and attempts < count * 30:
            attempts += 1
            tk = random.choice(self.selected_themes)
            try:
                p = self.generator.generate(theme_key=tk, difficulty_range=self.diff_range, count=1)
                if p and len(p[0].moves) >= min_moves:
                    fen = p[0].fen
                    if not manifest.has(fen):
                        puzzles.append(p[0])
                        manifest.add(fen)
                        self.progress.emit(len(puzzles), count, f"Generated puzzle {len(puzzles)}/{count}: {p[0].theme_name}")
                        logger.info(f"Shorts batch: generated puzzle {len(puzzles)}/{count}")
                    else:
                        self.progress.emit(len(puzzles), count, f"Skipping duplicate... ({attempts})")
            except Exception as e:
                logger.error(f"Shorts gen error (attempt {attempts}): {e}")

        if not puzzles:
            self.progress.emit(0, count, "Failed to generate enough unique puzzles. Try broader filters.")
            logger.error("Batch Shorts: failed to generate puzzles.")
            return
            
        metadata = []
        for i, p in enumerate(puzzles):
            self.progress.emit(i, count, f"Rendering video {i+1}/{count}: {p.theme_name}")
            filename = f"short_{i+1:03d}_{p.theme_key}.mp4"
            filepath = os.path.join(out_dir, filename)
            try:
                renderer = VideoRenderer(filepath)
                renderer.render(
                    p, 
                    assets_dir, 
                    enable_sfx=self.settings.get("enable_sfx", True), 
                    bgm_path=self.settings.get("bgm_path")
                )
                metadata.append({
                    "filename": filename,
                    "title": f"Chess Puzzle: {p.theme_name} (Difficulty {p.difficulty}/10)",
                    "description": f"Can you find the best move?\nTheme: {p.theme_name}\nCategory: {p.category}\nSolution: {p.moves}",
                    "tags": f"chess, puzzle, {p.theme_name}, {p.category}, shorts",
                    "theme": p.theme_name,
                    "difficulty": p.difficulty
                })
            except Exception as e:
                logger.error(f"Error rendering {filename}: {e}")
                
        json_path = os.path.join(out_dir, "youtube_metadata.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Batch metadata exported: {json_path}")
        except Exception as e:
            logger.error(f"Batch JSON export error: {e}")
            
        logger.info(f"Batch Shorts complete: {len(puzzles)} videos exported to {out_dir}")
        self.progress.emit(count, count, "Batch complete!")
        self.finished_batch.emit(out_dir)

class BatchGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Generate YouTube Shorts")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Batch Size (Videos):"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        size_layout.addWidget(self.size_spin)
        layout.addLayout(size_layout)
        
        moves_layout = QHBoxLayout()
        moves_layout.addWidget(QLabel("Min Solution Moves:"))
        self.moves_spin = QSpinBox()
        self.moves_spin.setRange(1, 5)
        self.moves_spin.setValue(2)
        moves_layout.addWidget(self.moves_spin)
        layout.addLayout(moves_layout)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output Folder:"))
        self.dir_edit = QLineEdit(os.path.abspath("shorts_output"))
        self.dir_btn = QPushButton("Browse")
        self.dir_btn.clicked.connect(self._browse)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.dir_btn)
        layout.addLayout(dir_layout)
        
        sound_group = QGroupBox("Sound Design")
        sound_layout = QVBoxLayout(sound_group)
        
        self.sfx_check = QCheckBox("Enable Synthesized Sound Effects (Moves, Captures, Checks)")
        self.sfx_check.setChecked(True)
        sound_layout.addWidget(self.sfx_check)
        
        self.bgm_check = QCheckBox("Enable Background Music (Ducked under SFX)")
        self.bgm_check.setChecked(True)
        sound_layout.addWidget(self.bgm_check)
        
        bgm_path_layout = QHBoxLayout()
        bgm_path_layout.addWidget(QLabel("BGM Track:"))
        self.bgm_path_edit = QLineEdit(os.path.abspath("assets/background.mp3"))
        self.bgm_path_btn = QPushButton("Browse")
        self.bgm_path_btn.clicked.connect(self._browse_bgm)
        bgm_path_layout.addWidget(self.bgm_path_edit)
        bgm_path_layout.addWidget(self.bgm_path_btn)
        sound_layout.addLayout(bgm_path_layout)
        
        layout.addWidget(sound_group)
        
        info = QLabel("Note: Requires 'opencv-python' and 'numpy'.\nPlace 'background.mp3' in an 'assets' folder for audio (requires ffmpeg).")
        info.setStyleSheet("color: #f9e2af; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        btn_layout = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Batch")
        self.gen_btn.setObjectName("primaryBtn")
        self.gen_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.gen_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d: self.dir_edit.setText(d)
        
    def _browse_bgm(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Background Music", "", "Audio Files (*.mp3 *.wav *.aac)")
        if f: self.bgm_path_edit.setText(f)
        
    def get_settings(self):
        return {
            "count": self.size_spin.value(),
            "min_moves": self.moves_spin.value(),
            "output_dir": self.dir_edit.text(),
            "enable_sfx": self.sfx_check.isChecked(),
            "enable_bgm": self.bgm_check.isChecked(),
            "bgm_path": self.bgm_path_edit.text() if self.bgm_check.isChecked() else None
        }

# ═══════════════════════════════════════════════════════════════════════════════════
# BOARD RENDERING INFRASTRUCTURE — Snippets: BoardWidget, CompositeLayout, render_board_image
# ═══════════════════════════════════════════════════════════════════════════════════

BOARD_PX = 640
SQ_SIZE = 80
ARROW_STYLE_THICK = "thick"
ARROW_STYLE_THIN = "thin"
ARROW_STYLE_LINE = "line"
ANIM_SPEED_DEFAULT = 250
ANIM_FPS = 60
FILES_STR = "abcdefgh"
RANKS_STR = "12345678"

# ─── Theme ───
@dataclass
class Theme:
    light_sq: QColor
    dark_sq: QColor
    highlight: QColor
    last_move: QColor
    arrow_clr: QColor
    coord_light: QColor = None
    coord_dark: QColor = None

    def __post_init__(self):
        if self.coord_light is None:
            self.coord_light = self.dark_sq
        if self.coord_dark is None:
            self.coord_dark = self.light_sq

THEMES: Dict[str, Theme] = {
    "Classic": Theme(
        light_sq=QColor(238, 238, 210),
        dark_sq=QColor(118, 150, 86),
        highlight=QColor(255, 255, 100, 130),
        last_move=QColor(155, 199, 0, 100),
        arrow_clr=QColor(137, 180, 250, 220),
    ),
    "Dark": Theme(
        light_sq=QColor(55, 63, 82),
        dark_sq=QColor(36, 41, 55),
        highlight=QColor(137, 180, 250, 130),
        last_move=QColor(80, 100, 140, 120),
        arrow_clr=QColor(245, 194, 99, 220),
    ),
    "Walnut": Theme(
        light_sq=QColor(233, 209, 167),
        dark_sq=QColor(122, 80, 45),
        highlight=QColor(255, 215, 0, 130),
        last_move=QColor(205, 133, 63, 100),
        arrow_clr=QColor(220, 50, 50, 220),
    ),
}

# ─── paint_context ───
@contextmanager
def paint_context(target):
    p = QPainter(target)
    try:
        yield p
    finally:
        p.end()

# ─── get_render_assets ───
def get_render_assets(sq_size):
    font_piece = QFont("Segoe UI Symbol", int(sq_size * 0.78))
    font_piece.setBold(True)
    font_coord = QFont("Arial", max(9, sq_size // 6), QFont.Bold)
    return font_piece, font_coord

# ─── _ease_out_cubic ───
def _ease_out_cubic(t):
    return 1 - (1 - t) ** 3

# ─── _draw_piece ───
def _draw_piece(p, piece, sr, sc, sz, font):
    if piece is None:
        return
    t = PIECE_UNICODE.get(piece.piece_type, {}).get(piece.color, "?")
    if t == "?":
        return
    p.setFont(font)
    x, y = int(sc * sz), int(sr * sz)
    p.setPen(QColor(0, 0, 0, 90))
    p.drawText(QRect(x + 2, y + 3, sz, sz), Qt.AlignmentFlag.AlignCenter, t)
    p.setPen(QColor(255, 255, 255) if piece.color == chess.WHITE else QColor(20, 20, 20))
    p.drawText(QRect(x, y, sz, sz), Qt.AlignmentFlag.AlignCenter, t)

# ─── _draw_piece_at (for animation — supports fractional positions and scaling) ───
def _draw_piece_at(p, piece, sr, sc, sz, width, height, font):
    if piece is None:
        return
    t = PIECE_UNICODE.get(piece.piece_type, {}).get(piece.color, "?")
    if t == "?":
        return
    p.setFont(font)
    x = sc * sz + (sz - width) / 2.0
    y = sr * sz + (sz - height) / 2.0
    p.setPen(QColor(0, 0, 0, 90))
    p.drawText(QRectF(x + 2, y + 3, width, height), Qt.AlignmentFlag.AlignCenter, t)
    p.setPen(QColor(255, 255, 255) if piece.color == chess.WHITE else QColor(20, 20, 20))
    p.drawText(QRectF(x, y, width, height), Qt.AlignmentFlag.AlignCenter, t)

# ─── _draw_arrow ───
def _draw_arrow(p, x1, y1, x2, y2, color, sz, style=None):
    p.setPen(QPen(color, max(3, sz // 12),
                  Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(color)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    angle = math.atan2(y2 - y1, x2 - x1)
    length = sz // 5
    p1 = QPointF(x2 - length * math.cos(angle - math.pi / 6),
                 y2 - length * math.sin(angle - math.pi / 6))
    p2 = QPointF(x2 - length * math.cos(angle + math.pi / 6),
                 y2 - length * math.sin(angle + math.pi / 6))
    poly = QPolygonF([QPointF(x2, y2), p1, p2])
    p.drawPolygon(poly)

# ─── _reconstruct_piece ───
_PIECE_CHAR_MAP = {
    'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP,
    'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING,
}

def _reconstruct_piece(c):
    if c == '.' or c is None or c == '':
        return None
    c = str(c)
    if len(c) != 1:
        return None
    pt = _PIECE_CHAR_MAP.get(c.lower())
    if pt is None:
        return None
    color = chess.WHITE if c.isupper() else chess.BLACK
    return chess.Piece(pt, color)


# ═══════════════════════════════════════════════════════════════════════════════════
# CHESS ENGINE ADAPTER — Bridges chess.Board to the BoardWidget API
# ═══════════════════════════════════════════════════════════════════════════════════

class ChessEngine:
    """Lightweight adapter wrapping chess.Board for use with BoardWidget."""

    def __init__(self, board: chess.Board = None):
        self.board = board if board is not None else chess.Board()
        self.last_move = None  # ((fr, fc), (tr, tc)) or None

    @staticmethod
    def rc_to_sq(r, c):
        """Convert (row, col) where row 0 = rank 8, col 0 = file a → chess square index."""
        return chess.square(c, 7 - r)

    def check_squares(self):
        """Return list of (row, col) tuples for squares in check."""
        if self.board.is_check():
            ks = self.board.king(self.board.turn)
            r = 7 - chess.square_rank(ks)
            c = chess.square_file(ks)
            return [(r, c)]
        return []

    def legal_moves(self, r, c):
        """Return list of (row, col) legal target squares from (r, c)."""
        sq = self.rc_to_sq(r, c)
        piece = self.board.piece_at(sq)
        if not piece:
            return []
        targets = []
        for m in self.board.legal_moves:
            if m.from_square == sq:
                tr = 7 - chess.square_rank(m.to_square)
                tc = chess.square_file(m.to_square)
                targets.append((tr, tc))
        return targets

    def make_move(self, fr, fc, tr, tc, promo=None):
        from_sq = self.rc_to_sq(fr, fc)
        to_sq = self.rc_to_sq(tr, tc)
        piece = self.board.piece_at(from_sq)
        if not piece:
            return None
        move = chess.Move(from_sq, to_sq, promotion=promo)
        if move not in self.board.legal_moves:
            return None

        captured_piece = self.board.piece_at(to_sq)
        captured_char = captured_piece.symbol() if captured_piece else '.'
        is_castle = self.board.is_castling(move)

        self.board.push(move)
        is_check = self.board.is_check()
        is_mate = self.board.is_checkmate()

        result = {
            'from': (fr, fc), 'to': (tr, tc),
            'piece_obj': piece, 'captured': captured_char,
            'check': is_check, 'castle': is_castle, 'mate': is_mate,
        }
        self.last_move = ((fr, fc), (tr, tc))
        return result

    def make_move_uci(self, uci_str):
        move = chess.Move.from_uci(uci_str)
        piece = self.board.piece_at(move.from_square)
        if not piece:
            return None

        captured_piece = self.board.piece_at(move.to_square)
        captured_char = captured_piece.symbol() if captured_piece else '.'
        is_castle = self.board.is_castling(move)

        fr = 7 - chess.square_rank(move.from_square)
        fc = chess.square_file(move.from_square)
        tr = 7 - chess.square_rank(move.to_square)
        tc = chess.square_file(move.to_square)

        self.board.push(move)
        is_check = self.board.is_check()
        is_mate = self.board.is_checkmate()

        result = {
            'from': (fr, fc), 'to': (tr, tc),
            'piece_obj': piece, 'captured': captured_char,
            'check': is_check, 'castle': is_castle, 'mate': is_mate,
        }
        self.last_move = ((fr, fc), (tr, tc))
        return result


# ═══════════════════════════════════════════════════════════════════════════════════
# COMPOSITE LAYOUT (from snippet — as-is)
# ═══════════════════════════════════════════════════════════════════════════════════

class CompositeLayout:
    def __init__(self, width, height, sq_size=None, is_vertical=False, is_square=False,
                 is_board_only=False, show_title=False):
        self.w = width
        self.h = height
        self.vert = is_vertical
        self.show_title = show_title
        self.is_board_only = is_board_only
        self.sf = min(width, height) / 1080.0

        if is_board_only:
            self._board_only_layout(sq_size)
        elif is_vertical:
            self._vert_layout(sq_size)
        else:
            self._horiz_layout(sq_size)

    def _make_sq(self, raw_pixels):
        """Round down to nearest multiple of 8, minimum 8."""
        return max(8, (int(raw_pixels) // 8) * 8)

    def _board_only_layout(self, sq_size):
        self.pad = max(4, int(min(self.w, self.h) * 0.01))
        self.title_h = 0
        if sq_size is None:
            board_px = min(self.w, self.h) - (self.pad * 2)
            self.sq_size = self._make_sq(board_px / 8)
        else:
            self.sq_size = sq_size
        self.bpx = self.sq_size * 8
        self.bx = (self.w - self.bpx) // 2
        self.by = (self.h - self.bpx) // 2
        self.mx = self.my = self.mw = self.mh = 0

    def _horiz_layout(self, sq_size):
        self.pad = max(8, int(12 * self.sf))
        self.title_h = max(44, int(80 * self.sf)) if self.show_title else 0
        gap = self.pad
        min_mw = max(180, int(220 * self.sf))
        max_mw = max(280, int(self.w * 0.35))
        target_mw = min(max_mw, max(min_mw, int(280 * self.sf)))
        max_board_w = self.w - target_mw - gap - self.pad * 2
        max_board_h = self.h - self.title_h - self.pad * 3
        if sq_size is None:
            self.sq_size = self._make_sq(min(max_board_w, max_board_h) / 8)
        else:
            self.sq_size = max(8, sq_size)
        self.bpx = self.sq_size * 8
        remaining_w = self.w - self.bpx - gap - self.pad * 2
        if remaining_w < min_mw:
            actual_mw = max(80, remaining_w)
        else:
            actual_mw = max(min_mw, min(max_mw, remaining_w))
        total_content_w = self.bpx + gap + actual_mw
        group_x = (self.w - total_content_w) // 2
        vert_space = self.h - self.title_h - self.pad * 2
        board_y = self.title_h + self.pad + max(0, (vert_space - self.bpx) // 2)
        self.bx = group_x
        self.by = board_y
        self.mx = group_x + self.bpx + gap
        self.my = board_y
        self.mw = actual_mw
        self.mh = self.bpx

    def _vert_layout(self, sq_size):
        self.pad = max(8, int(14 * self.sf))
        self.title_h = max(36, int(50 * self.sf)) if self.show_title else 0
        gap = self.pad
        min_mh = max(120, int(150 * self.sf))
        max_mh = max(180, int(self.h * 0.35))
        target_mh = min(max_mh, max(min_mh, int(200 * self.sf)))
        max_board_w = self.w - self.pad * 2
        max_board_h = self.h - self.title_h - target_mh - gap - self.pad * 3
        if sq_size is None:
            self.sq_size = self._make_sq(min(max_board_w, max_board_h) / 8)
        else:
            self.sq_size = max(8, sq_size)
        self.bpx = self.sq_size * 8
        remaining_h = self.h - self.title_h - self.bpx - gap - self.pad * 3
        if remaining_h < min_mh:
            actual_mh = max(60, remaining_h)
        else:
            actual_mh = max(min_mh, min(max_mh, remaining_h))
        board_x = (self.w - self.bpx) // 2
        total_content_h = self.bpx + gap + actual_mh
        vert_space = self.h - self.title_h - self.pad * 2
        group_y = self.title_h + self.pad + max(0, (vert_space - total_content_h) // 2)
        self.bx = board_x
        self.by = group_y
        self.mx = board_x
        self.my = group_y + self.bpx + gap
        self.mw = self.bpx
        self.mh = actual_mh


# ═══════════════════════════════════════════════════════════════════════════════════
# RENDER BOARD IMAGE (from snippet — as-is)
# ═══════════════════════════════════════════════════════════════════════════════════

def render_board_image(board, last_move=None, selected=None, legal_targets=None,
                       check_squares=None, anim_state=None, sq_size=SQ_SIZE,
                       show_arrow=True, theme=None, flipped=False, arrow_style=None,
                       static_only=False):
    if theme is None:
        theme = THEMES["Classic"]
    sz = sq_size
    img = QImage(sz * 8, sz * 8, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    with paint_context(img) as p:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        (font_piece, font_coord, *_) = get_render_assets(sz)
        check_set = set(check_squares or [])
        skip_sq = set()
        if anim_state:
            skip_sq.add(anim_state['from'])
            skip_sq.add(anim_state['to'])
        b2s = (lambda br, bc: (7 - br, 7 - bc)) if flipped else (lambda br, bc: (br, bc))
        effective_move = None
        if anim_state:
            effective_move = (anim_state['from'], anim_state['to'])
        elif last_move:
            effective_move = last_move

        # 1. Draw board squares and highlights
        for sq in chess.SQUARES:
            br, bc = 7 - chess.square_rank(sq), chess.square_file(sq)
            sr, sc = b2s(br, bc)
            x, y = sc * sz, sr * sz
            is_light = (br + bc) % 2 == 0
            p.fillRect(QRect(x, y, sz, sz), theme.light_sq if is_light else theme.dark_sq)
            if effective_move and (br, bc) in effective_move:
                p.fillRect(QRect(x, y, sz, sz), theme.last_move)
            if selected and (br, bc) == selected:
                p.fillRect(QRect(x, y, sz, sz), theme.highlight)
            if (br, bc) in check_set:
                grad = QRadialGradient(x + sz / 2, y + sz / 2, sz * 0.7)
                grad.setColorAt(0, QColor(255, 30, 30, 180))
                grad.setColorAt(1, QColor(255, 0, 0, 0))
                p.setBrush(QBrush(grad))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(QRect(x, y, sz, sz))
            if legal_targets and (br, bc) in legal_targets:
                cx, cy = x + sz // 2, y + sz // 2
                if board.piece_at(sq) is not None:
                    p.setPen(QPen(QColor(0, 0, 0, 90), max(3, sz // 14)))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(cx - sz * 5 // 12, cy - sz * 5 // 12,
                                  sz * 10 // 12, sz * 10 // 12)
                else:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(0, 0, 0, 90))
                    p.drawEllipse(cx - sz // 6, cy - sz // 6, sz // 3, sz // 3)

        # 2. Draw static pieces (skips 'from' and 'to' squares if animating)
        for sq in chess.SQUARES:
            br, bc = 7 - chess.square_rank(sq), chess.square_file(sq)
            if (br, bc) in skip_sq:
                continue
            piece = board.piece_at(sq)
            if piece:
                sr, sc = b2s(br, bc)
                _draw_piece(p, piece, sr, sc, sz, font_piece)

        if not static_only:
            # 3. Draw captured piece fade (if animating)
            if anim_state and anim_state.get('captured', '.') != '.':
                bfr, bfc_ = anim_state['from']
                btr, btc_ = anim_state['to']
                cap_piece = _reconstruct_piece(anim_state['captured'])
                if cap_piece is not None:
                    fade = max(0, int(200 * (1.0 - anim_state['progress'])))
                    p.setOpacity(fade / 255.0)
                    sr, sc = b2s(btr, btc_)
                    _draw_piece(p, cap_piece, sr, sc, sz, font_piece)
                    p.setOpacity(1.0)

            # 4. Draw arrow
            if show_arrow and effective_move:
                (bfr, bfc), (btr, btc) = effective_move
                sfr, sfc = b2s(bfr, bfc)
                str_, stc = b2s(btr, btc)
                _draw_arrow(p, sfc * sz + sz // 2, sfr * sz + sz // 2,
                            stc * sz + sz // 2, str_ * sz + sz // 2,
                            theme.arrow_clr, sz, arrow_style)

            # 5. Draw animated piece
            if anim_state:
                bfr, bfc_ = anim_state['from']
                btr, btc_ = anim_state['to']
                t = anim_state['progress']
                anim_obj = anim_state.get('piece_obj')
                if anim_obj:
                    lift = 4.0 * t * (1.0 - t) * 0.20
                    scale = 1.0 + 4.0 * t * (1.0 - t) * 0.15

                    ir = bfr + (btr - bfr) * t
                    ic = bfc_ + (btc_ - bfc_) * t
                    sir, sic = b2s(ir, ic)

                    shadow_w = sz * 0.65 * scale
                    shadow_h = sz * 0.12 * scale
                    sa = 30 + int(70 * (lift / 0.20))

                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(0, 0, 0, sa))

                    piece_center_x = sic * sz + sz / 2
                    shadow_x = piece_center_x - shadow_w / 2 + (sz * 0.05 * scale)
                    shadow_y = sir * sz + sz * 0.85

                    p.drawEllipse(QRectF(shadow_x, shadow_y, shadow_w, shadow_h))

                    _draw_piece_at(p, anim_obj,
                                   (sir * sz - sz * lift) / sz, sic,
                                   sz, sz * scale, sz * scale, font_piece)

        # 6. Draw coordinates
        p.setFont(font_coord)
        cm = max(3, int(sz * 0.04))
        cs = max(12, sz // 5)
        for c in range(8):
            is_light = (7 + c) % 2 == 0
            p.setPen(theme.dark_sq if is_light else theme.light_sq)
            p.drawText(QRect(c * sz + sz - cs - cm, 7 * sz + cm, cs, cs),
                       Qt.AlignmentFlag.AlignCenter,
                       FILES_STR[7 - c if flipped else c])
        for r in range(8):
            is_light = r % 2 == 0
            p.setPen(theme.dark_sq if is_light else theme.light_sq)
            p.drawText(QRect(cm, r * sz + cm, cs, cs),
                       Qt.AlignmentFlag.AlignCenter,
                       RANKS_STR[7 - r if flipped else r])

    return img


# ═══════════════════════════════════════════════════════════════════════════════════
# BOARD WIDGET (from snippet — as-is)
# ═══════════════════════════════════════════════════════════════════════════════════

class BoardWidget(QWidget):
    move_made = Signal(dict)
    move_animated = Signal(dict)
    move_attempted = Signal(int, int, int, int, object)

    def __init__(self, engine: ChessEngine, theme_name="Classic", parent=None):
        super().__init__(parent)
        self.engine = engine
        self.theme_name = theme_name
        self.flipped = False
        self.selected = None
        self.legal_targets = []
        self.anim_timer = None
        self.anim_state = None
        self._arrow_style = ARROW_STYLE_THICK
        self.puzzle_mode = False
        self.setMinimumSize(BOARD_PX, BOARD_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    @property
    def theme(self):
        return THEMES.get(self.theme_name, THEMES["Classic"])

    def _sq_size(self):
        return min(self.width(), self.height()) // 8

    def _rc_from_pos(self, pos):
        sz = self._sq_size()
        col = pos.x() // sz
        row = pos.y() // sz
        if self.flipped:
            col = 7 - col
            row = 7 - row
        if 0 <= row < 8 and 0 <= col < 8:
            return row, col
        return None, None

    @property
    def arrow_style(self):
        return self._arrow_style

    @arrow_style.setter
    def arrow_style(self, value):
        self._arrow_style = value
        self.update()

    def paintEvent(self, event):
        with paint_context(self) as p:
            sz = self._sq_size()
            check_sqs = self.engine.check_squares()
            anim = self.anim_state
            img = render_board_image(
                self.engine.board, last_move=self.engine.last_move,
                selected=self.selected, legal_targets=self.legal_targets,
                check_squares=check_sqs, anim_state=anim,
                sq_size=sz, theme=self.theme, flipped=self.flipped,
                arrow_style=self.arrow_style)
            p.drawImage(0, 0, img.scaled(self.width(), self.height(),
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        r, c = self._rc_from_pos(event.position().toPoint() if hasattr(event, 'position') else event.pos())
        if r is None:
            return
        if self.selected:
            sr, sc = self.selected
            if (r, c) in self.legal_targets:
                self._execute_move(sr, sc, r, c)
                return
        piece = self.engine.board.piece_at(ChessEngine.rc_to_sq(r, c))
        if piece:
            self.selected = (r, c)
            self.legal_targets = self.engine.legal_moves(r, c)
        else:
            self.selected = None
            self.legal_targets = []
        self.update()

    def _execute_move(self, fr, fc, tr, tc, promo=None):
        if self.puzzle_mode:
            self.move_attempted.emit(fr, fc, tr, tc, promo)
            self.selected = None
            self.legal_targets = []
            self.update()
            return
        result = self.engine.make_move(fr, fc, tr, tc, promo)
        if result:
            self.selected = None
            self.legal_targets = []
            self._start_animation(result)
            snd = ('check' if result['check'] else
                   ('capture' if result['captured'] != '.' else
                    ('castle' if result['castle'] else
                     ('checkmate' if result['mate'] else 'move'))))
            self.move_made.emit(result)

    def play_external_move(self, uci_str):
        result = self.engine.make_move_uci(uci_str)
        if result:
            self._start_animation(result)
            snd = ('check' if result['check'] else
                   ('capture' if result['captured'] != '.' else 'move'))
            self.move_made.emit(result)

    def _start_animation(self, result, duration=None):
        if duration is None:
            duration = ANIM_SPEED_DEFAULT
        if self.anim_timer:
            self.anim_timer.stop()
        self.anim_state = {
            'from': result['from'], 'to': result['to'],
            'piece_obj': result['piece_obj'], 'captured': result['captured'],
            'progress': 0.0}
        steps = max(1, int(duration / 1000 * ANIM_FPS))
        self._anim_step = 0
        self._anim_total = steps
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._tick_animation)
        interval = max(1, int(duration / steps))
        self.anim_timer.start(interval)

    def _tick_animation(self):
        if not self.anim_state:
            self.anim_timer.stop()
            return
        self._anim_step += 1
        t = min(1.0, self._anim_step / self._anim_total)
        self.anim_state['progress'] = _ease_out_cubic(t)
        self.update()
        if t >= 1.0:
            self.anim_timer.stop()
            result_copy = dict(self.anim_state)
            self.anim_state = None
            self.update()
            self.move_animated.emit(result_copy)

    def reset_view(self):
        self.selected = None
        self.legal_targets = []
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════════
# VIEW-ONLY BOARD WIDGET — BoardWidget with ALL interactions disabled.
# Provides the API ChessPuzzleApp expects (set_board, highlight_*, animate_move, etc.)
# while rendering via render_board_image from the snippet.
# ═══════════════════════════════════════════════════════════════════════════════════

class ViewOnlyBoardWidget(BoardWidget):
    """
    BoardWidget subclass that:
      • Disables all mouse/keyboard interactions (view-only).
      • Exposes the API methods ChessPuzzleApp calls (set_board, set_position,
        clear_highlights, highlight_*, add_arrow, clear_arrows, animate_move,
        is_animating, _stop_animation, set_interaction_locked).
      • Uses render_board_image for all rendering.
      • Draws custom arrows and extra highlights on top of the rendered image.
    """

    def __init__(self, parent=None):
        engine = ChessEngine(chess.Board())
        super().__init__(engine, theme_name="Classic", parent=parent)
        self.puzzle_mode = False
        # Extra highlight state for ChessPuzzleApp compatibility
        self._hint_sq = None        # chess square index
        self._correct_sq = None
        self._wrong_sq = None
        self._custom_check_sq = None
        self._custom_arrows = []    # list of (from_sq, to_sq, QColor)

    # ─── Disable ALL mouse interactions ───
    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    # ─── Expose engine board for ChessPuzzleApp ───
    @property
    def board(self):
        return self.engine.board

    # ─── ChessPuzzleApp API: interaction lock (no-op in view-only mode) ───
    def set_interaction_locked(self, locked: bool):
        pass  # Always view-only — locking is a no-op

    # ─── ChessPuzzleApp API: set board from chess.Board ───
    def set_board(self, b: chess.Board):
        self.engine.board = b.copy()
        self.engine.last_move = None
        self.selected = None
        self.legal_targets = []
        self._stop_animation()
        self._hint_sq = None
        self._correct_sq = None
        self._wrong_sq = None
        self._custom_check_sq = None
        self.update()

    def set_position(self, fen: str):
        self.engine.board = chess.Board(fen)
        self.engine.last_move = None
        self.selected = None
        self.legal_targets = []
        self._stop_animation()
        self._hint_sq = None
        self._correct_sq = None
        self._wrong_sq = None
        self._custom_check_sq = None
        self.update()

    # ─── ChessPuzzleApp API: highlight management ───
    def clear_highlights(self):
        self.selected = None
        self.legal_targets = []
        self._hint_sq = None
        self._correct_sq = None
        self._wrong_sq = None
        self._custom_check_sq = None
        self.update()

    def highlight_last_move(self, f: int, t: int):
        """f and t are chess square indices."""
        fr = 7 - chess.square_rank(f)
        fc = chess.square_file(f)
        tr = 7 - chess.square_rank(t)
        tc = chess.square_file(t)
        self.engine.last_move = ((fr, fc), (tr, tc))
        self.update()

    def highlight_check(self, s: int):
        self._custom_check_sq = s
        self.update()

    def highlight_correct(self, s: int):
        self._correct_sq = s
        self.update()

    def highlight_wrong(self, s: int):
        self._wrong_sq = s
        self.update()

    def highlight_hint(self, s: int):
        self._hint_sq = s
        self.update()

    # ─── ChessPuzzleApp API: arrow management ───
    def add_arrow(self, from_sq: int, to_sq: int, color: QColor):
        self._custom_arrows.append((from_sq, to_sq, color))
        self.update()

    def clear_arrows(self):
        self._custom_arrows.clear()
        self.update()

    # ─── ChessPuzzleApp API: animation ───
    def animate_move(self, from_sq: int, to_sq: int, duration_ms: int = 350):
        fr = 7 - chess.square_rank(from_sq)
        fc = chess.square_file(from_sq)
        tr = 7 - chess.square_rank(to_sq)
        tc = chess.square_file(to_sq)
        piece = self.engine.board.piece_at(from_sq)
        captured_piece = self.engine.board.piece_at(to_sq)
        captured_char = captured_piece.symbol() if captured_piece else '.'

        result = {
            'from': (fr, fc), 'to': (tr, tc),
            'piece_obj': piece, 'captured': captured_char,
            'check': False, 'castle': False, 'mate': False,
            'progress': 0.0,
        }
        self._start_animation(result, duration_ms)

    def is_animating(self) -> bool:
        return self.anim_state is not None

    def _stop_animation(self):
        if self.anim_timer:
            self.anim_timer.stop()
        self.anim_state = None
        self.update()

    # ─── Override paintEvent to center board + draw custom arrows/highlights ───
    def paintEvent(self, event):
        with paint_context(self) as p:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

            sz = self._sq_size()
            bpx = sz * 8

            # Center board in widget
            ox = (self.width() - bpx) // 2
            oy = (self.height() - bpx) // 2

            # Fill background
            p.fillRect(self.rect(), QColor(20, 20, 30))

            # Draw frame
            frame_rect = QRect(ox - 6, oy - 6, bpx + 12, bpx + 12)
            p.setPen(QPen(QColor(80, 80, 100), 2))
            p.setBrush(QColor(40, 40, 50))
            p.drawRoundedRect(frame_rect, 6, 6)

            # Build check_squares (list of (row, col) tuples)
            check_sqs = []
            if self._custom_check_sq is not None:
                r = 7 - chess.square_rank(self._custom_check_sq)
                c = chess.square_file(self._custom_check_sq)
                check_sqs.append((r, c))
            elif self.engine.board.is_check():
                ks = self.engine.board.king(self.engine.board.turn)
                r = 7 - chess.square_rank(ks)
                c = chess.square_file(ks)
                check_sqs.append((r, c))

            # Determine selected square for highlight (hint > correct > wrong)
            selected = self.selected
            if selected is None:
                for sq_attr in [self._hint_sq, self._correct_sq, self._wrong_sq]:
                    if sq_attr is not None:
                        r = 7 - chess.square_rank(sq_attr)
                        c = chess.square_file(sq_attr)
                        selected = (r, c)
                        break

            anim = self.anim_state

            # Use render_board_image from snippet
            img = render_board_image(
                self.engine.board, last_move=self.engine.last_move,
                selected=selected, legal_targets=self.legal_targets,
                check_squares=check_sqs, anim_state=anim,
                sq_size=sz, theme=self.theme, flipped=self.flipped,
                arrow_style=self.arrow_style)
            p.drawImage(ox, oy, img)

            # Draw custom arrows on top (from ChessPuzzleApp.add_arrow)
            for from_sq, to_sq, color in self._custom_arrows:
                ff, fr_ = chess.square_file(from_sq), chess.square_rank(from_sq)
                tf, tr_ = chess.square_file(to_sq), chess.square_rank(to_sq)
                if self.flipped:
                    ff, fr_ = 7 - ff, 7 - fr_
                    tf, tr_ = 7 - tf, 7 - tr_
                x1 = ox + ff * sz + sz / 2
                y1 = oy + (7 - fr_) * sz + sz / 2
                x2 = ox + tf * sz + sz / 2
                y2 = oy + (7 - tr_) * sz + sz / 2
                _draw_arrow(p, x1, y1, x2, y2, color, sz)


# ═══════════════════════════════════════════════════════════════════════════════════
# THEME FILTER PANEL
# ═══════════════════════════════════════════════════════════════════════════════════

class ThemeFilterPanel(QScrollArea):
    theme_changed = Signal()
    generate_requested = Signal()
    
    def __init__(self,parent=None):
        super().__init__(parent); self._tc={}; self._cc={}
        self.setWidgetResizable(True); self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMaximumWidth(280); self.setMinimumWidth(220); self._build()
        
    def _build(self):
        c=QWidget(); l=QVBoxLayout(c); l.setSpacing(8); l.setContentsMargins(8,8,8,8)
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search themes...")
        self.search_edit.textChanged.connect(self._filter_themes)
        l.addWidget(self.search_edit)
        
        cg=QGroupBox("Categories"); cl=QVBoxLayout(cg); cl.setSpacing(4)
        for cat in CATEGORIES:
            cb=QCheckBox(cat); cb.setChecked(True); cb.stateChanged.connect(self._oc); self._cc[cat]=cb; cl.addWidget(cb)
        br=QHBoxLayout(); b1=QPushButton("Select All"); b1.setFixedHeight(28); b1.clicked.connect(lambda:self._sa(True))
        b2=QPushButton("Deselect All"); b2.setFixedHeight(28); b2.clicked.connect(lambda:self._sa(False))
        br.addWidget(b1); br.addWidget(b2); cl.addLayout(br); l.addWidget(cg)
        
        tg=QGroupBox("Themes"); tl=QVBoxLayout(tg); tl.setSpacing(2)
        self._cat_labels = {}
        for cat in CATEGORIES:
            lbl = QLabel(f"<b style='color:#89b4fa;'>{cat}</b>")
            tl.addWidget(lbl)
            self._cat_labels[cat] = lbl
            for t in get_themes_by_category(cat):
                cb=QCheckBox(f"  {t.icon_hint}  {t.display_name}"); cb.setChecked(True)
                cb.stateChanged.connect(self._ot); cb.setToolTip(t.description)
                self._tc[t.name]=(cb,t.category); tl.addWidget(cb)
        l.addWidget(tg); l.addStretch()
        
        gen_btn = QPushButton("  \u25b6 Generate Puzzles  ")
        gen_btn.setObjectName("primaryBtn")
        gen_btn.clicked.connect(self.generate_requested.emit)
        l.addWidget(gen_btn)
        
        self.setWidget(c)
        
    def _filter_themes(self, text):
        text = text.lower().strip()
        if not text:
            for cb, _ in self._tc.values(): cb.setVisible(True)
            for lbl in self._cat_labels.values(): lbl.setVisible(True)
            return
            
        for tk, (cb, cat) in self._tc.items():
            theme_info = THEME_REGISTRY[tk]
            match = text in theme_info.display_name.lower() or text in theme_info.name.lower()
            cb.setVisible(match)
            
        for cat, lbl in self._cat_labels.items():
            any_visible = any(cb.isVisible() for tk, (cb, c) in self._tc.items() if c == cat)
            lbl.setVisible(any_visible)

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
    
    def set_selected_themes(self, themes):
        for tk, (cb, _) in self._tc.items():
            cb.blockSignals(True)
            cb.setChecked(tk in themes)
            cb.blockSignals(False)
        for cat, cb in self._cc.items():
            cb.blockSignals(True)
            all_selected = all((tk in themes) for tk, (c, c_cat) in self._tc.items() if c_cat == cat)
            cb.setChecked(all_selected)
            cb.blockSignals(False)

# ═══════════════════════════════════════════════════════════════════════════════════
# ASYNC WORKERS
# ═══════════════════════════════════════════════════════════════════════════════════

class PuzzleGeneratorWorker(QThread):
    finished = Signal(list)
    progress = Signal(int, int, str)

    def __init__(self, generator, difficulty_range, count, themes, avoid_fens=None):
        super().__init__()
        self.generator = generator
        self.difficulty_range = difficulty_range
        self.count = count
        self.themes = themes
        self.avoid_fens = avoid_fens
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        logger.info(f"Puzzle generation worker started: count={self.count}, range={self.difficulty_range}, themes={len(self.themes)}")
        self.progress.emit(0, self.count, "Generating puzzles...")
        
        generated_puzzles = []
        try:
            generated_puzzles = self.generator.generate(
                theme_keys=self.themes,
                difficulty_range=self.difficulty_range,
                count=self.count,
                avoid_fens=self.avoid_fens if self.avoid_fens is not None else set(),
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._is_cancelled
            )
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            
        self.progress.emit(len(generated_puzzles), self.count, "Done")
        self.finished.emit(generated_puzzles)

class SingleVideoExporter(QThread):
    finished_export = Signal(str)
    error = Signal(str)

    def __init__(self, puzzle, filepath, assets_dir, enable_sfx=True, bgm_path=None):
        super().__init__()
        self.puzzle = puzzle
        self.filepath = filepath
        self.assets_dir = assets_dir
        self.enable_sfx = enable_sfx
        self.bgm_path = bgm_path

    def run(self):
        try:
            renderer = VideoRenderer(self.filepath)
            renderer.render(self.puzzle, self.assets_dir, enable_sfx=self.enable_sfx, bgm_path=self.bgm_path)
            self.finished_export.emit(self.filepath)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY COLOR SYSTEM & PREVIEW HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "Attack":     QColor(243, 139, 168),
    "Checkmate":  QColor(249, 226, 175),
    "Sacrifice":  QColor(245, 194, 99),
    "Tactic":     QColor(137, 180, 250),
    "Endgame":    QColor(166, 227, 161),
}

def category_color(cat: str) -> QColor:
    return CATEGORY_COLORS.get(cat, QColor(180, 180, 180))

def material_balance(board: chess.Board) -> Tuple[int, Dict[int, int]]:
    PV_LOCAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    start = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
    counts_w = {pt: 0 for pt in start}
    counts_b = {pt: 0 for pt in start}
    for sq, pc in board.piece_map().items():
        if pc.piece_type == chess.KING: continue
        if pc.color == chess.WHITE: counts_w[pc.piece_type] += 1
        else:                       counts_b[pc.piece_type] += 1
    balance = 0
    for pt in start:
        balance += PV_LOCAL[pt] * (counts_w[pt] - counts_b[pt])
    return balance, {"white": counts_w, "black": counts_b, "start": start}

# ═══════════════════════════════════════════════════════════════════════════════════
# MINI BOARD PREVIEW WIDGET
# ═══════════════════════════════════════════════════════════════════════════════════

class MiniBoardPreview(QWidget):
    clicked = Signal(int)

    def __init__(self, index: int, puzzle: Puzzle, size: int = 120, parent=None):
        super().__init__(parent)
        self.index = index
        self.puzzle = puzzle
        self.target_size = size
        self.setFixedSize(size + 8, size + 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False
        self._selected = False
        self._board = chess.Board(puzzle.fen)
        self._sq = size // 8
        self._offset_x = 4
        self._offset_y = 4

    def set_selected(self, sel: bool):
        self._selected = sel
        self.update()

    def enterEvent(self, e):
        self._hovered = True
        self.update()

    def leaveEvent(self, e):
        self._hovered = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor(24, 24, 37) if not self._selected else QColor(49, 50, 68)
        if self._hovered and not self._selected:
            bg = QColor(40, 40, 60)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        cat_c = category_color(self.puzzle.category)
        if self._selected:
            p.setPen(QPen(cat_c, 2))
        elif self._hovered:
            p.setPen(QPen(cat_c.lighter(120), 1))
        else:
            p.setPen(QPen(QColor(69, 71, 90), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        light, dark = QColor(240, 217, 181), QColor(181, 136, 99)
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x = self._offset_x + f * self._sq
            y = self._offset_y + (7 - r) * self._sq
            p.fillRect(QRect(x, y, self._sq, self._sq), light if (f + r) % 2 == 0 else dark)

        font = QFont("Segoe UI Symbol", max(8, int(self._sq * 0.75)))
        font.setBold(True)
        p.setFont(font)
        for sq in chess.SQUARES:
            pc = self._board.piece_at(sq)
            if pc is None: continue
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x = self._offset_x + f * self._sq
            y = self._offset_y + (7 - r) * self._sq
            t = PIECE_UNICODE.get(pc.piece_type, {}).get(pc.color, "?")
            p.setPen(QColor(0, 0, 0, 80))
            p.drawText(QRect(x + 1, y + 1, self._sq, self._sq), Qt.AlignmentFlag.AlignCenter, t)
            p.setPen(QColor(255, 255, 255) if pc.color == chess.WHITE else QColor(30, 30, 30))
            p.drawText(QRect(x, y, self._sq, self._sq), Qt.AlignmentFlag.AlignCenter, t)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cat_c)
        p.drawRect(QRect(self._offset_x, self._offset_y, self.target_size, 3))

        label_font = QFont("Segoe UI", 8, QFont.Bold)
        p.setFont(label_font)
        stars = FILLED_STAR * self.puzzle.difficulty + EMPTY_STAR * (10 - self.puzzle.difficulty)
        p.setPen(QColor(249, 226, 175))
        p.drawText(QRect(0, self._offset_y + self.target_size + 2, self.width(), 12),
                   Qt.AlignmentFlag.AlignCenter, stars)

        dot_color = QColor(255, 255, 255) if self._board.turn == chess.WHITE else QColor(30, 30, 30)
        p.setBrush(dot_color)
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.drawEllipse(QPoint(self._offset_x + self.target_size - 8, self._offset_y - 2), 4, 4)

# ═══════════════════════════════════════════════════════════════════════════════════
# PUZZLE QUEUE PANEL
# ═══════════════════════════════════════════════════════════════════════════════════

class PuzzleQueuePanel(QScrollArea):
    puzzle_selected = Signal(int)
    request_more = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(180)
        self.setMaximumWidth(220)
        self._previews: List[MiniBoardPreview] = []
        self._current_idx = -1
        self._build()

    def _build(self):
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("\u2b50 Puzzle Queue")
        header.setStyleSheet("font-weight:bold;color:#cdd6f4;font-size:13px;padding:4px;")
        self._layout.addWidget(header)

        self._count_label = QLabel("0 puzzles")
        self._count_label.setStyleSheet("color:#a6adc8;font-size:11px;padding:0 4px 6px;")
        self._layout.addWidget(self._count_label)

        self._previews_layout = QVBoxLayout()
        self._previews_layout.setSpacing(6)
        self._layout.addLayout(self._previews_layout)

        self._more_btn = QPushButton("+ Generate More")
        self._more_btn.setObjectName("primaryBtn")
        self._more_btn.setFixedHeight(32)
        self._more_btn.clicked.connect(self.request_more.emit)
        self._layout.addWidget(self._more_btn)

        self._layout.addStretch()
        self.setWidget(self._container)

    def set_puzzles(self, puzzles: List[Puzzle]):
        for p in self._previews:
            p.setParent(None)
            p.deleteLater()
        self._previews.clear()

        for i, puzzle in enumerate(puzzles):
            thumb = MiniBoardPreview(i, puzzle, size=140)
            thumb.clicked.connect(self._on_preview_clicked)
            self._previews_layout.addWidget(thumb)
            self._previews.append(thumb)

        self._count_label.setText(f"{len(puzzles)} puzzles queued")
        if puzzles:
            self.set_current_index(0)

    def set_current_index(self, idx: int):
        self._current_idx = idx
        for i, thumb in enumerate(self._previews):
            thumb.set_selected(i == idx)
        if 0 <= idx < len(self._previews):
            self.ensureWidgetVisible(self._previews[idx])

    def _on_preview_clicked(self, idx: int):
        self.puzzle_selected.emit(idx)

    def clear(self):
        for p in self._previews:
            p.setParent(None)
            p.deleteLater()
        self._previews.clear()
        self._count_label.setText("0 puzzles")

    def set_more_enabled(self, enabled: bool):
        self._more_btn.setEnabled(enabled)

    def scroll_to_index(self, idx: int):
        if 0 <= idx < len(self._previews):
            self.ensureWidgetVisible(self._previews[idx])

# ═══════════════════════════════════════════════════════════════════════════════════
# SOLUTION PREVIEW WIDGET
# ═══════════════════════════════════════════════════════════════════════════════════

class SolutionPreviewWidget(QWidget):
    move_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._moves_san: List[str] = []
        self._current_step = -1
        self._revealed_up_to = 0
        self.setMinimumHeight(80)

    def set_solution(self, fen: str, uci_moves: List[str]):
        board = chess.Board(fen)
        self._moves_san = []
        for uci in uci_moves:
            try:
                m = chess.Move.from_uci(uci)
                san = board.san(m)
                self._moves_san.append(san)
                board.push(m)
            except Exception:
                self._moves_san.append(uci)
        self._current_step = -1
        self._revealed_up_to = 0
        self.update()

    def reveal_step(self, step: int):
        self._revealed_up_to = max(self._revealed_up_to, step + 1)
        self._current_step = step
        self.update()

    def reveal_all(self):
        self._revealed_up_to = len(self._moves_san)
        self.update()

    def clear(self):
        self._moves_san = []
        self._current_step = -1
        self._revealed_up_to = 0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 37))

        if not self._moves_san:
            p.setPen(QColor(108, 112, 134))
            font = QFont("Segoe UI", 10)
            font.setItalic(True)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Solution preview will appear here...")
            return

        font = QFont("Consolas", 10, QFont.Bold)
        p.setFont(font)
        y = 10
        x = 8
        max_x = self.width() - 8
        line_h = 22

        for i, san in enumerate(self._moves_san):
            revealed = i < self._revealed_up_to
            is_current = i == self._current_step

            text = f"{i+1}. " if i % 2 == 0 else ""
            display = san if revealed else "???"
            move_text = f"{text}{display}"

            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(move_text)

            if x + tw > max_x:
                x = 8
                y += line_h

            if is_current:
                p.fillRect(QRect(x - 4, y - 2, tw + 8, line_h - 2), QColor(137, 180, 250, 40))
                p.setPen(QColor(137, 180, 250))
            elif revealed:
                p.setPen(QColor(205, 214, 244) if i % 2 == 0 else QColor(180, 180, 200))
            else:
                p.setPen(QColor(80, 80, 90))

            p.drawText(QRect(x, y, tw + 8, line_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, move_text)
            x += tw + 12

# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED PUZZLE INFO PANEL
# ═══════════════════════════════════════════════════════════════════════════════════

class EnhancedInfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._puzzle: Optional[Puzzle] = None
        self._board: Optional[chess.Board] = None
        self.setMinimumWidth(260)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        info_group = QGroupBox("Puzzle Info")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(6)

        badge_row = QHBoxLayout()
        self._theme_label = QLabel("Theme: --")
        self._theme_label.setObjectName("statLabel")
        self._theme_label.setWordWrap(True)
        badge_row.addWidget(self._theme_label, stretch=1)

        self._category_badge = QLabel("--")
        self._category_badge.setFixedHeight(22)
        self._category_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._category_badge.setStyleSheet("QLabel{color:#1e1e2e;font-weight:bold;font-size:11px;padding:2px 10px;border-radius:11px;background:#89b4fa;}")
        badge_row.addWidget(self._category_badge)
        info_layout.addLayout(badge_row)

        self._desc = QTextEdit()
        self._desc.setReadOnly(True)
        self._desc.setMaximumHeight(110)
        self._desc.setPlaceholderText("No puzzle loaded...")
        info_layout.addWidget(self._desc)

        meta_row = QHBoxLayout()

        self._difficulty_label = QLabel("\u2605" * 5)
        self._difficulty_label.setStyleSheet("color:#f9e2af;font-size:12px;")
        meta_row.addWidget(self._difficulty_label)

        meta_row.addStretch()

        self._move_count_label = QLabel("\u2694 0 moves")
        self._move_count_label.setStyleSheet("color:#a6adc8;font-size:11px;")
        meta_row.addWidget(self._move_count_label)

        meta_row.addStretch()

        self._side_label = QLabel("\u265f Black to move")
        self._side_label.setStyleSheet("color:#cdd6f4;font-size:11px;font-weight:bold;")
        meta_row.addWidget(self._side_label)

        info_layout.addLayout(meta_row)

        self._material_bar = QWidget()
        self._material_bar.setFixedHeight(28)
        self._material_bar_layout = QHBoxLayout(self._material_bar)
        self._material_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._material_bar_layout.setSpacing(2)
        self._white_captures = QLabel("")
        self._white_captures.setStyleSheet("color:#cdd6f4;font-size:14px;")
        self._balance_label = QLabel("0")
        self._balance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._balance_label.setStyleSheet("font-weight:bold;color:#a6e3a1;font-size:12px;")
        self._black_captures = QLabel("")
        self._black_captures.setStyleSheet("color:#cdd6f4;font-size:14px;")
        self._material_bar_layout.addWidget(self._white_captures)
        self._material_bar_layout.addWidget(self._balance_label, stretch=1)
        self._material_bar_layout.addWidget(self._black_captures)
        info_layout.addWidget(self._material_bar)

        layout.addWidget(info_group)

        sol_group = QGroupBox("Solution Preview")
        sol_layout = QVBoxLayout(sol_group)
        self._solution_preview = SolutionPreviewWidget()
        self._solution_preview.setMinimumHeight(100)
        sol_layout.addWidget(self._solution_preview)
        layout.addWidget(sol_group)

        hist_group = QGroupBox("Move History")
        hist_layout = QVBoxLayout(hist_group)
        self._history = QTextEdit()
        self._history.setReadOnly(True)
        self._history.setMaximumHeight(120)
        self._history.setPlaceholderText("Moves will appear here during animation...")
        hist_layout.addWidget(self._history)
        layout.addWidget(hist_group)

        layout.addStretch()

    def set_puzzle(self, puzzle: Puzzle):
        self._puzzle = puzzle
        ti = THEME_REGISTRY.get(puzzle.theme_key)
        self._theme_label.setText(f"{ti.icon_hint if ti else ''}  {puzzle.theme_name}")

        cat_c = category_color(puzzle.category)
        self._category_badge.setText(puzzle.category)
        self._category_badge.setStyleSheet(
            f"QLabel{{color:#1e1e2e;font-weight:bold;font-size:11px;"
            f"padding:2px 10px;border-radius:11px;background:rgb({cat_c.red()},{cat_c.green()},{cat_c.blue()});}}"
        )

        html = f"<h3 style='color:#89b4fa;margin:0 0 6px 0;'>{puzzle.theme_name}</h3>"
        html += f"<p style='margin:2px 0;'>{puzzle.description}</p>"
        if ti:
            html += f"<p style='margin:2px 0;color:#a6adc8;font-size:11px;'>{ti.description}</p>"
        html += f"<p style='margin:6px 0 0 0;color:#a6adc8;font-size:10px;'>Difficulty: {puzzle.difficulty}/10</p>"
        self._desc.setHtml(html)

        self._difficulty_label.setText(FILLED_STAR * puzzle.difficulty + EMPTY_STAR * (10 - puzzle.difficulty))
        self._move_count_label.setText(f"\u2694 {len(puzzle.moves)} move{'s' if len(puzzle.moves) != 1 else ''}")

        board = chess.Board(puzzle.fen)
        self._board = board
        if board.turn == chess.WHITE:
            self._side_label.setText("\u2654 White to move")
            self._side_label.setStyleSheet("color:#f9e2af;font-size:11px;font-weight:bold;")
        else:
            self._side_label.setText("\u265a Black to move")
            self._side_label.setStyleSheet("color:#cdd6f4;font-size:11px;font-weight:bold;")

        balance, counts = material_balance(board)
        self._balance_label.setText(f"{balance:+d}" if balance != 0 else "0")
        if balance > 0:
            self._balance_label.setStyleSheet("font-weight:bold;color:#a6e3a1;font-size:12px;")
        elif balance < 0:
            self._balance_label.setStyleSheet("font-weight:bold;color:#f38ba8;font-size:12px;")
        else:
            self._balance_label.setStyleSheet("font-weight:bold;color:#a6adc8;font-size:12px;")

        w_caps = []
        b_caps = []
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            missing_w = counts["start"][pt] - counts["white"][pt]
            missing_b = counts["start"][pt] - counts["black"][pt]
            w_caps.append(PIECE_UNICODE[pt][chess.BLACK] * missing_w)
            b_caps.append(PIECE_UNICODE[pt][chess.WHITE] * missing_b)
        self._white_captures.setText("".join(w_caps))
        self._black_captures.setText("".join(b_caps))

        self._solution_preview.set_solution(puzzle.fen, puzzle.moves)
        self._history.clear()

    def append_history(self, san: str, correct: bool):
        color = "#a6e3a1" if correct else "#f38ba8"
        icon = "\u2713" if correct else "\u274c"
        self._history.append(f"<span style='color:{color};'>{icon} {san}</span>")

    def reveal_solution_step(self, step: int):
        self._solution_preview.reveal_step(step)

    def reveal_full_solution(self):
        self._solution_preview.reveal_all()

    def clear_solution(self):
        self._solution_preview.clear()

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW — VIEW-ONLY MODE, NO PUZZLE-TRYING INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════════

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
        self.setWindowTitle("Chess Puzzle Viewer — Procedural Generation & Video Export")
        self.setMinimumSize(1400, 800)
        self.resize(1500, 900)

        self.generator = ProceduralGenerator()
        self.current_puzzle: Optional[Puzzle] = None
        self.puzzle_queue: List[Puzzle] = []
        self.puzzle_index = 0
        self.worker = None
        self.gen_dlg = None
        self.batch_proc = None

        # Animation state
        self._animating = False
        self._anim_move_idx = 0
        self._anim_phase = "idle"
        self._anim_board: Optional[chess.Board] = None
        self._anim_timer = QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._advance_animation)

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        self._load_settings()

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        ml = QHBoxLayout(c); ml.setSpacing(0); ml.setContentsMargins(0,0,0,0)
        
        self.filter_panel = ThemeFilterPanel()
        self.filter_panel.theme_changed.connect(self._on_filter)
        self.filter_panel.generate_requested.connect(self._new)
        ml.addWidget(self.filter_panel)

        cw = QWidget(); cl = QVBoxLayout(cw); cl.setSpacing(8); cl.setContentsMargins(12,12,12,12)
        
        info_bar = QHBoxLayout()
        self.theme_label = QLabel("Theme: --")
        self.theme_label.setObjectName("statLabel")
        self.diff_label = QLabel("Difficulty: --")
        self.diff_label.setObjectName("statLabel")
        self.cat_label = QLabel("Category: --")
        self.cat_label.setObjectName("statLabel")
        self.prog_label = QLabel("Puzzle 0/0")
        info_bar.addWidget(self.theme_label)
        info_bar.addSpacing(20)
        info_bar.addWidget(self.cat_label)
        info_bar.addSpacing(20)
        info_bar.addWidget(self.diff_label)
        info_bar.addStretch()
        info_bar.addWidget(self.prog_label)
        cl.addLayout(info_bar)

        self.pbar = QProgressBar(); self.pbar.setFixedHeight(6); self.pbar.setTextVisible(False); cl.addWidget(self.pbar)

        board_frame = QFrame()
        board_frame.setStyleSheet("QFrame{background-color:#181825;border-radius:8px;padding:8px}")
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(8,8,8,8)
        self.board = ViewOnlyBoardWidget()
        # NOTE: move_made signal is NOT connected — no puzzle-trying interaction
        board_layout.addWidget(self.board, alignment=Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(board_frame, stretch=1)

        self.msg = QLabel("Click 'New Puzzle' to generate procedural puzzles!")
        self.msg.setWordWrap(True)
        self.msg.setMinimumHeight(36)
        cl.addWidget(self.msg)

        ctrls = QHBoxLayout()
        self.prev_btn = QPushButton("  \u25c0  Previous  ")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QPushButton("  Next  \u25b6  ")
        self.next_btn.clicked.connect(self._next)
        self.new_btn = QPushButton("  New Puzzle  ")
        self.new_btn.setObjectName("primaryBtn")
        self.new_btn.clicked.connect(self._new)
        
        anim_layout = QHBoxLayout()
        self.anim_btn = QPushButton("  \u25b6 Animate Solution  ")
        self.anim_btn.setObjectName("warningBtn")
        self.anim_btn.clicked.connect(self._toggle_animation)
        anim_layout.addWidget(self.anim_btn)
        
        anim_layout.addWidget(QLabel("Speed:"))
        self.anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.anim_speed_slider.setMinimum(25)
        self.anim_speed_slider.setMaximum(200)
        self.anim_speed_slider.setValue(100)
        self.anim_speed_slider.setFixedWidth(100)
        self.anim_speed_slider.valueChanged.connect(self._update_anim_speed_label)
        anim_layout.addWidget(self.anim_speed_slider)
        
        self.anim_speed_label = QLabel("1.00x")
        self.anim_speed_label.setFixedWidth(40)
        anim_layout.addWidget(self.anim_speed_label)
        
        self.sol_btn = QPushButton("  Show Solution  ")
        self.sol_btn.setObjectName("dangerBtn")
        self.sol_btn.clicked.connect(self._show_sol)
        
        self.reset_btn = QPushButton("  Reset Board  ")
        self.reset_btn.clicked.connect(self._reset)
        
        for b in [self.prev_btn, self.reset_btn]:
            ctrls.addWidget(b)
        ctrls.addLayout(anim_layout)
        for b in [self.new_btn, self.sol_btn, self.next_btn]:
            ctrls.addWidget(b)
        cl.addLayout(ctrls)

        dr = QHBoxLayout()
        dr.addWidget(QLabel("Difficulty Range:"))
        self.dmin_l = QLabel("1"); self.dmin_l.setFixedWidth(16)
        self.dmax_l = QLabel("10"); self.dmax_l.setFixedWidth(20)
        self.ds1 = QSlider(Qt.Orientation.Horizontal); self.ds1.setRange(1,10); self.ds1.setValue(1)
        self.ds1.valueChanged.connect(self._ddiff); self._dmin = 1
        self.ds2 = QSlider(Qt.Orientation.Horizontal); self.ds2.setRange(1,10); self.ds2.setValue(10)
        self.ds2.valueChanged.connect(self._ddiff); self._dmax = 10
        dr.addWidget(self.dmin_l); dr.addWidget(self.ds1); dr.addWidget(QLabel("-")); dr.addWidget(self.ds2); dr.addWidget(self.dmax_l)
        cl.addLayout(dr)
        
        ml.addWidget(cw, stretch=1)

        self.queue_panel = PuzzleQueuePanel()
        self.queue_panel.puzzle_selected.connect(self._on_queue_select)
        self.queue_panel.request_more.connect(lambda: self._gen(cnt=20, append=True))
        ml.addWidget(self.queue_panel)

        self.info_panel = EnhancedInfoPanel()
        self.info_panel.setMaximumWidth(320)
        self.info_panel.setMinimumWidth(280)
        ml.addWidget(self.info_panel)

    def _build_toolbar(self):
        tb = QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        a=QAction("New Puzzle",self); a.setShortcut("Ctrl+N"); a.triggered.connect(self._new); tb.addAction(a); tb.addSeparator()
        a=QAction("Reset Board",self); a.setShortcut("Ctrl+R"); a.triggered.connect(self._reset); tb.addAction(a); tb.addSeparator()
        a=QAction("Random Puzzle",self); a.setShortcut("Ctrl+Shift+N"); a.triggered.connect(self._rand); tb.addAction(a); tb.addSeparator()
        a=QAction("Flip Board",self); a.setShortcut("Ctrl+F"); a.triggered.connect(self._flip); tb.addAction(a); tb.addSeparator()
        a=QAction("Copy FEN",self); a.setShortcut("Ctrl+C"); a.triggered.connect(self._copy_fen); tb.addAction(a)
        a=QAction("Paste FEN",self); a.setShortcut("Ctrl+V"); a.triggered.connect(self._paste_fen); tb.addAction(a); tb.addSeparator()
        a=QAction("Export Current to MP4",self); a.setShortcut("Ctrl+E"); a.triggered.connect(self._export_current); tb.addAction(a)
        a=QAction("Export to PGN",self); a.setShortcut("Ctrl+G"); a.triggered.connect(self._export_pgn); tb.addAction(a); tb.addSeparator()
        a=QAction("Batch YouTube Shorts",self); a.setShortcut("Ctrl+B"); a.triggered.connect(self._batch_gen); tb.addAction(a)

    def _build_statusbar(self):
        status_msg = "Ready. View-only mode — puzzles are displayed, not interactive."
        if NUMBA_ENABLED: status_msg += " (Numba JIT Enabled)"
        self.statusBar().showMessage(status_msg)

    def _ddiff(self):
        self._dmin=min(self.ds1.value(),self.ds2.value()); self._dmax=max(self.ds1.value(),self.ds2.value())
        self.dmin_l.setText(str(self._dmin)); self.dmax_l.setText(str(self._dmax))

    def _on_filter(self): pass

    def _save_settings(self):
        settings = {
            "themes": self.filter_panel.get_selected_themes(),
            "dmin": self._dmin,
            "dmax": self._dmax
        }
        try:
            with open("chess_settings.json", "w") as f:
                json.dump(settings, f)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def _load_settings(self):
        try:
            if os.path.exists("chess_settings.json"):
                with open("chess_settings.json", "r") as f:
                    settings = json.load(f)
                    themes = settings.get("themes", [])
                    if themes:
                        self.filter_panel.set_selected_themes(themes)
                    dmin = settings.get("dmin", 1)
                    dmax = settings.get("dmax", 10)
                    self.ds1.setValue(dmin)
                    self.ds2.setValue(dmax)
                    self._ddiff()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def _gen(self, cnt=20, append=False):
        themes = self.filter_panel.get_selected_themes()
        if not themes:
            self.msg.setText("\u26a0 No themes selected!")
            self.msg.setObjectName("errorMessage")
            self.msg.setStyleSheet("")
            return

        self._save_settings()
        existing_fens = set(p.fen for p in self.puzzle_queue) if append else None

        self.msg.setText("\u23f3 Generating puzzles in background...")
        self.msg.setStyleSheet("")

        self.new_btn.setEnabled(False)
        self.queue_panel.set_more_enabled(False)

        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)

        self.gen_dlg = QProgressDialog("Generating puzzles...", "Cancel", 0, cnt, self)
        self.gen_dlg.setWindowTitle("Generating Puzzles")
        self.gen_dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.gen_dlg.setMinimumDuration(0)
        self.gen_dlg.setAutoClose(False)
        self.gen_dlg.setValue(0)
        self.gen_dlg.show()

        self._gen_append_mode = append

        self.worker = PuzzleGeneratorWorker(
            self.generator, (self._dmin, self._dmax), cnt, themes, existing_fens
        )
        self.worker.progress.connect(self._gen_progress)
        self.worker.finished.connect(self._on_gen_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.gen_dlg.canceled.connect(self._on_gen_cancelled)
        self.worker.start()

    def _gen_progress(self, current, total, msg):
        if self.gen_dlg:
            self.gen_dlg.setMaximum(total)
            self.gen_dlg.setValue(current)
            self.gen_dlg.setLabelText(msg)

    def _on_gen_cancelled(self):
        if self.worker:
            self.worker.cancel()
            self.worker.wait(2000)
            
        if self.gen_dlg:
            self.gen_dlg.close()
            self.gen_dlg = None
            
        self.new_btn.setEnabled(True)
        self.queue_panel.set_more_enabled(True)
        self.msg.setText("Generation cancelled.")
        self.msg.setStyleSheet("")

    def _on_gen_finished(self, puzzle_queue):
        if self.gen_dlg:
            self.gen_dlg.close()
            self.gen_dlg = None

        self.new_btn.setEnabled(True)
        self.queue_panel.set_more_enabled(True)

        if not puzzle_queue:
            self.msg.setText("\u26a0 No puzzles generated. Try broadening filters.")
            self.msg.setObjectName("errorMessage")
            self.msg.setStyleSheet("")
            return

        if getattr(self, '_gen_append_mode', False) and self.puzzle_queue:
            old_count = len(self.puzzle_queue)
            self.puzzle_queue.extend(puzzle_queue)
            self.queue_panel.set_puzzles(self.puzzle_queue)
            self.queue_panel.set_current_index(self.puzzle_index)
            self.queue_panel.scroll_to_index(old_count)
            self._upd_info()
            self._upd_nav()
            self._setmsg(f"Added {len(puzzle_queue)} new puzzles! Total: {len(self.puzzle_queue)}")
        else:
            self.puzzle_queue = puzzle_queue
            self.puzzle_index = 0
            self.queue_panel.set_puzzles(puzzle_queue)
            self._load(0)

    def _on_queue_select(self, idx: int):
        if 0 <= idx < len(self.puzzle_queue):
            self._load(idx)

    def _load(self, idx: int):
        if self._animating:
            self._stop_animation()

        if 0 <= idx < len(self.puzzle_queue):
            self.current_puzzle = self.puzzle_queue[idx]
            self.puzzle_index = idx
            self.board.set_interaction_locked(False)
            self.board.set_position(self.current_puzzle.fen)
            
            # Auto-flip board to side to move
            self.board.flipped = (self.board.board.turn == chess.BLACK)
            self.board.update()
            
            self.board.clear_highlights()
            self.board.clear_arrows()
            self.info_panel.set_puzzle(self.current_puzzle)
            self.queue_panel.set_current_index(idx)
            self._upd_info()
            self._upd_nav()
            self._setmsg(f"Viewing puzzle: {self.current_puzzle.theme_name}. Use 'Animate Solution' or 'Show Solution' to reveal the answer.")
            self.reset_btn.setEnabled(True)
            self.sol_btn.setEnabled(True)
            self.anim_btn.setEnabled(True)
            self.statusBar().showMessage(
                f"Puzzle {idx+1}/{len(self.puzzle_queue)} | "
                f"Theme: {self.current_puzzle.theme_name} | "
                f"Difficulty: {self.current_puzzle.difficulty}/10"
            )

    def _upd_info(self):
        if not self.current_puzzle: return
        p = self.current_puzzle
        self.theme_label.setText(f"Theme: {p.theme_name}")
        self.diff_label.setText(f"Difficulty: {FILLED_STAR * p.difficulty}{EMPTY_STAR * (10 - p.difficulty)}")
        self.cat_label.setText(f"Category: {p.category}")
        self.prog_label.setText(f"Puzzle {self.puzzle_index + 1}/{len(self.puzzle_queue)}")
        self.pbar.setMaximum(len(self.puzzle_queue))
        self.pbar.setValue(self.puzzle_index + 1)

    def _upd_nav(self):
        self.prev_btn.setEnabled(self.puzzle_index > 0)
        self.next_btn.setEnabled(self.puzzle_index < len(self.puzzle_queue) - 1)

    def _setmsg(self, t):
        self.msg.setText(t)
        self.msg.setObjectName("")
        self.msg.setStyleSheet("")

    def _new(self):
        self._gen()

    def _next(self):
        if self.puzzle_index < len(self.puzzle_queue) - 1:
            self._load(self.puzzle_index + 1)
        else:
            self.statusBar().showMessage("Queue finished. Generating new puzzles...")
            self._gen()

    def _prev(self):
        if self.puzzle_index > 0:
            self._load(self.puzzle_index - 1)

    def _rand(self):
        p = self.generator.get_random_puzzle(difficulty_range=(self._dmin, self._dmax))
        if p:
            self.puzzle_queue = [p]
            self.puzzle_index = 0
            self.queue_panel.set_puzzles([p])
            self._load(0)
            self.statusBar().showMessage(f"Random: {p.theme_name} (Difficulty {p.difficulty}/10)")

    def _reset(self):
        if self._animating:
            self._stop_animation()
            return
            
        if self.current_puzzle:
            self.board.set_position(self.current_puzzle.fen)
            self.board.clear_highlights()
            self.board.clear_arrows()
            self.info_panel.set_puzzle(self.current_puzzle)
            self._setmsg("Board reset to starting position.")
            self.msg.setStyleSheet("")

    def _show_sol(self):
        if not self.current_puzzle: return
        if self._animating:
            self._stop_animation()
        
        # Instantly reveal: jump to final position
        board = chess.Board(self.current_puzzle.fen)
        for uci in self.current_puzzle.moves:
            m = chess.Move.from_uci(uci)
            if board.is_legal(m):
                board.push(m)
        self.board.set_board(board)
        self.info_panel.reveal_full_solution()
        self._setmsg("Solution revealed. Click 'Reset Board' to return to the starting position.")

    def _toggle_animation(self):
        if not self.current_puzzle: return
        if self._animating:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if not self.current_puzzle: return
            
        self._anim_board = chess.Board(self.current_puzzle.fen)
        self.board.set_board(self._anim_board)
        self.board.clear_highlights()
        self.board.clear_arrows()
        
        self._animating = True
        self._anim_move_idx = 0
        self._anim_phase = "highlight"
        self.board.set_interaction_locked(True)
        
        self.anim_btn.setText("  \u23f8 Stop Animation  ")
        self.anim_btn.setObjectName("dangerBtn")
        self.anim_btn.setStyleSheet("")
        
        self._set_anim_controls_enabled(False)
        
        self.info_panel.set_puzzle(self.current_puzzle)
        
        self._setmsg("\u25b6 Animating solution...")
        self.msg.setObjectName("hintMessage")
        self.msg.setStyleSheet("")
        
        self._advance_animation()

    def _get_anim_speed_multiplier(self):
        return self.anim_speed_slider.value() / 100.0

    def _advance_animation(self):
        if not self._animating or not self.current_puzzle: return
            
        if self.board.is_animating():
            QTimer.singleShot(50, self._advance_animation)
            return
            
        if self._anim_move_idx >= len(self.current_puzzle.moves):
            self._setmsg("\u2705 Animation complete! Click 'Reset Board' to view the starting position.")
            self.msg.setObjectName("successMessage")
            self.msg.setStyleSheet("")
            QTimer.singleShot(2000, self._stop_animation)
            return
            
        uci = self.current_puzzle.moves[self._anim_move_idx]
        m = chess.Move.from_uci(uci)
        
        base_highlight = 600
        base_move = 500
        base_pause = 500
        
        mult = self._get_anim_speed_multiplier()
        
        if self._anim_phase == "highlight":
            self.board.highlight_hint(m.from_square)
            self.anim_btn.setText(f"  \u23f8 Stop ({self._anim_move_idx + 1}/{len(self.current_puzzle.moves)})  ")
            self._anim_phase = "moving"
            self._anim_timer.start(int(base_highlight / mult))
            
        elif self._anim_phase == "moving":
            if self._anim_board.is_legal(m):
                move_duration = int(base_move / mult)
                self.board.animate_move(m.from_square, m.to_square, duration_ms=move_duration)
                
                def _finalize_move():
                    if not self._animating: return
                    # Calculate SAN before pushing
                    try:
                        san = self._anim_board.san(m)
                    except Exception:
                        san = uci
                    
                    self._anim_board.push(m)
                    self.board.set_board(self._anim_board)
                    self.board.highlight_last_move(m.from_square, m.to_square)
                    self.board.clear_arrows()
                    self.board.add_arrow(m.from_square, m.to_square, QColor(137, 180, 250, 220))
                    
                    if self._anim_board.is_check():
                        ks = self._anim_board.king(self._anim_board.turn)
                        if ks is not None: self.board.highlight_check(ks)
                            
                    self.info_panel.append_history(san, correct=True)
                    self.info_panel.reveal_solution_step(self._anim_move_idx)
                    
                    self._anim_move_idx += 1
                    self._anim_phase = "highlight"
                    self._anim_timer.start(int(base_pause / mult))
                
                QTimer.singleShot(int(base_move / mult) + 50, _finalize_move)
            else:
                self._anim_move_idx += 1
                QTimer.singleShot(10, self._advance_animation)

    def _stop_animation(self):
        was_animating = self._animating
        self._animating = False
        self._anim_timer.stop()
        self._anim_phase = "idle"
        
        self.board._stop_animation()
        self.board.set_interaction_locked(False)
        
        self.anim_btn.setText("  \u25b6 Animate Solution  ")
        self.anim_btn.setObjectName("warningBtn")
        self.anim_btn.setStyleSheet("")
        
        self._set_anim_controls_enabled(True)
        
        if self.current_puzzle:
            self.board.set_position(self.current_puzzle.fen)
            self.board.clear_highlights()
            self.board.clear_arrows()
            self.info_panel.set_puzzle(self.current_puzzle)
            
            if was_animating:
                self._setmsg("\u23f9 Animation stopped. Board reset to starting position.")
                self.msg.setStyleSheet("")

    def _set_anim_controls_enabled(self, enabled: bool):
        self.reset_btn.setEnabled(enabled)
        self.sol_btn.setEnabled(enabled)
        self.new_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled and self.puzzle_index > 0)
        self.next_btn.setEnabled(enabled and self.puzzle_index < len(self.puzzle_queue) - 1)
        self.queue_panel.set_more_enabled(enabled)

    def _update_anim_speed_label(self):
        mult = self._get_anim_speed_multiplier()
        self.anim_speed_label.setText(f"{mult:.2f}x")

    def _flip(self):
        self.board.flipped = not self.board.flipped
        self.board.update()
        
    def _copy_fen(self):
        if self.current_puzzle:
            QApplication.clipboard().setText(self.current_puzzle.fen)
            self.statusBar().showMessage("FEN copied to clipboard!", 3000)

    def _paste_fen(self):
        text = QApplication.clipboard().text()
        try:
            board = chess.Board(text)
            p = Puzzle(fen=text, theme_key="custom", theme_name="Custom Position", 
                       difficulty=0, moves=[], description="Imported from clipboard.", 
                       hint="", category="Custom")
            self.puzzle_queue = [p]
            self.puzzle_index = 0
            self.queue_panel.set_puzzles([p])
            self._load(0)
            self.statusBar().showMessage("Loaded FEN from clipboard.", 3000)
        except Exception:
            self.statusBar().showMessage("Invalid FEN in clipboard.", 3000)

    def _export_current(self):
        if not self.current_puzzle:
            QMessageBox.warning(self, "No Puzzle", "Please load a puzzle first.")
            return
        if not VIDEO_EXPORT_ENABLED:
            QMessageBox.critical(self, "Missing Dependency", "Video export requires 'opencv-python' and 'numpy'.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Video", f"puzzle_{self.current_puzzle.theme_key}.mp4", "MP4 Files (*.mp4)")
        if not filepath: return
        
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        bgm_path = os.path.join(assets_dir, "background.mp3")
        if not os.path.exists(bgm_path): bgm_path = None
        
        self.statusBar().showMessage("Rendering video, please wait...")
        self.exporter = SingleVideoExporter(self.current_puzzle, filepath, assets_dir, enable_sfx=True, bgm_path=bgm_path)
        self.exporter.finished_export.connect(self._on_export_done)
        self.exporter.error.connect(self._on_export_error)
        self.exporter.start()

    def _on_export_done(self, filepath):
        self.statusBar().showMessage(f"Video exported to {filepath}")
        QMessageBox.information(self, "Success", f"Video exported successfully to:\n{filepath}")

    def _on_export_error(self, err):
        self.statusBar().showMessage("Video export failed.")
        QMessageBox.critical(self, "Error", f"Failed to render video:\n{err}")

    def _export_pgn(self):
        if not self.current_puzzle:
            QMessageBox.warning(self, "No Puzzle", "Please load a puzzle first.")
            return
            
        board = chess.Board(self.current_puzzle.fen)
        game = chess.pgn.Game()
        game.headers["Event"] = "Procedurally Generated Chess Puzzle"
        game.headers["White"] = "Player" if board.turn == chess.WHITE else "Engine"
        game.headers["Black"] = "Player" if board.turn == chess.BLACK else "Engine"
        game.headers["Theme"] = self.current_puzzle.theme_name
        game.headers["Difficulty"] = str(self.current_puzzle.difficulty)
        
        node = game.add_main_variation(chess.Move.from_uci(self.current_puzzle.moves[0]))
        for uci in self.current_puzzle.moves[1:]:
            node = node.add_main_variation(chess.Move.from_uci(uci))
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save PGN", f"puzzle_{self.current_puzzle.theme_key}.pgn", "PGN Files (*.pgn)")
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(str(game))
                self.statusBar().showMessage(f"PGN exported to {filepath}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PGN:\n{e}")

    def _batch_gen(self):
        if not VIDEO_EXPORT_ENABLED:
            QMessageBox.critical(self, "Missing Dependency", "Video export requires 'opencv-python' and 'numpy'.\nPlease run: pip install opencv-python numpy")
            return
        themes = self.filter_panel.get_selected_themes()
        if not themes:
            QMessageBox.warning(self, "No Themes", "Please select at least one theme in the filter panel.")
            return
        dialog = BatchGenerationDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            self.batch_proc = BatchProcessor(self.generator, settings, themes, (self._dmin, self._dmax))
            self.batch_dlg = QProgressDialog("Starting batch...", "Cancel", 0, 100, self)
            self.batch_dlg.setWindowTitle("Batch Generation")
            self.batch_dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            self.batch_dlg.setAutoClose(True)
            self.batch_dlg.setMinimumDuration(0)
            self.batch_proc.progress.connect(self._batch_progress)
            self.batch_proc.finished_batch.connect(self._batch_finished)
            self.batch_dlg.canceled.connect(self.batch_proc.terminate)
            self.batch_proc.start()
            self.batch_dlg.show()

    def _batch_progress(self, current, total, msg):
        self.batch_dlg.setMaximum(total)
        self.batch_dlg.setValue(current)
        self.batch_dlg.setLabelText(msg)

    def _batch_finished(self, out_dir):
        QMessageBox.information(self, "Batch Complete", f"Successfully generated videos to:\n{out_dir}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Left: self._prev()
        elif e.key() == Qt.Key.Key_Right: self._next()
        elif e.key() == Qt.Key.Key_R: self._reset()
        elif e.key() == Qt.Key.Key_F: self._flip()
        elif e.key() == Qt.Key.Key_P: self._toggle_animation()
        else: super().keyPressEvent(e)


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
    w=ChessPuzzleApp(); w.show()
    sys.exit(app.exec())

if __name__=="__main__": main()