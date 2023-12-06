# cython: language_level=3
# Import necessary modules
import numpy as np
cimport numpy as np
from cython.parallel import prange



# Define your cdef function
cpdef np.ndarray[np.int_t, ndim=2] getValidMoves(np.ndarray[np.int_t, ndim=2] board, int color):
    cdef set moves = set()
    cdef int y, x, ydir, xdir, size, flips_count
    cdef int[:, :] direction = np.array([(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)])
    cdef int i
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y, x] == color:
                for i in range(8):
                    flips_count = 0
                    for size in range(1, board.shape[0]):
                        ydir = y + direction[i][1] * size
                        xdir = x + direction[i][0] * size
                        if 0 <= xdir < board.shape[0] and 0 <= ydir < board.shape[0]:
                            if board[ydir, xdir] == -color:
                                flips_count += 1
                            elif board[ydir, xdir] == 0:
                                if flips_count != 0:
                                    moves.add((ydir, xdir))
                                break
                            else:
                                break
                        else:
                            break
    return np.array(list(moves))


cpdef np_2_str(position):
    return str(position[0]) + ',' + str(position[1])

cpdef str_2_np(position):
    return np.array([int(position.split(',')[0]), int(position.split(',')[1])])

cpdef executeMove(np.ndarray[np.int_t, ndim=2] board, int color, np.ndarray[np.int_t, ndim=1] position):
    cdef int y, x, ydir, xdir, size
    cdef list flips
    y, x = position[0], position[1]
    board[y, x] = color

    for direction in [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]:
        flips = []
        for size in range(1, len(board)):
            ydir = y + direction[1] * size
            xdir = x + direction[0] * size
            if 0 <= xdir < len(board) and 0 <= ydir < len(board):
                if board[ydir, xdir] == -color:
                    flips.append((ydir, xdir))
                elif board[ydir, xdir] == color:
                    if flips:
                        board[tuple(zip(*flips))] = color
                    break
                else:
                    break
            else:
                break

def find_opp_move(np.ndarray[np.int_t, ndim=2] board, np.ndarray[np.int_t, ndim=2] current_board):
    cdef list opp_moves = []
    cdef int y, x

    for y, x in zip(*np.where(board != current_board)):
        if current_board[y, x] == 0 and board[y, x] != 0:
            return []
        if current_board[y, x] != 0 and board[y, x] == 0:
            opp_moves.append(np_2_str((y, x)))
    return opp_moves

def judge_next_game(np.ndarray[np.int_t, ndim=2] board, np.ndarray[np.int_t, ndim=2] current_board):
    cdef int y, x

    for y, x in zip(*np.where(board != current_board)):
        if current_board[y, x] == 0 and board[y, x] != 0:
            return True
    return False

def isEndGame(np.ndarray[np.int_t, ndim=2] board):
    cdef int valid_moves_white = getValidMoves(board, -1).shape[0]
    cdef int valid_moves_black = getValidMoves(board, 1).shape[0]
    cdef int white_count, black_count

    if valid_moves_white == 0 and valid_moves_black == 0:
        white_count = np.sum(board == -1)
        black_count = np.sum(board == 1)

        if white_count > black_count:
            return -1
        elif black_count > white_count:
            return 1
        else:
            return 0
    else:
        return None