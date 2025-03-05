# some assistance from chat gpt

import math
import random
import numpy as np

# Board dimensions for Connect Four
ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    """
    Creates an empty Connect Four board (numpy 2D array).

    Returns:
    np.ndarray:
        A 2D numpy array of shape (ROW_COUNT, COLUMN_COUNT) filled with zeros (float).
    """
    # TODO: implement
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype = float)


def drop_piece(board, row, col, piece):
    """
    Places a piece (1 or 2) at the specified (row, col) position on the board.

    Parameters:
    board (np.ndarray): The current board, shape (ROW_COUNT, COLUMN_COUNT).
    row (int): The row index where the piece should be placed.
    col (int): The column index where the piece should be placed.
    piece (int): The piece identifier (1 or 2).
    
    Returns:
    None. The 'board' is modified in-place. Do NOT return a new board!
    """
    # TODO: implement
    if row is not None: 

        board[row, col] = piece


def is_valid_location(board, col):
    """
    Checks if dropping a piece in 'col' is valid (column not full).

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to check.

    Returns:
    bool: True if it's valid to drop a piece in this column, False otherwise.
    """
    # TODO: implement
    #return board[0, col] == 0
    return np.any(board[:, col] == 0)


def get_next_open_row(board, col):
    """
    Gets the next open row in the given column.

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to search.

    Returns:
    int: The row index of the lowest empty cell in this column.
    """
    # TODO: implement
    for row in range(ROW_COUNT):

        if board[row, col] == 0:

            return row
        
    return None 


def winning_move(board, piece):
    """
    Checks if the board state is a winning state for the given piece.

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    bool: True if 'piece' has a winning 4 in a row, False otherwise.
    This requires checking horizontally, vertically, and diagonally.
    """
    # TODO: implement
    ROW_COUNT, COLUMN_COUNT = board.shape

    for row in range(ROW_COUNT):

        for col in range(COLUMN_COUNT - 3):
            
            if np.all(board[row, col:col + 4] == piece):

                return True  # horizontal win
            
    for row in range(ROW_COUNT - 3):
            
        for col in range(COLUMN_COUNT - 3):

            if np.all(board[row: row + 4, col] == piece):

                return True  # vertical win

    for row in range(ROW_COUNT - 3):
            
        for col in range(COLUMN_COUNT - 3):

            if np.all([board[row + i, col + i] == piece for i in range(4)]):

                return True  # \ diagonal win
            
    for row in range(3, ROW_COUNT):
            
        for col in range(COLUMN_COUNT - 3):

            if np.all([board[row - i, col + i] == piece for i in range(4)]):

                return True  # / diagonal win
    


def get_valid_locations(board):
    """
    Returns a list of columns that are valid to drop a piece.

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    list of int: The list of column indices that are not full.
    """
    # TODO: implement
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]


def is_terminal_node(board):
    """
    Checks if the board is in a terminal state:
      - Either player has won
      - Or no valid moves remain

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    # TODO: implement
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0




def score_position(board, piece):
    """
    Evaluates the board for the given piece.
    (Already implemented to highlight center-column preference.)

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    int: Score favoring the center column. 
         (This can be extended with more advanced heuristics.)
    """
    # This is already done for you; no need to modify
    # The heuristic here scores the center column higher, which means
    # it prefers to play in the center column.
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Performs minimax with alpha-beta pruning to choose the best column.

    Parameters:
    board (np.ndarray): The current board.
    depth (int): Depth to which the minimax tree should be explored.
    alpha (float): Alpha for alpha-beta pruning.
    beta (float): Beta for alpha-beta pruning.
    maximizingPlayer (bool): Whether it's the maximizing player's turn.

    Returns:
    tuple:
        - (column (int or None), score (float)):
          column: The chosen column index (None if no moves).
          score: The heuristic score of the board state.
    """
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 1):
                return (None, math.inf) 
            elif winning_move(board, 2):
                return (None, -math.inf) 
            else:
                return (None, 0) 
        return (None, score_position(board, 1))

    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)
            new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:  # minimising player
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)
            new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value



if __name__ == "__main__":
    # Simple debug scenario
    # Example usage: create a board, drop some pieces, then call minimax
    example_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    print("Debug: Created an empty Connect Four board.\n")
    print(example_board)

    # TODO: Students can test their own logic here once implemented, e.g.:
    drop_piece(example_board, 5, 3, 1)  # Player 1 drops a piece in column 3 (row 5)
    drop_piece(example_board, 5, 4, 2)  # Player 2 drops a piece in column 4 (row 5)
    drop_piece(example_board, 4, 3, 1)  # Player 1 drops a piece in column 3 (row 4)
    drop_piece(example_board, 4, 2, 2)  # Player 2 drops a piece in column 2 (row 4)
    drop_piece(example_board, 3, 3, 1) 

    col, score = minimax(example_board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    print("Chosen column:", col, "Score:", score)


if __name__ == "__main__":
    board = create_board()
    print("Initial Board:")
    print_board(board)

    test_moves = [(0, 3, 1), (1, 3, 1), (2, 3, 1), (3, 3, 1)] 

    for move in test_moves:
        row = get_next_open_row(board, move[1])
        if row is not None:
            drop_piece(board, row, move[1], move[2])

    print("\nBoard after test moves:")
    print_board(board)

    if winning_move(board, 1):
        print("\nPlayer 1 has won!")

    # If minimax is implemented, test it
    # col, score = minimax(board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    # print("\nMinimax chosen column:", col, "Score:", score)

