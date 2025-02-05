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
    return board[0, col] == 0


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
    for row in range(ROW_COUNT-1, -1, -1):  # Start from the last row, move up

        if board[row, col] == 0:  # Check if the spot is empty
            
            return row  # Found the first open row
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
            
    for row in range(ROW_COUNT - 3):
            
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
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if board[0, col] == 0:  # If the top row of the column is empty
            valid_locations.append(col)
    return valid_locations


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
    # Check if either player has won
    if winning_move(board, 1) or winning_move(board, 2):
        return True
    # Check if there are no valid locations left
    if len(get_valid_locations(board)) == 0:
        return True
    return False



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
    # TODO: implement
    valid_locations = get_valid_locations(board)
    
    # Base case: If we've reached the depth limit or the game is over
    if depth == 0 or is_terminal_node(board):
        return None, score_position(board, 1)  # Use the score for Player 1's evaluation
    
    if maximizingPlayer:
        # Maximizing Player (Player 1) wants to maximize the score
        best_score = -float('inf')  # Start with a very low score
        best_col = valid_locations[0]  # Initialize best column

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)  # Player 1 drops the piece

            # Recursively call minimax for the next move
            _, score = minimax(temp_board, depth - 1, False)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col, best_score

    else:
        # Minimizing Player (Player 2) wants to minimize the score
        best_score = float('inf')  # Start with a very high score
        best_col = valid_locations[0]  # Initialize best column

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)  # Player 2 drops the piece

            # Recursively call minimax for the next move
            _, score = minimax(temp_board, depth - 1, True)
            if score < best_score:
                best_score = score
                best_col = col

        return best_col, best_score



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
    
