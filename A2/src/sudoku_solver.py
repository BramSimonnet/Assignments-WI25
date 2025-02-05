def print_board(board):
    """
    Prints the Sudoku board in a grid format.
    0 indicates an empty cell.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    None
    """
    for row_idx, row in enumerate(board):
        # Print a horizontal separator every 3 rows (for sub-grids)
        if row_idx % 3 == 0 and row_idx != 0:
            print("- - - - - - - - - - -")

        row_str = ""
        for col_idx, value in enumerate(row):
            # Print a vertical separator every 3 columns (for sub-grids)
            if col_idx % 3 == 0 and col_idx != 0:
                row_str += "| "

            if value == 0:
                row_str += ". "
            else:
                row_str += str(value) + " "
        print(row_str.strip())


def find_empty_cell(board):
    """
    Finds an empty cell (indicated by 0) in the Sudoku board.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    tuple or None:
        - If there is an empty cell, returns (row_index, col_index).
        - If there are no empty cells, returns None.
    """
    # TODO: implement
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return (row, col)  # Return the position of the empty cell
    return None  # No empty cells



def is_valid(board, row, col, num):
    """
    Checks if placing 'num' at board[row][col] is valid under Sudoku rules:
      1) 'num' is not already in the same row
      2) 'num' is not already in the same column
      3) 'num' is not already in the 3x3 sub-box containing that cell

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.
    row (int): Row index of the cell.
    col (int): Column index of the cell.
    num (int): The candidate number to place.

    Returns:
    bool: True if valid, False otherwise.
    """
    # TODO: implement
    if num in board[row]:
        return False
    
    # Check the column
    for r in range(9):
        if board[r][col] == num:
            return False

    # Check the 3x3 sub-grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    
    return True


def solve_sudoku(board):
    """
    Solves the Sudoku puzzle in 'board' using backtracking.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 indicates an empty cell.

    Returns:
    bool:
        - True if the puzzle is solved successfully.
        - False if the puzzle is unsolvable.
    """
    # TODO: implement
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        return True  # Solved

    row, col = empty_cell

    # Try all numbers from 1 to 9
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num  # Try placing num
            if solve_sudoku(board):  # Recursively try to solve the board
                return True
            board[row][col] = 0  # Backtrack if no solution is found

    return False  # No solution found


def is_solved_correctly(board):
    """
    Checks that the board is fully and correctly solved:
    - Each row contains digits 1-9 exactly once
    - Each column contains digits 1-9 exactly once
    - Each 3x3 sub-box contains digits 1-9 exactly once

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.

    Returns:
    bool: True if the board is correctly solved, False otherwise.
    """
    # TODO: implement
    for row in board:
        if sorted(row) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False
    
    # Check columns
    for col in range(9):
        column = [board[row][col] for row in range(9)]
        if sorted(column) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False
    
    # Check 3x3 subgrids
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = [board[r][c] for r in range(row, row + 3) for c in range(col, col + 3)]
            if sorted(subgrid) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                return False
    
    return True


if __name__ == "__main__":
    # Example usage / debugging:
    example_board = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7],
    ]

    print("Debug: Original board:\n")
    print_board(example_board)
  
    if solve_sudoku(example_board):
        print("\nSudoku Solved!\n")
        print_board(example_board)
    else:
        print("\nNo solution exists.")
