import math
from collections import namedtuple
from itertools import product

Cell = namedtuple("Cell", ("row", "col", "val"))


class ConflictingCellError(Exception):
    pass


class Constraint:
    def __init__(self, limit):
        self.limiting_set = frozenset(limit)

    def check_if_met(self, known_cells):
        if not self.limiting_set.isdisjoint(known_cells):
            intersection = self.limiting_set & known_cells
            if len(intersection) != 1:
                raise ConflictingCellError("Conflicting cell definitions in known_cells")
            return True, self.limiting_set - intersection
        return False, None

    def remove_impossible(self, impossible):
        updated_set = set()
        for cell in self.limiting_set:
            if cell not in impossible:
                updated_set.add(cell)
        return Constraint(updated_set)

    @classmethod
    def cell_constraint(cls, row, column, dim):
        return cls({Cell(row, column, v)
                    for v in range(1, dim + 1)})

    @classmethod
    def row_constraint(cls, row, value, dim):
        return cls({Cell(row, c, value)
                    for c in range(dim)})

    @classmethod
    def column_constraint(cls, column, value, dim):
        return cls({Cell(r, column, value)
                    for r in range(dim)})

    @classmethod
    # row and col represent top left corner of square
    def square_constraint(cls, square_row, square_col, value, sqr_dim):
        return cls({Cell(r, c, value)
                    for r, c in product(range(square_row, square_row + sqr_dim),
                                        range(square_col, square_col + sqr_dim))})


class Sudoku:
    def __init__(self, board):
        valid, msg = self.is_valid(board)
        if not valid:
            raise ValueError("Invalid sudoku board: {0}".format(msg))
        self.board = board

    @staticmethod
    def get_rows(board):
        for row in board:
            yield row

    @staticmethod
    def get_cols(board):
        dim = len(board)
        for c in range(dim):
            yield [row[c] for row in board]

    @staticmethod
    def get_squares(board):
        dim = len(board)
        sqrt_dim = int(math.sqrt(dim))
        for big_row in range(0, dim, sqrt_dim):
            for big_col in range(0, dim, sqrt_dim):
                square = []
                for row in range(sqrt_dim):
                    for col in range(sqrt_dim):
                        square.append(board[big_row + row][big_col + col])
                yield square

    @classmethod
    def is_valid(cls, board):
        dim = len(board)
        if int(math.sqrt(dim)) ** 2 != dim:
            return False, "Dimension is not a square number"
        if any(len(row) != dim for row in board):
            return False, "Board is not square"
        if any(any(num < 0 or num > dim for num in row) for row in board):
            return False, "Value out of range"

        for row in cls.get_rows(board):
            filtered_row = list(filter(lambda a: a != 0, row))
            if len(filtered_row) != len(set(filtered_row)):
                return False, "Row contains duplicates"
        for col in cls.get_cols(board):
            filtered_col = list(filter(lambda a: a != 0, col))
            if len(filtered_col) != len(set(filtered_col)):
                return False, "Column contains duplicates"
        for sqr in cls.get_squares(board):
            filtered_sqr = list(filter(lambda a: a != 0, sqr))
            if len(filtered_sqr) != len(set(filtered_sqr)):
                return False, "Square contains duplicates"

        return True, None

    def solve(self):
        constraints = self._filter_constraints(self._make_constraints())
        return self._next_level(constraints)[1]

    @property
    def _known_cells(self):
        dim = len(self.board)
        cells = set()
        for row in range(dim):
            for col in range(dim):
                if self.board[row][col] != 0:
                    cells.add(Cell(row, col, self.board[row][col]))
        return cells

    def _make_constraints(self):
        dim = len(self.board)
        sqr_dim = int(math.sqrt(dim))
        constraints = set()
        for r, c in product(range(dim), range(dim)):
            constraints.add(Constraint.cell_constraint(r, c, dim))
        for r, v in product(range(dim), range(1, dim + 1)):
            constraints.add(Constraint.row_constraint(r, v, dim))
        for c, v in product(range(dim), range(1, dim + 1)):
            constraints.add(Constraint.column_constraint(c, v, dim))
        for s_r, s_c, v in product(range(0, dim, sqr_dim),
                                   range(0, dim, sqr_dim),
                                   range(1, dim + 1)):
            constraints.add(Constraint.square_constraint(s_r, s_c, v, int(math.sqrt(dim))))
        return constraints

    def _filter_constraints(self, constraints, known=None):
        if known is None:
            known = self._known_cells
        filtered = set()
        impossible_cells = set()
        for con in constraints:
            already_met, became_impossible = con.check_if_met(known)
            if already_met:
                impossible_cells.update(became_impossible)
            else:
                filtered.add(con)
        return set(map(lambda c: c.remove_impossible(impossible_cells), filtered))

    def _next_level(self, constraints, known=None):
        if known is None:
            known = self._known_cells
        if len(constraints) == 0:  # terminating successfully
            return True, known
        tried_cons = set()
        while True:
            remaining = constraints - tried_cons
            if len(remaining) == 0:
                return False, None
            selected = min(remaining, key=lambda c: len(c.limiting_set))
            # print(len(known), len(tried_cons), len(constraints), len(selected.limiting_set))
            if len(selected.limiting_set) == 0:  # terminating unsuccessfully
                return False, None
            for assumed in selected.limiting_set:
                speculated_cells = known | {assumed}
                implied_constraints = self._filter_constraints(constraints, known=speculated_cells)
                done, solution_cells = self._next_level(implied_constraints, known=speculated_cells)
                if done:
                    return done, solution_cells
            tried_cons.add(selected)

    @staticmethod
    def format(cells):
        dict_of_cells = {(cell.row, cell.col): cell.val for cell in cells}
        dim = max(cells, key=lambda c: c.row).row + 1
        width = len(str(dim))
        board = []
        for row in range(dim):
            entries = []
            for col in range(dim):
                if (row, col) in dict_of_cells:
                    val = str(dict_of_cells[(row, col)])
                else:
                    val = "?"
                entries.append(val.rjust(width))
            board.append(" ".join(entries))
        return "\n".join(board)


def main():
    while True:
        try:
            dim = int(input("Enter the dimension of the sudoku board: "))
            print("Enter sudoku, with 0s for unknown values and spaces to separate values:")
            b = []
            for i in range(dim):
                row = input()
                b.append([int(e) for e in row.split()])

            s = Sudoku(b)
            break
        except ValueError as e:
            print(e)
    print("yeet")
    print(s.solve())


def test():
    """
    b = [[8, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 3, 6, 0, 0, 0, 0, 0],
         [0, 7, 0, 0, 9, 0, 2, 0, 0],
         [0, 5, 0, 0, 0, 7, 0, 0, 0],
         [0, 0, 0, 0, 4, 5, 7, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 3, 0],
         [0, 0, 1, 0, 0, 0, 0, 6, 8],
         [0, 0, 8, 5, 0, 0, 0, 1, 0],
         [0, 9, 0, 0, 0, 0, 4, 0, 0]]

    """
    b = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
         [6, 0, 0, 1, 9, 5, 0, 0, 0],
         [0, 9, 8, 0, 0, 0, 0, 6, 0],
         [8, 0, 0, 0, 6, 0, 0, 0, 3],
         [4, 0, 0, 8, 0, 3, 0, 0, 1],
         [7, 0, 0, 0, 2, 0, 0, 0, 6],
         [0, 6, 0, 0, 0, 0, 2, 8, 0],
         [0, 0, 0, 4, 1, 9, 0, 0, 5],
         [0, 0, 0, 0, 8, 0, 0, 7, 9]]
    s = Sudoku(b)
    print(Sudoku.format(s.solve()))


if __name__ == "__main__":
    test()
