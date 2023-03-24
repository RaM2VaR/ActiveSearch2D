"""OR-Tools solution to the N-queens problem."""
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np

def main(board_size):
    # Creates the solver.
    solver = pywrapcp.Solver('n-queens')

    # Creates the variables.
    # The array index is the column, and the value is the row.
    queens = [
        solver.IntVar(0, board_size - 1, f'x{i}') for i in range(board_size)
    ]

    # Creates the constraints.
    # All rows must be different.
    solver.Add(solver.AllDifferent(queens))

    # No two queens can be on the same diagonal.
    solver.Add(solver.AllDifferent([queens[i] + i for i in range(board_size)]))
    solver.Add(solver.AllDifferent([queens[i] - i for i in range(board_size)]))

    db = solver.Phase(queens, solver.CHOOSE_FIRST_UNBOUND,
                      solver.ASSIGN_MIN_VALUE)

    # Iterates through the solutions, displaying each.
    num_solutions = 0
    solver.NewSearch(db)
    listTSV_sols = []
    while solver.NextSolution():
        # Displays the solution just computed.
        for i in range(board_size):
            col = []
            for j in range(board_size):
                if queens[j].Value() == i:
                    # There is a queen in column j, row i.
                    col.append(1)
                    # print('Q', end=' ')
                else:
                    # print('_', end=' ')
                    col.append(0)
            listTSV_sols.append(col)
            # print()
            # print(TSV_sols)
        print()

        
        num_solutions += 1
    solver.EndSearch()
    sizes = [2,10,4,40,92]
    TSV_sols = np.resize(listTSV_sols,(sizes[board_size-4],board_size,board_size))
    # Statistics.
    print('\nStatistics')
    print(f'  failures: {solver.Failures()}')
    print(f'  branches: {solver.Branches()}')
    print(f'  wall time: {solver.WallTime()} ms')
    print(f'  Solutions found: {num_solutions}')
    return TSV_sols

if __name__ == '__main__':
    # By default, solve the 8x8 problem.
    size = 4
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    TSV_sols = main(size)