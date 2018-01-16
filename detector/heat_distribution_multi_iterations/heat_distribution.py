import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

T = 100                     # total number of iterations
ROWS = 20                   # rows of the plate
COLUMNS= 20                 # columns of the plate
HEATING_DEVICE_SIZE = 3     # the length of the heating source

# heating device shaped like X
# backgoround is 0, heating device is 1
Gr = np.eye(HEATING_DEVICE_SIZE)
for iGr in range(HEATING_DEVICE_SIZE):
    Gr[iGr,-iGr-1] = 1

# Function to set M values corresponding to non-zero Gr values
def assert_heaters(M, Gr):
    start_row, end_row = (ROWS-HEATING_DEVICE_SIZE)/2, (ROWS+HEATING_DEVICE_SIZE)/2
    start_column , end_column = (COLUMNS-HEATING_DEVICE_SIZE)/2, (COLUMNS+HEATING_DEVICE_SIZE)/2
    M[start_row:end_row, start_column:end_column] = np.where(Gr > 0, Gr, M[start_row:end_row, start_column:end_column])


# Build MM, a list of matrices, each element corresponding to M at a given step
def heat_distribution(all_with_error = True):
    MM = np.zeros((ROWS, COLUMNS))
    error_position = [-1, -1]

    assert_heaters(MM, Gr)
    for i in range(1, T):
        for x in range(1, ROWS-1):
            for y in range(1, COLUMNS-1):
                MM[x,y] = (MM[x-1,y] + MM[x+1,y] + MM[x,y-1] + MM[x,y+1])/4
        # Re-assert heaters
        assert_heaters(MM, Gr)

        insert_error = all_with_error or random.randint(0, 1)

        if i == 10 and insert_error :    # insert error and return it after 10 iterations
            x = np.random.randint(0, ROWS)
            y = np.random.randint(0, COLUMNS)
            error_position = [x, y]
            MM[x, y] = 1

        if i == 15 :
            return MM, error_position

# All frames contain error
def get_all_error_data():
    N = 30000
    MM = np.zeros((N, ROWS, COLUMNS))
    error_positions = np.zeros((N, 2))
    for i in range(N):
        MM[i], error_positions[i] = heat_distribution()
    return MM, error_positions

# only random number of frames contain error
def get_random_error_data():
    N = 10000
    MM = np.zeros((N, ROWS, COLUMNS))
    error_positions = np.zeros((N, 2))
    for i in range(N):
        MM[i], error_positions[i] = heat_distribution(False)
    return MM, error_positions
