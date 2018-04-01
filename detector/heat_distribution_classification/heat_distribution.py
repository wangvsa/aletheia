import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

T = 200                     # total number of iterations
ROWS = 480                   # rows of the plate
COLUMNS= 480                 # columns of the plate
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
def heat_distribution():
    MM = np.zeros((T, ROWS, COLUMNS))
    assert_heaters(MM[0], Gr)
    for i in range(1, T):
        MM[i] = MM[i-1].copy()
        for x in range(1, ROWS-1):
            for y in range(1, COLUMNS-1):
                MM[i,x,y] = (MM[i,x-1,y] + MM[i,x+1,y] + MM[i,x,y-1] + MM[i,x,y+1])/4
        # Re-assert heaters
        assert_heaters(MM[i], Gr)
    return MM

def insert_error(MM):
    has_error = np.zeros(len(MM))    # if has error
    sign = 1
    THREASHOLD = 0.5
    for i in range(len(MM)):
        if random.randint(0, 1):
            x = np.random.randint(0, ROWS)
            y = np.random.randint(0, COLUMNS)
            has_error[i] = 1
            error = np.random.uniform(MM[i][x, y] * THREASHOLD, 1.0, size=1)  # from origin_val*THREASHOLD ~ 1.0
            MM[i][x, y] = MM[i][x,y] + sign * error
            MM[i][x, y] = min(MM[i][x,y], 1)
            #MM[i][x, y] = max(MM[i][x,y], 0)
            #sign = sign * -1
            #MM[i][x, y] = 1
    return MM, has_error
