import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.util.shape import view_as_windows, view_as_blocks


T = 200                     # total number of iterations
ROWS = 480                   # rows of the plate
COLUMNS= 480                 # columns of the plate

WINDOW_ROWS = 60
WINDOW_COLS = 60
WINDOW_OVERLAP = 20

HEATING_DEVICE_SIZE = 3     # the length of the heating source

# heating device shaped like X
# backgoround is 0, heating device is 1
Gr = np.eye(HEATING_DEVICE_SIZE)
for iGr in range(HEATING_DEVICE_SIZE):
    Gr[iGr,-iGr-1] = 1

# Insert heating device in the center
# Function to set M values corresponding to non-zero Gr values
def assert_heaters(frame, Gr):
    start_row, end_row = (ROWS-HEATING_DEVICE_SIZE)/2, (ROWS+HEATING_DEVICE_SIZE)/2
    start_column , end_column = (COLUMNS-HEATING_DEVICE_SIZE)/2, (COLUMNS+HEATING_DEVICE_SIZE)/2
    frame[start_row:end_row, start_column:end_column] = np.where(Gr > 0, Gr, frame[start_row:end_row, start_column:end_column])


def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step=step)
    return np.vstack(windows)

def split_to_blocks(frame, rows, cols):
    blocks = view_as_blocks(frame, (rows, cols))
    return np.vstack(blcosk)


# Split a frame to windows/blocks and save to a npy file
def dump_data_file(frame, filename):
    windows = split_to_windows(frame, WINDOW_ROWS, WINDOW_COLS, WINDOW_OVERLAP)
    np.save(filename, windows)


# Build MM, a list of matrices, each element corresponding to M at a given step
def heat_distribution(interval = 1):
    frame = np.zeros((ROWS, COLUMNS))
    assert_heaters(frame, Gr)
    last_frame = np.copy(frame)
    for i in range(0, T):

        print("iter: %s" %(i))

        # Stencil computation
        for x in range(1, ROWS-1):
            for y in range(1, COLUMNS-1):
                frame[x,y] = (last_frame[x-1,y] + last_frame[x+1,y] +
                                last_frame[x,y-1] + last_frame[x,y+1])/4.0
        # Dump to file
        if i % interval == 0:
            filename = str(i)+".npy"
            print("save to file: %s" %(filename))
            dump_data_file(frame, filename)

        # Re-assert heaters
        assert_heaters(frame, Gr)

if __name__ == "__main__":
    heat_distribution(interval = 20)


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
