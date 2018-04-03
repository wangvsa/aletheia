import numpy as np
import random
import sys
import math
import argparse
from skimage.util.shape import view_as_windows, view_as_blocks
from bits import bit_flip


T = 200                     # total number of iterations
ROWS = 480                   # rows of the plate
COLUMNS= 480                 # columns of the plate

WINDOW_ROWS = 60
WINDOW_COLS = 60
WINDOW_OVERLAP = 20

SAVE_WINDOWS = True
SAVE_INTERVAL = 1

HEATING_DEVICE_SIZE = 20     # the length of the heating source



# heating device shaped like X
# backgoround is 0, heating device is 1
Gr = np.eye(HEATING_DEVICE_SIZE)
for iGr in range(HEATING_DEVICE_SIZE):
    Gr[iGr,-iGr-1] = 10
    Gr[iGr, iGr] = 10

# Insert heating device in the center
# Function to set M values corresponding to non-zero Gr values
def assert_heaters(frame, Gr):
    start_row, end_row = (ROWS-HEATING_DEVICE_SIZE)/2, (ROWS+HEATING_DEVICE_SIZE)/2
    start_column , end_column = (COLUMNS-HEATING_DEVICE_SIZE)/2, (COLUMNS+HEATING_DEVICE_SIZE)/2
    frame[start_row:end_row, start_column:end_column] = np.where(Gr > 0, Gr, frame[start_row:end_row, start_column:end_column])
    return frame


def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step=step)
    return np.vstack(windows)

def split_to_blocks(frame, rows, cols):
    blocks = view_as_blocks(frame, (rows, cols))
    return np.vstack(blcosk)


# Split a frame to windows/blocks and save to a npy file
# split: Wether split to windows/blocks
def dump_data_file(frame, filename, split=True):
    if split:
        windows = split_to_windows(frame, WINDOW_ROWS, WINDOW_COLS, WINDOW_OVERLAP)
        np.save(filename, windows)
    else :
        np.save(filename, frame)


def get_flip_error(val):
    while True:
        pos = random.randint(0, 20)
        error = bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    #error = max(10e+5, error)
    #error = min(-10e+5, error)
    return error


# Insert errors in a given frame
# multiple: whether to insert multiple errors
def insert_error(frame, multiple=True):
    if not multiple:
        x = random.randint(0, ROWS-1)
        y = random.randint(0, COLUMNS-1)
        frame[x, y] = get_flip_error(frame[x,y])
    else:
        for start_x in range(20, 460, 40):
            for start_y in range(20, 460, 40):
                x = start_x + random.randint(0, 20)
                y = start_y + random.randint(0, 20)
                old = frame[x, y]
                frame[x, y] = get_flip_error(frame[x,y])
                #print("insert error at %s, old:%s new:%s" %((x,y), old, frame[x,y]))
    return frame

# Heat distribution program, a clean run
# interval: the interval to save data
def heat_distribution_clean(interval = 1):
    frame = np.zeros((ROWS, COLUMNS))+2
    frame = assert_heaters(frame, Gr)
    for i in range(0, T):

        print("iter: %s" %(i))

        # Stencil computation
        last_frame = np.copy(frame)
        for x in range(1, ROWS-1):
            for y in range(1, COLUMNS-1):
                frame[x,y] = (last_frame[x-1,y] + last_frame[x+1,y] +
                                last_frame[x,y-1] + last_frame[x,y+1])/4.0
        # Re-assert heaters
        frame = assert_heaters(frame, Gr)

        # Dump to file
        if i % interval == 0:
            filename = str(i) + ".npy"
            print("save to file: %s" %(filename))
            dump_data_file(frame, filename, SAVE_WINDOWS)


# Rean clean data and insert error at given iteration then continue to run
# interval: the interval to save data
# error_iter: insert error at this iteration
# multiple_error: whether to insert multiple errors
def heat_distribution_error(error_iter, interval=1, multiple_error=True):
    frames = np.load("./data/clean_no_split.npy")

    # Insert error
    frame = np.copy(frames[error_iter])
    frame = insert_error(frame, multiple_error)
    print("insert error at iter: %s" %(error_iter))

    for i in range(error_iter, T):

        # Dump to file
        if i % interval == 0:
            filename = str(error_iter)+"_"+str(i-error_iter)+".npy"
            print("save to file: %s" %(filename))
            dump_data_file(frame, filename, SAVE_WINDOWS)

        # Stencil computation
        last_frame = np.copy(frame)
        for x in range(1, ROWS-1):
            for y in range(1, COLUMNS-1):
                frame[x,y] = (last_frame[x-1,y] + last_frame[x+1,y] +
                                last_frame[x,y-1] + last_frame[x,y+1])/4.0

        # Re-assert heaters
        frame = assert_heaters(frame, Gr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--iteration", type=int, help="total iterations to run")
    parser.add_argument("-i", "--interval", type=int, help="intervals to dump data")
    parser.add_argument("-e", "--error_iter", type=int, help="Insert error at given iteration")
    args = parser.parse_args()
    print args

    if args.iteration: T = args.iteration
    if args.interval: SAVE_INTERVAL = args.interval
    if args.error_iter is not None:
        print("error run")
        heat_distribution_error(error_iter=args.error_iter, interval=SAVE_INTERVAL)
    else:
        print("clean run")
        heat_distribution_clean(interval=SAVE_INTERVAL)

