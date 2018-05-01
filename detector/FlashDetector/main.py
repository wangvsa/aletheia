import numpy as np
import glob, os, math, random, argparse
import torch, torchvision
import torch.nn as nn
from bits import bit_flip
import alex
from alex import FlashDataset, FlashNet
BATCH_SIZE = 64

def get_flip_error(val):
    while True:
        pos = random.randint(0, 20)
        error =  bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    error = min(10e+3, error)
    error = max(-10e+3, error)
    return error

def read_data(filename):
    data = np.load(filename)
    if data.ndim == 3:      # (N, nx, ny)
        data = np.expand_dims(data, axis=1)    # (N, channels, nx, ny)
    print(filename, data.shape)
    return data

def create_0_propagation_dataset(clean_data):
    error_data = np.copy(clean_data)
    for i in range(len(error_data)):
        x = random.randint(20, 40)
        y = random.randint(20, 40)
        error_data[i, 0, x, y] = get_flip_error(error_data[i, 0, x, y])
    print("0-propagation dataset:", error_data.shape)
    return error_data

def load_model(model_file):
    model = None
    if os.path.isfile(model_file):
        print "Load existing model"
        model = torch.load(model_file)
    else:
        model = FlashNet().double()
        if torch.cuda.is_available():
            print "Have CUDA!!!"
            model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "More than one GPU card!!!"
            model = nn.DataParallel(model)
    print model
    return model

if __name__ == "__main__":

    model_file = "./sedov_train_0.model"
    model = load_model(model_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", help="Train the model")
    parser.add_argument("-s", "--evaluating_file", help="Evaluating the model with single file")
    parser.add_argument("-m", "--evaluating_path", help="Evaluating the model with multiple files")
    args = parser.parse_args()

    if args.train_file:     # Train with 0-propagation dataset
        print("Training 0-propagation ...")
        # Training
        clean_data = read_data(args.train_file)
        error_data = create_0_propagation_dataset(clean_data)
        trainset = FlashDataset(clean_data, error_data)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        alex.training(model, train_loader, epochs=5)
        torch.save(model, model_file)
        # Testing
        error_data = create_0_propagation_dataset(clean_data)
        testset = FlashDataset(clean_data, error_data)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
        alex.evaluating(model, test_loader)
    elif args.evaluating_file:
        print("Evaluating with a signle file...")
        clean_data = read_data(args.evaluating_file)
        error_data = create_0_propagation_dataset(clean_data)
        testset = FlashDataset(clean_data, error_data)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
        alex.evaluating(model, test_loader)
    elif args.evaluating_path:
        print("Evaluating with multiple files...")
        for error_data_file in glob.iglob(args.evaluating_path+"/*.npy"):
            testset = FlashDataset(clean_data_file, error_data_file)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
            alex.evaluating(model, test_loader)

