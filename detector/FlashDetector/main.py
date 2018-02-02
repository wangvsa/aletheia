import sys
import argparse
import classifier

FLASH_APP_DIR = "/home/wangchen/Sources/FLASH4.4/objects/"
FLASH_APPS = ["Blast2", "BlastBS", "BrioWu", "Cellular",
        "DMReflection", "RHD_Sod", "Sedov", "Sod"]

def fun_train(app, N):
    data_dir = FLASH_APP_DIR + app
    model_file = app + ".h5"        # existing model file
    classifier.train(model_file, data_dir, N, epochs=10)
    classifier.evaluation(model_file, data_dir)
    return 0

def fun_test(app):
    data_dir = FLASH_APP_DIR + app
    model_file = app + ".h5"        # existing model file
    classifier.evaluation(model_file, data_dir)
    return 0

def fun_testall():

    for app in FLASH_APPS:
        print "Testing", app
        fun_test(app)
    return 0



if __name__ == "__main__":

    # allow three actions: {train, test, testall}
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="action")
    sp_train = sp.add_parser("train", help="train a model")
    sp_test = sp.add_parser("test", help="test a model")
    sp_testall = sp.add_parser("testall", help="test all available models")

    # arguments for action "train"
    sp_train.add_argument("app", help="the app's name", choices=FLASH_APPS)
    sp_train.add_argument("--N", help="duplicate N times of dataset for training", type=int, default=1)

    # arguments for action "test"
    sp_test.add_argument("app", help="the app's name", choices=FLASH_APPS)

    args = parser.parse_args()
    if args.action == "train":
        fun_train(args.app, args.N)
    elif args.action == "test":
        fun_test(args.app)
    elif args.action == "testall":
        fun_testall()
