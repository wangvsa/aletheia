import sys
import classifier

if __name__ == "__main__":

    FLASH_APP_DIR = "/home/wangchen/Sources/FLASH4.4/objects/"
    apps = ["Blast2", "BlastBS", "BrioWu", "Cellular",
            "DMReflection", "RHD_Sod", "Sedov", "Sod"]



    if len(sys.argv) < 2 or sys.argv[1] not in apps:
        print "classifier.py app_name [N]"
        print "Supported app:", apps
        sys.exit(0)


    N = 1
    if len(sys.argv) == 3 :
        N = int(sys.argv[2])

    app = sys.argv[1]
    data_dir = FLASH_APP_DIR + app
    model_file = app + ".h5"        # existing model file
    print "data dir:", data_dir
    print "model file:", model_file

    classifier.train(model_file, data_dir, N, epochs=10)
    classifier.evaluation(model_file, data_dir)
    #classifier.predict(sys.argv[1])
