# This file is used to configure the training parameters for each task

class Config_ACDC:
    data_path = "../../dataset/cardiac/"
    save_path = "./checkpoints/ACDC/"
    result_path = "./result/ACDC/"
    tensorboard_path = "./tensorboard/ACDC/"
    visual_result_path = "./visual_result/ACDC/"
    load_path = "xxxxxxxxxxxxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 4                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "trainofficial"  # the file name of training set
    val_split = "valofficial"     # the file name of testing set
    test_split = "testofficial"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "CSFormer"

class Config_ICH:
    data_path = "../../../dataset/CT0/INSTANCE/"
    save_path = "./checkpoints/INSTANCE512/"
    result_path = "./result/INSTANCE/"
    tensorboard_path = "./tensorboard/INSTANCE/"
    visual_result_path = "./visual_result/INSTANCE/"
    load_path = "xxxxxxxxxxxxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 0.0001 #1e-3        # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 512              # the input size of model
    train_split = "train2"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test2"
    crop = None                   # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ACSFormer"

class Config_CHAOS:
    data_path = "../../dataset/CT/CHAOS/"
    save_path = "./checkpoints/CHAOS/"
    result_path = "./result/CHAOS/"
    tensorboard_path = "./tensorboard/CHAOS/"
    visual_result_path = "./visual_result/CHAOS/"
    load_path = "xxxxxxxxxxxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"
    crop = None                   # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ACSFormer"

class Config_Task09Spleen:
    data_path = "../../dataset/CT/Task09Spleen/"
    save_path = "./checkpoints/Task09Spleen/"
    result_path = "./result/Task09Spleen/"
    tensorboard_path = "./tensorboard/Task09Spleen/"
    visual_result_path = "./visual_result/Task09Spleen/"
    load_path = "xxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"
    crop = None                   # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ACSFormer"


class Config_MosMed:
    data_path = "../../dataset/CT0/MosMed/"
    save_path = "./checkpoints/MosMed/"
    result_path = "./result/MosMed/"
    tensorboard_path = "./tensorboard/MosMed/"
    visual_result_path = "./visual_result/MosMed/"
    load_path = "xxxxxxxxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"
    crop = None                   # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ACSFormer"


class Config_INSTANCE:
    data_path = "../../dataset/INSTANCEmini/"
    save_path = "./checkpoints/INSTANCEmini/"
    result_path = "./result/INSTANCEmini/"
    tensorboard_path = "./tensorboard/INSTANCEmini/"
    visual_result_path = "./visual_result/INSTANCEmini/"
    load_path = "xxxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3
    assis_slice_inter = 3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "CSFormer"


class Config_Prostate:
    data_path = "../../../dataset/Prostate/"
    save_path = "./checkpoints/Prostate/"
    result_path = "./result/Prostate/"
    visual_result_path = "./visual_result/Prostate/"
    tensorboard_path = "./tensorboard/Prostate/"
    load_path = "xxxxx"
    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4        # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 3                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "val"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 2              # the channel of input image
    eval_mode = "patient2p5"        # the mode when evaluate the model, slice level or patient level
    assist_slice_number = 3  #2
    assis_slice_inter = 3    #3
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "CSFormer"

# ==================================================================================================
def get_config(task="Synapse"):
    if task == "ACDC":
        return Config_ACDC()
    elif task == "Prostate":
        return Config_Prostate()
    elif task == "INSTANCE":
        return Config_ICH()
    elif task == "CHAOS":
        return Config_CHAOS()
    elif task == "Task09Spleen":
        return Config_Task09Spleen()
    elif task == "MosMed":
        return Config_MosMed()
