import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval
from utils.utils_2p5Dv2 import ImageToImage2p5DRM_Dynamic_ACDC, ImageToImage2p5DRM_Dynamic_Prostate
from utils.visualization import network_inputs_visual
import torch.nn.functional as F
from utils.bbox_tools import loc_loss
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='ACSFormer', type=str, help='type of model')
    parser.add_argument('--task', default='Prostate', help='task or dataset name')
    parser.add_argument('--slices_ds', default=False, type=bool, help='whether utilize multi slices to supervise')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.mode = "eval"
    opt.modelname = 'ACSFormer'
    opt.load_path = "./xxxxxxxxxxxxx.pth"
    print(opt.load_path)

    opt.eval_mode = "patient2p5RM2"
    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_2p5Dv2 import JointTransform2p5D, ImageToImage2p5DRM, ImageToImage2p5DRM_Dynamic
    else:
        from utils.utils import JointTransform2D, ImageToImage2D


    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 30  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================

    tf_train = JointTransform2p5D(img_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None,
                                long_mask=True, half_resize=False)  # image reprocessing
    tf_val = JointTransform2p5D(img_size=opt.img_size, crop=opt.crop, p_flip=0, p_gama=0, color_jitter_params=None,
                              long_mask=True, half_resize=False)  # image reprocessing
    train_dataset = ImageToImage2p5DRM_Dynamic_Prostate(opt.data_path, opt.train_split, tf_train, opt.classes, opt.assist_slice_number, opt.img_size, opt.assis_slice_inter)
    val_dataset = ImageToImage2p5DRM_Dynamic_Prostate(opt.data_path, opt.test_split, tf_val, opt.classes, opt.assist_slice_number, opt.img_size, opt.assis_slice_inter)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes, assist_slice_number=opt.assist_slice_number)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_path))

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
  
    if opt.mode == "eval":
        mean_dice, mean_hds, mean_assds, mean_ravds, mean_iou, mean_acc, mean_se, mean_sp = get_eval(valloader, model, criterion=criterion, opt=opt)
        print(np.mean(mean_dice[1:]), np.mean(mean_iou[1:]), np.mean(mean_assds[1:]), np.mean(mean_ravds[1:]))
        print(mean_hds, mean_acc, mean_se, mean_sp)
    else:
        dices, mean_dice, val_losses, patient_dices = get_eval(valloader, model, criterion, opt)
        print("------------------------- evaluation result -------------------------")
        print("mean dices of all classes:\n", mean_dice)
        #print("mean dices of each class:\n", dices[7], dices[4], dices[3], dices[2], dices[5], dices[8], dices[1], dices[6]) 
        print(dices)   
        print("class dices of each patient:\n", patient_dices)
  

if __name__ == '__main__':
    main()



