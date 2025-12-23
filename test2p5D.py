import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval
from utils.visualization import network_inputs_visual
import torch.nn.functional as F
from utils.utils_2p5Dv2 import ImageToImage2p5DRM_Dynamic_ACDC, ImageToImage2p5DRM_Dynamic_INSTANCE, ImageToImage2p5DRM_Dynamic_CHAOS, ImageToImage2p5DRM_Dynamic_LITS, ImageToImage2p5DRM_Dynamic_MosMed, ImageToImage2p5DRM_Dynamic_Prostate, ImageToImage2p5DRM_Dynamic_Task09Spleen



def main():
    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='ACSFormerv2', type=str, help='type of model')
    parser.add_argument('--task', default='INSTANCE', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.mode = "eval"
    opt.modelname = args.modelname
    opt.eval_mode = "patient2p5RM2"
    opt.load_path = "./xxxxxxxxxxxx"
    print(opt.load_path)
    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_2p5Dv2 import JointTransform2p5D, ImageToImage2p5D
    else:
        from utils.utils import JointTransform2D, ImageToImage2D

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================

    tf_test = JointTransform2p5D(img_size=opt.img_size, crop=opt.crop, p_flip=0, p_gama=0, color_jitter_params=None,
                              long_mask=True, half_resize=False)  # image reprocessing
    test_dataset = ImageToImage2p5DRM_Dynamic_INSTANCE(opt.data_path, opt.test_split, tf_test, opt.classes, opt.assist_slice_number, opt.img_size, opt.assis_slice_inter)  # return image, mask, and filename
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes, assist_slice_number=opt.assist_slice_number)
    model.to(device)
    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ============================================= evaluate =============================================
    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    mean_dice, mean_hds, mean_assds, mean_ravds, mean_iou, mean_acc, mean_se, mean_sp = get_eval(testloader, model, criterion=criterion, opt=opt)
    print(mean_dice, mean_hds, mean_assds, mean_ravds)
    print(mean_iou, mean_acc, mean_se, mean_sp)
    print(np.mean(mean_dice[1:]),  np.mean(mean_iou[1:]),  np.mean(mean_assds[1:]), np.mean(mean_hds[1:]),np.mean(mean_ravds[1:]),np.mean(mean_acc[1:]), np.mean(mean_se[1:]), np.mean(mean_sp[1:]))
    with open("meanresult.txt", "a+") as file:
        file.write(args.task + " " + args.modelname + " " + "\n")
        file.write('%.2f'%(np.mean(mean_dice[1:])) + " ")
        file.write('%.2f'%(np.mean(mean_hds[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_assds[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_ravds[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_iou[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_acc[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_se[1:]))  + " ")
        file.write('%.2f'%(np.mean(mean_sp[1:]))  + "\n")
    with open("experiments.txt", "a+") as file:
        file.write(args.task + " " + args.modelname + " " + "\n")
        for i in range(1, opt.classes):
            file.write('%.2f'%(mean_dice[i]) + " ")
            file.write('%.2f'%(mean_hds[i])  + " ")
            file.write('%.2f'%(mean_assds[i])  + " ")
            file.write('%.2f'%(mean_ravds[i])  + " ")
            file.write('%.2f'%(mean_iou[i])  + " ")
            file.write('%.2f'%(mean_acc[i])  + " ")
            file.write('%.2f'%(mean_se[i])  + " ")
            file.write('%.2f'%(mean_sp[i])  + "\n")

if __name__ == '__main__':
    main()



