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
    save_path_code = "_"

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.eval_mode = "patient2p5RM"
    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_2p5Dv2 import JointTransform2p5D, ImageToImage2p5DRM, ImageToImage2p5DRM_Dynamic
    else:
        from utils.utils import JointTransform2D, ImageToImage2D

    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    boardpath = opt.tensorboard_path + args.modelname + save_path_code + timestr
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

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
    val_dataset = ImageToImage2p5DRM_Dynamic_Prostate(opt.data_path, opt.val_split, tf_val, opt.classes, opt.assist_slice_number, opt.img_size, opt.assis_slice_inter)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes, assist_slice_number=opt.assist_slice_number)
    model.to(device)
    if opt.pre_trained:
        model.load_state_dict(torch.load(opt.load_path))

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.001, weight_decay=0.05)
    # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=4)
    lr = opt.learning_rate

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================== begin to train the model =============================================

    best_dice, loss_log = 0.0, np.zeros(opt.epochs + 1)
    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (input_image, ground_truth, mask_mini_slices, image_slices, gt_bbox, gt_label, *rest) in enumerate(trainloader):
            #network_inputs_visual(input_image, assist_slices, slice_number=opt.assist_slice_number)
            input_image = Variable(input_image.to(device=opt.device))
            ground_truth = Variable(ground_truth.to(device=opt.device))
            mask_mini_slices = Variable(mask_mini_slices.to(device=opt.device))
            image_slices = Variable(image_slices.to(device=opt.device))
            gt_class_loc = Variable(gt_bbox.to(device=opt.device))
            gt_class_label = Variable(gt_label.to(device=opt.device)).long()

            #print(rest[0])
            # ---------------------------------- forward ----------------------------------
            #output = model(input_image)
            output, mclassd5, mclassd4 = model(image_slices) # (b 1 256 256) and (b asn 1 256 256)

            if args.slices_ds:
                b, k, c, h, w = mclass.shape
                mclass = mclass.reshape(b * k, c, h, w)
                mask_mini_slices = mask_mini_slices.reshape(b * k, h, w)
            else:
                #mclass = mclass[:, 0, :, :, :]
                mask_mini_slices = mask_mini_slices[:, 0, :, :]

            mask_mini_slices_d5 = F.interpolate(mask_mini_slices[:, None, :, :].float(), (opt.img_size//16, opt.img_size//16), mode="nearest")[:, 0, :, :]
            train_loss_mini = criterion(mclassd5, mask_mini_slices_d5.long())
            train_loss_mini4 = criterion(mclassd4, mask_mini_slices)
            train_loss = 0.8*criterion(output, ground_truth) + 0.1*train_loss_mini + 0.1*train_loss_mini4


            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)

        #scheduler.step()
        #  ---------------------------- log the train progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        TensorWriter.add_scalar('loss/train_loss', train_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('lr/learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        loss_log[epoch] = train_losses / (batch_idx + 1)
        #  ----------------------------------- evaluate -----------------------------------
        if epoch % opt.eval_freq == 0:
            dices, mean_dice, val_losses, _ = get_eval(valloader, model, criterion, opt)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            print("dice of each class:", dices)
            TensorWriter.add_scalar('loss/val_loss', val_losses, epoch)
            TensorWriter.add_scalar('dice/val_dices', mean_dice, epoch)
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + save_path_code + '%s' % timestr + '_'+ str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

        if (epoch+1) % opt.save_freq == 0 or epoch == (opt.epochs - 1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + save_path_code + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()



