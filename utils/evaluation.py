# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from utils.visualization import visual_segmentation
from hausdorff import hausdorff_distance
from utils.metrics import Compute_matrices
import json

def obtain_patien_id(filename):
    filename = filename.split('.')[0]
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_2d_slice(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        with torch.no_grad():
            predict = model(input_image)

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for i in range(0, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            dices[i] += metrics.dice_coefficient(pred_i, gt_i)
            del pred_i, gt_i
    dices = dices / (batch_idx + 1)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    return dices, mean_dice, val_losses

def eval_2d_patient(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 2000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))

    for batch_idx, (input_image, ground_truth, mask_mini, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        test_img_path = os.path.join(opt.data_path + '/img', image_filename)
        from utils.imagename import keep_img_name
        keep_img_name(test_img_path)

        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        with torch.no_grad():
            predict = model(input_image)
        predict = F.interpolate(predict, size=(opt.img_size, opt.img_size), mode='bilinear',align_corners=False) # (opt.img_size, opt.img_size)
        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        patientid = int(image_filename[:3])
        flag[patientid] = 1
        for i in range(1, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            if hd > hds[patientid, i]:
                hds[patientid, i] = hd
            tps[patientid, i] += tp
            fps[patientid, i] += fp
            tns[patientid, i] += tn
            fns[patientid, i] += fn

        if opt.visual:
            visual_segmentation(seg, image_filename, opt)
        
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :]#/ flag[flag>0][:, None]
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c

    # --------------- just unify with transunet -----------
    # for p in range(int(patients)):
    #     for c in range(1, opt.classes):
    #         if tps[p, c] == 0 and fps[p, c] > 0:
    #             patient_dices[p, c] = 1
    # -----------------------------------------------------

    dices = np.mean(patient_dices, axis=0)  # c
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    if opt.mode == "train":
        return dices, mean_dice, val_losses, patient_dices
    else:
        dice_mean = np.mean(patient_dices*100, axis=0)
        dices_std = np.std(patient_dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth)*100 # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)*100
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)*100
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)*100
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)

        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean


def eval_2d_patient2p5RM2(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    matrix_tool = Compute_matrices(n_classes=opt.classes)
    with open(os.path.join(opt.data_path, "spacing.json"), 'r') as load_f:
        spacing_dict = json.load(load_f)

    for batch_idx, (input_image, ground_truth, mask_mini, assist_slices, gt_bbox, gt_label, filenames) in enumerate(valloader):
        if isinstance(filenames[0], str):
            image_filename = filenames
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        test_img_path = os.path.join(opt.data_path + '/img', image_filename[0])
        from utils.imagename import keep_img_name
        keep_img_name(test_img_path)

        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        assist_slices = Variable(assist_slices.to(device=opt.device))
        with torch.no_grad():
            #predict, _, _, _ = model(assist_slices)
            predict, _, _ = model(assist_slices)
        #predict = F.interpolate(predict, size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)
        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            matrix_tool.add_slices(seg[j, :, :], gt[j, :, :], str(patientid), spacing_dict[str(patientid)])
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)

    mean_dice, mean_hds, mean_assds, mean_ravds, mean_iou, mean_acc, mean_se, mean_sp = matrix_tool.get_matrices()
    if opt.mode=="train":
        return mean_dice, np.mean(mean_dice[1:]), np.mean(mean_hds[1:]), val_losses
    else:
        return mean_dice, mean_hds, mean_assds, mean_ravds, mean_iou, mean_acc, mean_se, mean_sp



def get_eval(valloader, model, criterion, opt):
    if opt.eval_mode == "slice":
        return eval_2d_slice(valloader, model, criterion, opt)
    elif opt.eval_mode == "patient":
        return eval_2d_patient(valloader, model, criterion, opt)
    elif opt.eval_mode == "patient2p5RM2":
        return eval_2d_patient2p5RM2(valloader, model, criterion, opt)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)