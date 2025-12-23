import numpy as np
import torch
from hausdorff import hausdorff_distance
import surface_distance as surfdist
from collections import OrderedDict
import json

class Compute_matrices:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.preds = OrderedDict()
        self.gts = OrderedDict()
        self.spacings = OrderedDict()
        self.names = []

    def add_slices(self, pred_slice, gt_slice, patient_name, spacing):
        dict_preds = self.preds
        dict_gts = self.gts
        if patient_name not in dict_gts.keys():
            self.names.append(patient_name)
            dict_gts[patient_name] = []
            dict_preds[patient_name] = []
            self.spacings[patient_name] = spacing
        dict_gts[patient_name].append(gt_slice)
        dict_preds[patient_name].append(pred_slice)
        self.preds =  dict_preds
        self.gts = dict_gts

    def get_matrices(self, smooth=0.0001):
        dict_preds = self.preds
        dict_gts = self.gts
        number = len(self.names)
        tps, fps, tns, fns = np.zeros((number, self.n_classes)), np.zeros((number, self.n_classes)), np.zeros((number, self.n_classes)), np.zeros((number, self.n_classes))
        hds, assds, ravds = np.zeros((number, self.n_classes)), np.zeros((number, self.n_classes)), np.zeros((number, self.n_classes))
        for i in range(number):
            predi = dict_preds[self.names[i]]
            predi = np.stack(predi, axis=-1)
            gti = dict_gts[self.names[i]]
            gti = np.stack(gti, axis=-1)
            for c in range(1, self.n_classes):
                labelc = gti==c
                predictc = predi==c
                surface_distances = surfdist.compute_surface_distances(labelc, predictc, spacing_mm=(1.0, 1.0, 1.0))
                avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                mavg_surf_dist = (avg_surf_dist[0] + avg_surf_dist[1])/2
                hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
                dice3d = surfdist.compute_dice_coefficient(predictc, labelc)
                ravd = abs(np.sum(predictc) - np.sum(labelc))/np.sum(labelc)
                tp, fp, tn, fn = self.get_confusion_matrix(predictc, labelc)

                hds[i, c] = hd_dist_95
                assds[i, c] = mavg_surf_dist
                ravds[i, c] = ravd
                tps[i, c], fps[i, c], tns[i, c], fns[i, c] = tp, fp, tn, fn

        mean_hds = np.mean(hds, axis=0)
        mean_assds = np.mean(assds, axis=0)
        mean_ravds = np.mean(ravds, axis=0)

        dice = 2 * tps / (2 * tps + fps + fns + smooth)
        mean_dice = np.mean(dice, axis=0)
        iou = (tps + smooth) / (fps + tps + fns + smooth)
        mean_iou = np.mean(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        mean_acc = np.mean(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        mean_se = np.mean(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        mean_sp = np.mean(sp, axis=0)

        return mean_dice*100, mean_hds, mean_assds, mean_ravds*100, mean_iou*100, mean_acc*100, mean_se*100, mean_sp*100

    def get_confusion_matrix(self, pred, gt, smooth=1e-5):   
        N = 1
        pred[pred >= 1] = 1
        gt[gt >= 1] = 1
        pred_flat = pred.reshape(N, -1)
        gt_flat = gt.reshape(N, -1)
        TP = (pred_flat * gt_flat).sum(1)
        FN = gt_flat.sum(1) - TP
        pred_flat_no = (pred_flat + 1) % 2
        gt_flat_no = (gt_flat + 1) % 2
        TN = (pred_flat_no * gt_flat_no).sum(1)
        FP = pred_flat.sum(1) - TP
        return TP, FP, TN, FN           



def dice_coefficient(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # if (pred.sum() + gt.sum()) == 0:
    #     return 1
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N

def sespiou_coefficient(pred, gt, smooth=1e-5):
    """ computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    #pred_flat = pred.view(N, -1)
    #gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N

def sespiou_coefficient2(pred, gt, smooth=1e-5):
    """ computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    #pred_flat = pred.view(N, -1)
    #gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth)/(TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2*Precision*Recall/(Recall + Precision +smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N, Acc.sum()/N, F1.sum()/N, Precision.sum()/N, Recall.sum()/N

def get_matrix(pred, gt, smooth=1e-5):
    """ computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    return TP, FP, TN, FN