import matplotlib.pylab as plt
import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.imagename import read_img_name

def network_inputs_visual(center_input,  assist_slices,
                      out_dir='./visual_result/input', 
                      save_feature=True,  
                      slice_number=5,
                      show_feature=False, 
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = center_input.shape
    under_input = assist_slices[:, :slice_number, :, :, :]
    over_input = assist_slices[:, slice_number:, :, :, :]
    if b > 6:
        b = 6
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(slice_number+1)+30), dtype=np.uint8) + 255
        figure[10:h + 10, 10 + (w + 20) * 0: 10 + (w + 20) * 0 + w] = center_input[i, 0, :, :]*255
        for j in range(1, (slice_number+1)):
            overj = over_input[:, j-1, :, :, :]
            figure[10:h + 10, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = overj[i, 0, :, :]*255
        for j in range(1, (slice_number+1)):
            underj = under_input[:, j-1, :, :, :]
            figure[30+h:30+h+h, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = underj[i, 0, :, :]*255
        if save_feature:
            cv2.imwrite(out_dir + '/' + 'batch_' + str(i) + '.png', figure)
        if show_feature:
            cv2.imshow("attention-" + str(c), figure)
            cv2.waitKey(0)


def mid_supervise_result(ori_input, mid_result, 
                    out_dir='./visual_result/mid_supervise',
                    save_feature=True,
                    show_feature=False):
    b, c, h, w = ori_input.shape
    mid_result = F.interpolate(mid_result, size=(h, w),mode='bilinear',align_corners=False)
    mid_result = F.softmax(mid_result, dim=1)
    mid_result = mid_result.detach().cpu().numpy()
    mid_class = 1 - mid_result[:, 0, :, :]
    for i in range(b):
        ori_i = ori_input[i, 0, :, :].cpu().detach().numpy()
        norm_ori_i = np.zeros(ori_i.shape)
        cv2.normalize(ori_i, norm_ori_i, 0, 255, cv2.NORM_MINMAX)
        norm_ori_i = np.asarray(norm_ori_i, dtype=np.uint8)
        norm_ori_i = np.expand_dims(norm_ori_i, axis=2).repeat(3, axis=2)

        mid_i = mid_class[i, :, :]
        norm_mid_i = np.zeros(mid_i.shape)
        cv2.normalize(mid_i, norm_mid_i, 0, 255, cv2.NORM_MINMAX)
        norm_mid_i = np.asarray(norm_mid_i, dtype=np.uint8)

        heat_img = cv2.applyColorMap(norm_mid_i, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

        img_add = cv2.addWeighted(norm_ori_i, 0.3, heat_img, 0.7, 0)
        if save_feature:
            cv2.imwrite(out_dir + '/' + 'batch_' + str(i) + '.png', img_add)
        if show_feature:
            cv2.imshow("attention-" + str(c), img_add)
            cv2.waitKey(0)


def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
        # img_r[seg0 == i] = table[i + 1 - 1, 0]
        # img_g[seg0 == i] = table[i + 1 - 1, 1]
        # img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) # ACDC
    #img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) # ISIC
    img = cv2.addWeighted(img_ori0, 0.3, overlay, 0.7, 0) # Prostate
          
    fulldir = opt.result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_class_roi(heatmap):
    b, g, q, n = heatmap.shape
    heatmap = rearrange(heatmap, 'b g q (h w) -> b g q h w', h=32, w=32)
    heatmap = np.array(heatmap.cpu())
    heatmap = heatmap*1024
    heatmap = np.uint8(heatmap)
    x = np.random.randint(10)
    for j in range(g):
        for i in range(0, q):
            class_i = heatmap[0, j, i, :, :]
            mask = cv2.resize(class_i, (256, 256))
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET) 
            imagefile = read_img_name()
            filename = imagefile.split("/")[-1].split(".")[0]
            img = cv2.imread(imagefile)
            add_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0) 
            fulldir = "./visual_result/ACDC/heatmap/"
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            cv2.imwrite(fulldir + filename + "_head" + str(j) + "_" + str(x) + "_class_" + str(i) + ".png", add_img)
            a = 0


def visual_refer_region(predbbox, scale=8):
    imagefile = read_img_name()
    filename = imagefile.split("/")[-1].split(".")[0]
    #predbbox=np.array(predbbox[0, :, :]) # b 9 4
    for i in range(1, 4):
        img = cv2.imread(imagefile)
        cv2.rectangle(img, (int(predbbox[i, 1]*scale), int(predbbox[i, 0]*scale)), (int(predbbox[i, 3]*scale), int(predbbox[i, 2]*scale)), (0, 0, 255), 2) ## x1y1,x2y2,BGR
        save_path = "./visual_result/ACDC/bbox/"
        if not os.path.isdir(save_path):
                os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, filename + "_class_" + str(i) + '.png'), img)

