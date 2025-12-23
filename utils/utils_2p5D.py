import os
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from collections import OrderedDict
import json


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def read_slice_number(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def norm_zscore(tx):
    tx = np.array(tx)
    tx = tx.astype(np.float32)
    tx_flat = tx.flatten()
    if np.sum(tx_flat) > 0:
        tx_flat_no = tx_flat[tx_flat > 0]
        tx_normal = (tx - np.mean(tx_flat_no)) / (np.std(tx_flat_no) + 1e-5)
        tx_normal[tx == 0] = 0
    else:
        tx_normal = tx
    return tx_normal


class JointTransform2p5D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, z_score=False, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_affine=0, half_resize=True,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.zscore = z_score
        self.img_size = img_size
        self.half_resize = half_resize
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_affine = p_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        random_dict = OrderedDict()
        #  gamma enhancement
        r_gama = np.random.rand()
        random_dict["r_gama"] = r_gama
        if r_gama < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            random_dict["gama_g"] = g
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            random_dict["crop_crop"] = [i, j, h, w]
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        r_flip = np.random.rand()
        random_dict["r_flip"] = r_flip
        if r_flip < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        r_rota = np.random.rand()
        random_dict["r_rota"] = r_rota
        if r_rota < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            random_dict["rota_angle"] = angle
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        r_scale = np.random.rand()
        random_dict["r_scale"] = r_scale
        if r_scale < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            random_dict["scale_scale"] = scale
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), 2), F.resize(mask, (new_h, new_w), 0)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            random_dict["scale_crop"] = [i, j, h, w]
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        r_gaussn = np.random.rand()
        random_dict["r_gaussian"] = r_gaussn
        if r_gaussn < self.p_gaussn:
            ns = np.random.randint(3, 15)
            random_dict["gaussian_ns"] = ns
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            random_dict["gaussian_noise"] = noise
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        r_contr = np.random.rand()
        random_dict["r_contr"] = r_contr
        if r_contr < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            random_dict["contr_tf"] = contr_tf
            image = contr_tf(image)
        # random distortion
        r_distortion = np.random.rand()
        random_dict["r_distortion"] = r_distortion
        if r_distortion < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            random_dict["distortion_distortion"] = distortion
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        r_affine = np.random.rand()
        random_dict["r_affine"] = r_affine
        if r_affine < self.p_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            random_dict["affine_params"] = affine_params
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # if resize to 1/2 size?
        if self.half_resize:
            #image, mask = F.resize(image, (self.img_size//2, self.img_size//2), 2), F.resize(mask, (self.img_size//2, self.img_size//2), 0)
            image = F.resize(image, (self.img_size//2, self.img_size//2), 2)
        # transforming to tensor
        if self.zscore:
            image = norm_zscore(image)
            image = torch.from_numpy(image[None, :, :])
        else:
            image = F.to_tensor(image)

        mask_mini = F.resize(mask, (self.img_size//16, self.img_size//16), 0)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            mask_mini = F.to_tensor(mask_mini)
        else:
            mask = to_long_tensor(mask)
            mask_mini = to_long_tensor(mask_mini)
        return image, mask, mask_mini, random_dict

    def transform_image(self, image, random_dict):
        #  gamma enhancement
        r_gama = random_dict["r_gama"]
        if r_gama < self.p_gama:
            c = 1
            g = random_dict["gama_g"]
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image = F.to_pil_image(image)
        # random crop
        if self.crop:
            crop_params = random_dict["crop_crop"]
            i, j, h, w = crop_params[0], crop_params[1], crop_params[2], crop_params[3]
            image = F.crop(image, i, j, h, w)
        # random horizontal flip
        r_flip = random_dict["r_flip"]
        if r_flip < self.p_flip:
            image = F.hflip(image)
        # random rotation
        r_rota = random_dict["r_rota"]
        if r_rota < self.p_rota:
            angle = random_dict["rota_angle"]
            image = F.rotate(image, angle)
        # random scale and center resize to the original size
        r_scale = random_dict["r_scale"]
        if r_scale < self.p_scale:
            scale = random_dict["scale_scale"]
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image = F.resize(image, (new_h, new_w), 2)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            scale_crop_params = random_dict["scale_crop"]
            i, j, h, w = scale_crop_params[0], scale_crop_params[1], scale_crop_params[2], scale_crop_params[3]
            image = F.crop(image, i, j, h, w)
        # random add gaussian noise
        r_gaussn = random_dict["r_gaussian"]
        if r_gaussn < self.p_gaussn:
            noise = random_dict["gaussian_noise"]
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        r_contr = random_dict["r_contr"]
        if r_contr < self.p_contr:
            contr_tf = random_dict["contr_tf"]
            image = contr_tf(image)
        # random distortion
        r_distortion = random_dict["r_distortion"]
        if r_distortion < self.p_distortion:
            distortion = random_dict["distortion_distortion"]
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        r_affine = random_dict["r_affine"]
        if r_affine < self.p_affine:
            affine_params = random_dict["affine_params"]
            image = F.affine(image, *affine_params)
        # if resize to 1/2 size?
        if self.half_resize:
            #image, mask = F.resize(image, (self.img_size//2, self.img_size//2), 2), F.resize(mask, (self.img_size//2, self.img_size//2), 0)
            image = F.resize(image, (self.img_size//2, self.img_size//2), 2)
        # transforming to tensor
        if self.zscore:
            image = norm_zscore(image)
            image = torch.from_numpy(image[None, :, :])
        else:
            image = F.to_tensor(image)

        return image

class ImageToImage2p5D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, split='train1', joint_transform: Callable = None, classes=2, assist_slice=3, img_size=256,
                 inter=3, ncslice=10, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'imgL')
        self.label_path = os.path.join(dataset_path, 'labelL')
        patient_slice_path = os.path.join(dataset_path, "patient_slice.json")
        self.patient_slice_number = read_slice_number(patient_slice_path)
        self.assist_slice_number = assist_slice
        self.slice_class_number = ncslice
        self.inter = inter
        self.classes = classes
        self.one_hot_mask = one_hot_mask
        self.img_size = img_size
        id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        image = cv2.imread(os.path.join(self.img_path, id_ + '.png'), 0)
        mask = cv2.imread(os.path.join(self.label_path, id_ + '.png'), 0)
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)

        if self.classes == 2:
            mask[mask > 1] = 1
        if self.joint_transform:
            image, mask, mask_mini, random_dict = self.joint_transform(image, mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            mask_mini = torch.zeros((self.one_hot_mask, mask_mini.shape[1], mask_mini.shape[2])).scatter_(0, mask_mini.long(), 1)
        # ---------------------------- preprocess the assist slice ------------------------
        over_slice, under_slice = [], []
        patient_id, slice_id = id_[:3], int(id_[3:6])
        for i in range(1, self.assist_slice_number+1):
            overi = min(slice_id + i*self.inter, self.patient_slice_number[patient_id]-1)
            overi_id = patient_id + str(int(overi)).zfill(3)
            #print("overid", overi_id)
            imagei = cv2.imread(os.path.join(self.img_path, overi_id + '.png'), 0)
            imagei = correct_dims(imagei)
            if self.joint_transform:
                imagei = self.joint_transform.transform_image(imagei, random_dict)
            over_slice.append(imagei)
        for i in range(self.assist_slice_number, 0, -1):
            underi = max(slice_id - i*self.inter, 0)
            underi_id = patient_id + str(int(underi)).zfill(3)
            #print("underid", underi_id)
            imagei = cv2.imread(os.path.join(self.img_path, underi_id + '.png'), 0)
            imagei = correct_dims(imagei)
            if self.joint_transform:
                imagei = self.joint_transform.transform_image(imagei, random_dict)
            under_slice.append(imagei)
        over_slices = torch.stack(over_slice, dim=0)
        under_slices = torch.stack(under_slice, dim=0)
        assist_slices = torch.cat([under_slices, over_slices], dim=0)
        return image, mask, mask_mini, assist_slices, id_ + '.png'

def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}
