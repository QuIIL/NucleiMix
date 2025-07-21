import math
import os
from matplotlib import pyplot as plt
import skimage
import imgaug as ia
from imgaug import augmenters as iaa
import torchvision.ops.roi_align as roi_align
from scipy.ndimage import binary_dilation
import random
import cv2
import torch
import scipy.io as sio
import shutil
import numpy as np
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops


def get_bbox(inst_map, id):
    msk = (inst_map == id).astype(np.uint8)
    bbox = bounding_box(msk)
    return bbox

def remap_label(pred, label_list=None, pred_centroid=None, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred).astype(int))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_label = []
    new_centroid = []
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
        if label_list is not None:
            new_label.append(label_list[inst_id - 1])
        if pred_centroid is not None:
            new_centroid.append(pred_centroid[inst_id - 1])

    new_label = np.array(new_label).astype("int32")
    new_centroid = np.array(new_centroid).astype("int32")

    if label_list is not None and pred_centroid is not None:
        return new_pred, new_label, new_centroid


def get_geo_info(mask):
    imgray = mask.copy()
    # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda l: l.shape[0], reverse=True)
    rect = cv2.minAreaRect(contours[0])
    return rect


def label_preprocess(inst_map, inst_type, inst_cen, margin, dataset_name):
    inst_map_out, inst_type_out, inst_cen_out = np.array(inst_map), np.array(inst_type), np.array(inst_cen)
    if margin is not None:
        for i, cent in enumerate(inst_cen_out):  # old_value = 30
            if ((cent[1] < margin) or
                    (cent[1] > (inst_map.shape[0] - margin)) or
                    (cent[0] < margin) or
                    (cent[0] > (inst_map.shape[1] - margin))):
                inst_type_out[i] = 0
    # inst_map_out, inst_type_out, inst_cen_out = remap_label(inst_map_out, inst_type_out, inst_cen_out)
    return inst_map_out, inst_type_out, inst_cen_out


def process_pos(
        aug_fre, model, im_transform, inst_list,
        padded_img, msk_padding, ann,
        train_datapoints, train_feature_np_augs, train_descriptors,
        crop_size, basename, dataset_name, minor_class
):
    inst_map, inst_type, inst_cent = ann["inst_map"], ann["inst_type"], ann["inst_centroid"]
    inst_map_out, inst_type_out, inst_cen_out = label_preprocess(inst_map, inst_type, inst_cent, None, dataset_name)
    aug_count = 0
    while aug_count <= aug_fre:
        for inst_id in inst_list:
            if inst_type[inst_id - 1] != minor_class:
                continue
            inner_bbox, middle_bbox, input_crop, nei_info, glcm_features, size = nuclei_cen_crop(get_augmentation(aug_count),
                                                                                           inst_map_out,
                                                                                           inst_type,
                                                                                           inst_cent,
                                                                                           inst_id,
                                                                                           padded_img,
                                                                                           crop_size,
                                                                                           msk_padding,
                                                                                           dataset_name
                                                                                           )
            # im_transform = transforms.Compose([transforms.ToTensor()])
            input_torch = im_transform(input_crop.copy()).unsqueeze(0).cuda()
            inst_features = model(input_torch)
            # bbox_ts = torch.FloatTensor([0, 0, 0, crop_size, crop_size]).unsqueeze(0).cuda()
            # inst_features = roi_align(inst_features, bbox_ts, output_size=1,
            #                           spatial_scale=inst_features.shape[-1] / crop_size, aligned=True)
            inst_features = inst_features.view(-1).detach().cpu().numpy()
            inst_features = np.concatenate([inst_features, np.array(glcm_features)])
            # translate to the location on unpadded im_np
            cmin, rmin, cmax, rmax = inner_bbox
            inner_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
            cmin, rmin, cmax, rmax = middle_bbox
            middle_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
            datapoint = {
                "inner_bbox": inner_bbox,
                "middle_bbox": middle_bbox,
                "inst_id": inst_id,
                "basename": basename,
                "nei_info": nei_info,
                "size": size
            }
            train_feature_np_augs[aug_count].append(inst_features)
            if aug_count == 0:
                train_datapoints.append(datapoint)
                train_descriptors.append(np.array(list(nei_info.values())))
        aug_count += 1


def process_neg(augmentor, model, im_transform, inst_list, visited_list,
                padded_img, msk_padding, ann,
                prediction_datapoints_1, prediction_feature_np_1, prediction_descriptors_1,
                prediction_datapoints_2, prediction_feature_np_2, prediction_descriptors_2,
                crop_size, basename, dataset, stride, flex, minor_class, major_class):
    inst_map, inst_type, inst_cent = ann["inst_map"], ann["inst_type"], ann["inst_centroid"]
    inst_map_out, inst_type_out, inst_cent = label_preprocess(inst_map, inst_type, inst_cent, None, dataset)
    img_width, img_height = inst_map.shape[1], inst_map.shape[0]
    # inst_list = np.random.choice(inst_list, len(inst_list) // 2, replace=False)
    # minor_list = [x for x in inst_list if inst_type[x - 1] == 1]
    major_list = [x for x in inst_list if inst_type[x - 1] in major_class]
    # minor_list = np.array(minor_list)
    major_list = np.random.choice(major_list, len(major_list) // 1, replace=False)
    # major_list = np.array([])
    for inst_id in major_list.astype(int):
        # if inst_type_out[inst_id - 1] == 0 or inst_type_out[inst_id - 1] == 1:  # center at major nuclei
        #     continue
        inner_bbox, middle_bbox, input_crop, nei_info, glcm_features, size = nuclei_cen_crop(None,
                                                                                       inst_map_out,
                                                                                       inst_type,
                                                                                       inst_cent,
                                                                                       inst_id,
                                                                                       padded_img,
                                                                                       crop_size,
                                                                                       msk_padding,
                                                                                       dataset
                                                                                       )
        # within stride x stride range won't be available for picking nucleus
        box_center = [(inner_bbox[0] + inner_bbox[2])//2, (inner_bbox[1] + inner_bbox[3])//2]
        # visited_area = [box_center[0] - stride, box_center[1] - stride, box_center[0] + stride, box_center[1] + stride]
        # visited_list += np.unique(inst_map[visited_area[1]: visited_area[3], visited_area[0]: visited_area[2]]).tolist()
        # im_transform = transforms.Compose([transforms.ToTensor()])
        input_torch = im_transform(input_crop.copy()).unsqueeze(0).cuda()
        inst_features = model(input_torch)
        # bbox_ts = torch.FloatTensor([0, 0, 0, crop_size, crop_size]).unsqueeze(0).cuda()
        # inst_features = roi_align(inst_features, bbox_ts, output_size=1,
        #                           spatial_scale=inst_features.shape[-1] / crop_size, aligned=True)
        inst_features = inst_features.view(-1).detach().cpu().numpy()
        inst_features = np.concatenate([inst_features, np.array(glcm_features)])
        # translate to the location on unpadded im_np
        cmin, rmin, cmax, rmax = inner_bbox
        inner_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
        cmin, rmin, cmax, rmax = middle_bbox
        middle_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
        datapoint = {
            "inner_bbox": inner_bbox,
            "middle_bbox": middle_bbox,
            "inst_id": inst_id,
            "basename": basename,
            "nei_info": nei_info,
            "size": size
        }
        prediction_datapoints_1.append(datapoint)
        prediction_feature_np_1.append(inst_features)
        prediction_descriptors_1.append(np.array(list(nei_info.values())))

    # center at bg
    # for r in range(flex, img_width - flex - crop_size + 1, stride):
    #     for c in range(flex, img_height - flex- crop_size + 1, stride):
    #         w_flex = random.sample(range(-flex, flex), 1)[0]
    #         h_flex = random.sample(range(-flex, flex), 1)[0]
    #         crop_box = [c + w_flex, r + h_flex, c + crop_size + w_flex, r + crop_size + h_flex]

    x_range = (30, img_width - 30)
    y_range = (30, img_height - 30)
    # x_coordinates = np.random.uniform(x_range[0], x_range[1], 400)
    # y_coordinates = np.random.uniform(y_range[0], y_range[1], 400)
    x_coordinates = np.random.randint(x_range[0], x_range[1], 400)
    y_coordinates = np.random.randint(y_range[0], y_range[1], 400)
    coordinates = np.column_stack((x_coordinates, y_coordinates))
    for r, c in coordinates:
        crop_box = [c, r, c + crop_size, r + crop_size]
        inner_bbox, middle_bbox, input_crop, nei_info, glcm_features, inst_id = bg_cent_crop(None,
                                                                                             inst_map_out,
                                                                                             inst_type,
                                                                                             inst_cent,
                                                                                             padded_img,
                                                                                             msk_padding,
                                                                                             crop_box,
                                                                                             crop_size,
                                                                                             visited_list,
                                                                                             dataset
                                                                                             )
        if inst_id != 0:
            continue
        # im_transform = transforms.Compose([transforms.ToTensor()])
        input_torch = im_transform(input_crop.copy()).unsqueeze(0).cuda()
        inst_features = model(input_torch)
        # bbox_ts = torch.FloatTensor([0, 0, 0, crop_size, crop_size]).unsqueeze(0).cuda()
        # inst_features = roi_align(inst_features,
        #                           bbox_ts,
        #                           output_size=1,
        #                           spatial_scale=inst_features.shape[-1] / crop_size,
        #                           aligned=True
        #                           )
        inst_features = inst_features.view(-1).detach().cpu().numpy()
        inst_features = np.concatenate([inst_features, np.array(glcm_features)])
        cmin, rmin, cmax, rmax = inner_bbox
        inner_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
        has_negative = any(n < 0 for n in inner_bbox)
        if has_negative:
            print("negative box")
            continue
        cmin, rmin, cmax, rmax = middle_bbox
        middle_bbox = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2
        datapoint = {
            "inner_bbox": inner_bbox,
            "middle_bbox": middle_bbox,
            "inst_id": inst_id,
            "basename": basename,
            "nei_info": nei_info
        }
        prediction_datapoints_2.append(datapoint)
        prediction_feature_np_2.append(inst_features)
        prediction_descriptors_2.append(np.array(list(nei_info.values())))


def bg_cent_crop(augmentor, inst_map, inst_type, inst_cen, img_np, msk_padding, middle_bbox, crop_size, visited_list, dataset_name):
    input_augs, shape_augs = iaa.Sequential([]), iaa.Sequential([])
    if augmentor is not None:
        input_augs, shape_augs = augmentor
        input_augs = input_augs.to_deterministic()
        shape_augs = shape_augs.to_deterministic()
    if msk_padding is not None:
        inst_map = msk_padding(image=inst_map)['image']
    cur_cent = [(middle_bbox[0] + middle_bbox[2]) / 2, (middle_bbox[1] + middle_bbox[3]) / 2]
    img_crop = img_np[middle_bbox[1]: middle_bbox[3], middle_bbox[0]: middle_bbox[2]][None, ...]
    inst_map_crop = inst_map[middle_bbox[1]: middle_bbox[3], middle_bbox[0]: middle_bbox[2]][None, ...]
    # # skimage.io.imshow(img_crop[0])
    # plt.show()
    img_crop = input_augs(images=img_crop)
    img_crop = shape_augs(images=img_crop)[0]
    # # skimage.io.imshow(img_crop)
    # plt.show()
    inst_map_crop = shape_augs(images=inst_map_crop)[0]

    closest_nuclei_index = np.argpartition(np.linalg.norm(inst_cen[:] - cur_cent, axis=1), 1)[0]
    closest_nuclei_bbox = get_bbox(inst_map, closest_nuclei_index + 1)
    closest_nuclei_width, closest_nuclei_height = closest_nuclei_bbox[2] - closest_nuclei_bbox[0], closest_nuclei_bbox[3] - closest_nuclei_bbox[1]
    # TODO only in the case of macro in monusac
    # if inst_type[closest_nuclei_index][0] != 3:
    #     closest_nuclei_width *= 5
    #     closest_nuclei_height *= 5
    inner_bbox = [
        int(cur_cent[0] - closest_nuclei_width // 2),
        int(cur_cent[1] - closest_nuclei_height // 2),
        int(cur_cent[0] - closest_nuclei_width // 2 + closest_nuclei_width),
        int(cur_cent[1] - closest_nuclei_height // 2 + closest_nuclei_height)
    ]
    cmin, rmin, cmax, rmax = inner_bbox
    img_crop[rmin - middle_bbox[1]: rmax - middle_bbox[1], cmin - middle_bbox[0]: cmax - middle_bbox[0]] = 0
    # # skimage.io.imshow(img_crop)
    # plt.show()
    # # skimage.io.imshow(img_np[rmin: rmax + 1, cmin: cmax + 1])
    # plt.show()
    # refine the inner box
    covered_inst, covered_inst_cnt = np.unique(
        inst_map_crop[rmin - middle_bbox[1]: rmax - middle_bbox[1], cmin - middle_bbox[0]: cmax - middle_bbox[0]]
        , return_counts=True
    )
    inst_id = covered_inst[np.argmax(covered_inst_cnt)]

    if dataset_name == "consep":
        class_num = 4
    elif dataset_name == "glysac":
        class_num = 3
    else:
        class_num = 4
    nei_info = dict()
    for k in range(1, class_num + 1):
        nei_info[k] = 0
    for i in np.unique(inst_map_crop):
        if i == 0:
            continue
        type = int(inst_type[int(i - 1)][0])
        nei_info[type] = nei_info.get(type) + 1

    if sum(list(nei_info.values())) > 0:
        for i in nei_info.keys():
            nei_info[i] = nei_info[i]/sum(list(nei_info.values()))
    glcm_features = get_glcm_features(img_crop)
    return inner_bbox, middle_bbox, img_crop, nei_info, glcm_features, inst_id


def nuclei_cen_crop(augmentor, inst_map, inst_type, inst_cent, inst_id, img_np, crop_size, msk_padding, dataset_name):
    input_augs, shape_augs = iaa.Sequential([]), iaa.Sequential([])
    if augmentor is not None:
        input_augs, shape_augs = augmentor
        input_augs = input_augs.to_deterministic()
        shape_augs = shape_augs.to_deterministic()
    if msk_padding is not None:
        inst_map = msk_padding(image=inst_map)['image']
    msk = (inst_map == inst_id).astype(np.uint8)
    cmin, rmin, cmax, rmax = bounding_box(msk)
    # # skimage.io.imshow(inst_map)
    # plt.show()
    # # skimage.io.imshow(msk)
    # plt.show()
    [cent_x, cent_y], _, _ = get_geo_info(msk)
    # [cent_x, cent_y] = inst_cent[inst_id - 1]
    middle_bbox = (
        int(max(cent_x - crop_size // 2, 0)),
        int(max(cent_y - crop_size // 2, 0)),
        int(max(cent_x - crop_size // 2, 0) + crop_size),
        int(max(cent_y - crop_size // 2, 0) + crop_size)
    )
    img_crop = img_np[middle_bbox[1]: middle_bbox[3], middle_bbox[0]: middle_bbox[2]][None,...]
    inst_map_crop = inst_map[middle_bbox[1]: middle_bbox[3], middle_bbox[0]: middle_bbox[2]][None,...]
    # images = np.array(
    #     [ia.quokka(size=(64, 64)) for _ in range(32)],
    #     dtype=np.uint8
    # )
    # # skimage.io.imshow(img_crop[0])
    # plt.show()
    img_crop = input_augs(images=img_crop)
    img_crop = shape_augs(images=img_crop)[0]
    # # skimage.io.imshow(img_crop)
    # plt.show()
    inst_map_crop = shape_augs(images=inst_map_crop)[0]
    # # skimage.io.imshow(inst_map_crop)
    # plt.show()
    msk = (inst_map_crop == inst_id).astype(np.uint8)
    size = msk.sum()
    msk = binary_dilation(msk, iterations=1).astype(np.uint8)
    if not (msk.sum() == 0 and augmentor is not None):
        cmin, rmin, cmax, rmax = bounding_box(msk)
    nuclei_width = cmax - cmin
    nuclei_height = rmax - rmin
    # cover nuclei with bbox
    img_crop[rmin: rmax, cmin: cmax] = 0
    # # skimage.io.imshow(img_crop)
    # plt.show()
    # inner box in crop local -> coordinates in original image
    inner_bbox = [
        cmin + middle_bbox[0],
        rmin + middle_bbox[1],
        cmax + middle_bbox[0],
        rmax + middle_bbox[1]
    ]

    if dataset_name == "consep":
        class_num = 4
    elif dataset_name == "glysac":
        class_num = 3
    else:
        class_num = 4
    nei_info = dict()
    for k in range(1, class_num + 1):
        nei_info[k] = 0
    for i in np.unique(inst_map_crop):
        if i == 0:
            continue
        type = int(inst_type[int(i - 1)][0])
        nei_info[type] = nei_info.get(type) + 1
    # normalize
    nei_sum = sum(list(nei_info.values()))
    for i in nei_info.keys():
        nei_info[i] = nei_info[i]/nei_sum

    glcm_features = get_glcm_features(img_crop)
    return inner_bbox, middle_bbox, img_crop, nei_info, glcm_features, size


def get_glcm_features(img):
    glcm_features = []
    glcm = graycomatrix(np.mean(img, axis=2).astype(np.uint8), distances=[2], angles=[0], levels=256)
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    glcm_features.append(dissimilarity)
    glcm_features.append(homogeneity)
    glcm_features.append(energy)
    glcm_features.append(asm)
    return glcm_features


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    # return [rmin, rmax, cmin, cmax]
    return [cmin, rmin, cmax, rmax]


def get_augmentation(mode, rng=8, input_shape=(270,270)):
    if mode == 0:
        iaa.Sequential([]), iaa.Sequential([])

    ####
    def add_to_hue(images, random_state, parents, hooks, range=(-5, 5)):
        """Perturbe the hue of input images."""
        img = images[0]  # aleju input batch as default (always=1 in our case)
        hue = random_state.uniform(*range)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if hsv.dtype.itemsize == 1:
            # OpenCV uses 0-179 for 8-bit images
            hsv[..., 0] = (hsv[..., 0] + hue) % 180
        else:
            # OpenCV uses 0-360 for floating point images
            hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
        ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        ret = ret.astype(np.uint8)
        return [ret]
    ####
    def add_to_saturation(images, random_state, parents, hooks, range=(-0.3, 0.3)):
        """Perturbe the saturation of input images."""
        img = images[0]  # aleju input batch as default (always=1 in our case)
        value = 1 + random_state.uniform(*range)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
        ret = np.clip(ret, 0, 255)
        ret = ret.astype(np.uint8)
        return [ret]
    ####
    def add_to_contrast(images, random_state, parents, hooks, range=(0.75, 1.25)):
        """Perturbe the contrast of input images."""
        img = images[0]  # aleju input batch as default (always=1 in our case)
        value = random_state.uniform(*range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        ret = img * value + mean * (1 - value)
        ret = np.clip(img, 0, 255)
        ret = ret.astype(np.uint8)
        return [ret]
    ####
    def add_to_brightness(images, random_state, parents, hooks, range=(-3, 3)):
        """Perturbe the brightness of input images."""
        img = images[0]  # aleju input batch as default (always=1 in our case)
        value = random_state.uniform(*range)
        ret = np.clip(img + value, 0, 255)
        ret = ret.astype(np.uint8)
        return [ret]
    ia.seed(rng)
    shape_augs = [
        # * order = ``0`` -> ``cv2.INTER_NEAREST``
        # * order = ``1`` -> ``cv2.INTER_LINEAR``
        # * order = ``2`` -> ``cv2.INTER_CUBIC``
        # * order = ``3`` -> ``cv2.INTER_CUBIC``
        # * order = ``4`` -> ``cv2.INTER_CUBIC``
        # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
        iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.1), "y": (0.8, 1.2)},
            # translate by -A to +A percent (per axis)
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            shear=(-5, 5),  # shear by -5 to +5 degrees
            rotate=(-mode*90 + 1, mode*90 - 1),  # rotate by -90 to +90 degrees
            order=0,  # default 0 use nearest neighbour
            backend="cv2",  # opencv for fast processing
            seed=rng,
        ),
        # set position to 'center' for center crop
        # else 'uniform' for random crop
        iaa.CropToFixedSize(
            input_shape[0], input_shape[1], position="center"
        ),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ]
    input_augs = [
        iaa.Sequential(
            [
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(0.5, iaa.Lambda(
                    seed=rng,
                    func_images=add_to_hue,
                )),
                iaa.Sometimes(0.5, iaa.Lambda(
                    seed=rng,
                    func_images=add_to_saturation,
                )),
                iaa.Sometimes(0.5, iaa.Lambda(
                    seed=rng,
                    func_images=add_to_brightness,
                )),
                iaa.Sometimes(0.5, iaa.Lambda(
                    seed=rng,
                    func_images=add_to_contrast,
                )),
            ],
            random_order=True,
        ),
    ]
    return iaa.Sequential(input_augs), iaa.Sequential(shape_augs)


def filter_test_neg(test_neg_datapoints, test_neg_all, top_k):
    test_nei_info = [x["nei_info"] for x in test_neg_datapoints]
    test_nei_info = [x[1] / (sum(x) + 1e-6) for x in test_nei_info]
    sorted_idx = sorted(range(len(test_nei_info)), key=lambda k: test_nei_info[k])[:top_k]

    test_neg_datapoints = np.array(test_neg_datapoints)[sorted_idx]
    test_neg_all = test_neg_all[sorted_idx]
    return test_neg_datapoints, test_neg_all


def process_nuclei_patches(
        model, im_transform, inst_list,
        padded_img, msk_padding, ann,
        nuclei_datapoints, nuclei_feature_np, nuclei_labels,
        crop_size, basename, dataset_name, class_list, minor_list
):
    inst_map, inst_type, inst_cent = ann["inst_map"], ann["inst_type"], ann["inst_centroid"]
    inst_map_out, inst_type_out, inst_cen_out = label_preprocess(inst_map, inst_type, inst_cent, None, dataset_name)
    for inst_id in inst_list:
        if inst_type[inst_id - 1] not in class_list:
            continue
        inner_bbox, crop_box, input_crop, nei_info, glcm_features, size = cen_crop(
            inst_map_out,
            inst_type_out,
            inst_cen_out,
            inst_id,
            padded_img,
            crop_size,
            msk_padding,
            dataset_name,
            minor_list
        )

        input_tensor = im_transform(input_crop.copy()).unsqueeze(0).cuda()
        inst_features = model(input_tensor)
        inst_features = inst_features.view(-1).detach().cpu().numpy()
        inst_features = np.concatenate([inst_features, np.array(glcm_features)])
        datapoint = {
            "inner_bbox": inner_bbox,
            "middle_bbox": crop_box,
            "inst_id": inst_id,
            "basename": basename,
            "label": nei_info,
            "size": size
        }
        nuclei_datapoints.append(datapoint)
        nuclei_feature_np.append(inst_features)
        nuclei_labels.append(nei_info)


def cen_crop(inst_map, inst_type, inst_cent, inst_id, img_np, crop_size, msk_padding, dataset_name, minor_list):
    msk = (inst_map == inst_id).astype(np.uint8)
    size = msk.sum()
    cmin, rmin, cmax, rmax = bounding_box(msk)
    inner_bbox = [
        cmin,
        rmin,
        cmax,
        rmax,
    ]
    if msk_padding is not None:
        inst_map = msk_padding(image=inst_map)['image']
    msk = (inst_map == inst_id).astype(np.uint8)
    [cent_x, cent_y], _, _ = get_geo_info(msk)

    crop_box = ( # cmin, rmin, cmax, rmax
        int(cent_x - crop_size // 2),
        int(cent_y - crop_size // 2),
        int(cent_x - crop_size // 2) + crop_size,
        int(cent_y - crop_size // 2) + crop_size
    )
    img_crop = img_np[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
    inst_map_crop = inst_map[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
    msk_crop = msk[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
    cmin, rmin, cmax, rmax = bounding_box(msk_crop)

    width, height = cmax - cmin, rmax - rmin
    cmin, rmin, cmax, rmax = max(cmin - width // 2, 0), max(rmin - height//2, 0), min(cmax + width//2, inst_map_crop.shape[1]), min(rmax + height//2, inst_map_crop.shape[0])

    rm_inst_list = np.unique(inst_map_crop[rmin: rmax, cmin: cmax])
    if 0 in rm_inst_list:
        rm_inst_list = np.delete(rm_inst_list, 0)

    for rm_inst in rm_inst_list:
        cur_mask = (inst_map_crop == rm_inst)
        cur_mask = binary_dilation(cur_mask, iterations=2)
        msk_crop[cur_mask] = 1

    msk_crop[msk_crop > 0] = 255
    img_crop = cv2.inpaint(img_crop, msk_crop, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cmin, rmin, cmax, rmax = crop_box
    crop_box = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2

    neighbour_info = dict()
    for i in np.unique(inst_map_crop):
        if i == 0:
            continue
        type = int(inst_type[int(i - 1)][0])
        binary_type = 0 if type in minor_list else 1
        if neighbour_info.get(binary_type) is None:
            neighbour_info[binary_type] = 0
        neighbour_info[binary_type] = neighbour_info.get(binary_type) + 1

    glcm_features = get_glcm_features(img_crop)
    return inner_bbox, crop_box, img_crop, neighbour_info, glcm_features, size


def process_void_patches(model, im_transform, inst_list,
                         padded_img, msk_padding, ann,
                         void_datapoints, void_feature_np, void_labels,
                         crop_size, basename, dataset, minor_class):
    inst_map, inst_type, inst_cent = ann["inst_map"], ann["inst_type"], ann["inst_centroid"]
    inst_map_out, inst_type_out, inst_cent = label_preprocess(inst_map, inst_type, inst_cent, None, dataset)
    img_width, img_height = inst_map.shape[1], inst_map.shape[0]

    random_fre = int(math.floor(img_width*img_height/2000))
    x_range = (crop_size//2, img_width + crop_size//2)
    y_range = (crop_size//2, img_height + crop_size//2)
    x_coordinates = np.random.randint(x_range[0], x_range[1], random_fre)
    y_coordinates = np.random.randint(y_range[0], y_range[1], random_fre)
    coordinates = np.column_stack((x_coordinates, y_coordinates))
    for c, r in coordinates:
        inner_bbox, middle_bbox, input_crop, nei_info, glcm_features, inst_id = void_cent_crop(
            inst_map_out,
            inst_type,
            inst_cent,
            padded_img,
            msk_padding,
            (c, r),
            crop_size,
            dataset,
            minor_class
        )
        if inst_id != 0: # ignore nuclei
            continue

        input_torch = im_transform(input_crop.copy()).unsqueeze(0).cuda()
        inst_features = model(input_torch)
        inst_features = inst_features.view(-1).detach().cpu().numpy()
        inst_features = np.concatenate([inst_features, np.array(glcm_features)])
        has_negative = any(n < 0 for n in inner_bbox)
        if has_negative:
            print("abandon border box")
            continue

        datapoint = {
            "inner_bbox": inner_bbox,
            "middle_bbox": middle_bbox,
            "inst_id": inst_id,
            "basename": basename,
            "nei_info": nei_info
        }
        void_datapoints.append(datapoint)
        void_feature_np.append(inst_features)
        void_labels.append(nei_info)

def void_cent_crop(inst_map, inst_type, inst_cen, img_np, msk_padding, cur_cent, crop_size, dataset_name, minor_class):
    (c, r) = cur_cent
    cur_cent_unpadding = (c - crop_size//2, r - crop_size//2)
    closest_nuclei_index = np.argpartition(np.linalg.norm(inst_cen[:] - cur_cent_unpadding, axis=1), 1)[0]
    closest_nuclei_bbox = get_bbox(inst_map, closest_nuclei_index + 1)
    closest_nuclei_width, closest_nuclei_height = max(closest_nuclei_bbox[2] - closest_nuclei_bbox[0], 3), \
                                                  max(closest_nuclei_bbox[3] - closest_nuclei_bbox[1], 3)
    inner_bbox = [
        max(int(cur_cent_unpadding[0] - closest_nuclei_width // 2), 0),
        max(int(cur_cent_unpadding[1] - closest_nuclei_height // 2), 0),
        min(int(cur_cent_unpadding[0] + closest_nuclei_width // 2), inst_map.shape[1]),
        min(int(cur_cent_unpadding[1] + closest_nuclei_height // 2), inst_map.shape[0])
    ]
    cmin, rmin, cmax, rmax = inner_bbox
    covered_inst, covered_inst_cnt = np.unique(
        inst_map[rmin: rmax, cmin: cmax]
        , return_counts=True
    )
    if len(covered_inst_cnt) == 0:
        print()
    inst_id = covered_inst[np.argmax(covered_inst_cnt)]

    if msk_padding is not None:
        inst_map = msk_padding(image=inst_map)['image']
    crop_box = [c - crop_size // 2, r - crop_size // 2, c + crop_size // 2, r + crop_size // 2]
    img_crop = img_np[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
    inst_map_crop = inst_map[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
    cmin, rmin, cmax, rmax = crop_box
    crop_box = cmin - crop_size // 2, rmin - crop_size // 2, cmax - crop_size // 2, rmax - crop_size // 2

    # only in the case of macro in monusac we need a large box for macrophages (label 3)
    if dataset_name == "monusac" and inst_type[closest_nuclei_index][0] != 3:
        closest_nuclei_width *= 5
        closest_nuclei_height *= 5

    neighbour_info = dict()
    for i in np.unique(inst_map_crop):
        if i == 0:
            continue
        type = int(inst_type[int(i - 1)][0])
        binary_type = 0 if type in minor_class else 1
        if neighbour_info.get(binary_type) is None:
            neighbour_info[binary_type] = 0
        neighbour_info[binary_type] = neighbour_info.get(binary_type) + 1

    glcm_features = get_glcm_features(img_crop)
    return inner_bbox, crop_box, img_crop, neighbour_info, glcm_features, inst_id


def process_ordinal_labels(minor_labels, major_labels, void_labels):
    minor_len, major_len, void_len = len(minor_labels), len(major_labels), len(void_labels)
    minor_list, major_list, void_list = [], [], []
    for i in range(minor_len):
        if minor_labels[i].get(0) == None:
            minor_labels[i][0] = 0
            minor_list.append(0)
        elif minor_labels[i].get(1) == None:
            minor_labels[i][1] = 0
            minor_list.append(1)
        else:
            minor_list.append((minor_labels[i].get(0)/minor_len) / ((minor_labels[i].get(0)/minor_len) + (minor_labels[i].get(1)/major_len)))

    for i in range(major_len):
        if major_labels[i].get(0) == None:
            major_labels[i][0] = 0
            major_list.append(0)
        elif major_labels[i].get(1) == None:
            major_labels[i][1] = 0
            major_list.append(1)
        else:
            major_list.append((major_labels[i].get(0)/minor_len) / ((major_labels[i].get(0)/minor_len) + (major_labels[i].get(1)/major_len)))

    for i in range(void_len):
        if void_labels[i].get(0) == None:
            void_labels[i][0] = 0
            void_list.append(0)
        elif void_labels[i].get(1) == None:
            void_labels[i][1] = 0
            void_list.append(1)
        else:
            void_list.append((void_labels[i].get(0)/minor_len) / ((void_labels[i].get(0)/minor_len) + (void_labels[i].get(1)/major_len)))

    fig, axs = plt.subplots(3, 1)
    axs[0].hist(minor_list, bins=10, edgecolor='black', range=(0, 1))
    axs[1].hist(major_list, bins=10, edgecolor='black', range=(0, 1))
    axs[2].hist(void_list, bins=10, edgecolor='black', range=(0, 1))
    plt.show()

    return minor_list, major_list, void_list
