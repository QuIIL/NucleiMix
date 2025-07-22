import math
from scipy.ndimage import binary_dilation
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def get_bbox(inst_map, id):
    msk = (inst_map == id).astype(np.uint8)
    bbox = bounding_box(msk)
    return bbox


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
