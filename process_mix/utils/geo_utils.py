import cv2
import cv2 as cv
import numpy as np
from scipy.ndimage.morphology import binary_dilation


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [cmin, rmin, cmax, rmax]


def fill_with_square_canvas(img, length, center2, rotate_box):
    canvas_center = (length, length)
    if img.shape[-1] == 3:
        canvas = np.zeros((2 * length, 2 * length, 3), dtype=np.uint8)
        canvas[
        round(canvas_center[1] - (center2[1] - rotate_box[1])): round(canvas_center[1] + (rotate_box[3] - center2[1])),
        round(canvas_center[0] - (center2[0] - rotate_box[0])): round(canvas_center[0] + (rotate_box[2] - center2[0])), :] \
            = img
    else:
        canvas = np.zeros((2 * length, 2 * length), dtype=np.uint8)
        canvas[
            round(canvas_center[1] - (center2[1] - rotate_box[1])): round(canvas_center[1] + (rotate_box[3] - center2[1])),
            round(canvas_center[0] - (center2[0] - rotate_box[0])): round(canvas_center[0] + (rotate_box[2] - center2[0]))
        ] = img
    return canvas


def get_angle(msk):
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=cv2.contourArea)
    # Make sure there's at least one contour found
    contour_points = contours.astype(np.float32)
    if len(contour_points) < 5:
        return 0
    # Fit the ellipse
    ellipse = cv2.fitEllipse(contour_points)
    return ellipse[-1]


def geo_augmentation(img, msk2, msk1_r2, cent2, bbox2):

    rotated_img = np.zeros_like(img)
    rotated_msk2 = np.zeros_like(msk2)

    length = max((bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1]))
    rotate_box = bbox2.copy()
    rotate_box[1] = round(max(0, cent2[1] - length))
    rotate_box[3] = round(min(img.shape[0], cent2[1] + length))
    rotate_box[0] = round(max(0, cent2[0] - length))
    rotate_box[2] = round(min(img.shape[1], cent2[0] + length))

    crop_msk2 = msk2[rotate_box[1]: rotate_box[3], rotate_box[0]: rotate_box[2]]
    crop_img = img[rotate_box[1]: rotate_box[3], rotate_box[0]: rotate_box[2], :]

    if rotate_box[2] - rotate_box[0] != rotate_box[3] - rotate_box[1]:  # if the crop area is not a perfect square
        crop_msk2 = fill_with_square_canvas(crop_msk2, length, cent2, rotate_box)
        crop_img = fill_with_square_canvas(crop_img, length, cent2, rotate_box)

    # flip
    flip_chance = 0.1
    if np.random.rand() < flip_chance:
        crop_msk2 = cv2.flip(crop_msk2, 0)
        crop_img = cv2.flip(crop_img, 0)
    if np.random.rand() < flip_chance:
        crop_msk2 = cv2.flip(crop_msk2, 1)
        crop_img = cv2.flip(crop_img, 1)

    center = ((rotate_box[2] - rotate_box[0]) // 2, (rotate_box[3] - rotate_box[1]) // 2)
    angle2 = get_angle(crop_msk2)
    angle1 = get_angle(msk1_r2)

    if abs(angle2 - angle1) > 30: # we do a rotation if the angle difference is larger than 30 degrees
        rotate_Matrix = cv.getRotationMatrix2D(center, angle2 - angle1, 1)
        crop_img = cv.warpAffine(crop_img, rotate_Matrix, (crop_img.shape[1], crop_img.shape[0]))
        crop_msk2 = cv.warpAffine(crop_msk2, rotate_Matrix, (crop_msk2.shape[1], crop_msk2.shape[0]))

    translate_c2_c1(rotated_img, cent2, crop_img, center, [0, 0, crop_img.shape[1], crop_img.shape[0]])
    translate_c2_c1(rotated_msk2, cent2, crop_msk2, center, [0, 0, crop_img.shape[1], crop_img.shape[0]])
    return rotated_img, rotated_msk2, bounding_box(binary_dilation(rotated_msk2, iterations=2))


def get_geo_info(mask):
    imgray = mask.copy()
    ret, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda l: l.shape[0], reverse=True)
    rect = cv.minAreaRect(contours[0])
    return rect


def paste_idx2_on_img1(
        inst_id1,
        img_1,
        msk1,
        cent1,
        bbox1,
        inst_id2,
        img_2,
        ann_2,
        masked,
        final_mask_nuclei2,
        final_mask_1,
        final_mask_2srd,
        final_mask_3srd,
        visited_list,
        inst_map,
        inst_map_out,
        type_map_out,
        cent_ann_out,
        inst_type_out,
        minor_list_list,
        is_major
):
    msk1_r2 = binary_dilation(msk1, iterations=2).astype(np.uint8)
    bbox1 = bounding_box(msk1_r2)
    inst_map_2, inst_type_2 = ann_2['inst_map'], ann_2['inst_type']
    msk2 = (inst_map_2 == inst_id2).astype(np.uint8)
    msk2_r2 = binary_dilation(msk2, iterations=2).astype(np.uint8)
    bbox2 = bounding_box(msk2_r2)

    cent2, _, _ = get_geo_info(msk2_r2)
    cent2 = (int(cent2[0]), int(cent2[1]))
    # flip and rotation
    rotated_nuclei_2, rotated_mask2, _ = geo_augmentation(img_2, msk2, msk1_r2, cent2, bbox2)
    img2, msk2 = rotated_nuclei_2, rotated_mask2

    msk2_r2 = binary_dilation(msk2, iterations=2).astype(np.uint8)
    bbox2 = bounding_box(msk2_r2)

    cen_c = (bbox2[2] + bbox2[0]) // 2
    cen_r = (bbox2[3] + bbox2[1]) // 2
    bbox1_w = bbox1[2] - bbox1[0]
    bbox1_h = bbox1[3] - bbox1[1]
    bbox2[0] = max(cen_c - bbox1_w // 2, 0)
    bbox2[2] = bbox2[0] + bbox1_w
    bbox2[1] = max(cen_r - bbox1_h // 2, 0)
    bbox2[3] = bbox2[1] + bbox1_h

    cent2, wh2, _ = get_geo_info(msk2_r2)
    cent2 = (int(cent2[0]), int(cent2[1]))
    img_2_color = np.array(img2)

    msk2_translated = np.zeros_like(inst_map_out).astype(np.uint8)
    translate_c2_c1(msk2_translated, cent1, msk2, cent2, bbox2)

    img2_translated = np.zeros_like(masked).astype(np.uint8)
    translate_c2_c1(img2_translated, cent1, img_2_color, cent2, bbox2)

    # For the area that uncovered by mask1
    msk2_r1_translated = binary_dilation(msk2_translated, iterations=1).astype(np.uint8)
    msk2_r2_translated = binary_dilation(msk2_r1_translated, iterations=1).astype(np.uint8)
    msk2_r3_translated = binary_dilation(msk2_r2_translated, iterations=1).astype(np.uint8)
    rw = max(wh2[0], wh2[1]) * 10
    msk2_rw_translated = binary_dilation(msk2_translated, iterations=int(rw)).astype(np.uint8)

    # we try to avoid the later paste to overlap
    visited_list += np.unique(msk2_rw_translated * inst_map).astype(int).tolist()
    cent1 = get_geo_info(msk2_r2_translated)[0]

    img2_translated = img2_translated * msk2_r1_translated[..., np.newaxis]

    mask_merged = msk2_r3_translated.copy()
    if inst_id1 > 0:
        mask_merged[msk1_r2 > 0] = 1
    mask_merged[mask_merged > 0] = 1
    masked = masked * (~mask_merged[..., np.newaxis].astype(bool)).astype(np.uint8)

    # paste minor on the black region
    masked += img2_translated

    final_mask_1[mask_merged == 1] = 0
    final_mask_nuclei2[msk2_r3_translated == 1] = 1
    final_mask_nuclei2[msk2_r1_translated == 1] = 0
    final_mask_2srd[msk2_r1_translated > 0] = 1
    final_mask_2srd[(msk2_r3_translated - msk2_r1_translated) == 1] = 0
    final_mask_3srd[msk2_translated > 0] = 1
    final_mask_3srd[(msk2_r2_translated - msk2_translated) == 1] = 0

    new_minor_idx = int(np.max(inst_map_out) + 1)
    inst_map_out[mask_merged > 0] = 0
    inst_map_out[msk2_translated > 0] = new_minor_idx
    type_map_out[mask_merged > 0] = 0
    type_map_out[msk2_translated > 0] = inst_type_2[inst_id2 - 1]
    inst_type_out = np.append(inst_type_out, inst_type_2[inst_id2 - 1])
    cent_ann_out = np.append(cent_ann_out, [cent1], axis=0)

    return visited_list, inst_map_out, type_map_out, cent_ann_out, inst_type_out, masked


def translate_c2_c1(image1, center1, image2, center2, bbox):
    # Calculate the shift
    (min_x, min_y, max_x, max_y) = bbox

    # Extract the bounding box region from image2
    bbox_region = image2[min_y:max_y, min_x:max_x]

    # Calculate new location in image1
    new_min_x = max(center1[0] - bbox_region.shape[1] // 2, 0)
    new_min_y = max(center1[1] - bbox_region.shape[0] // 2, 0)
    new_max_x = new_min_x + bbox_region.shape[1]
    new_max_y = new_min_y + bbox_region.shape[0]

    # Ensure new location does not exceed image1 boundaries
    new_max_x = min(new_max_x, image1.shape[1])
    new_max_y = min(new_max_y, image1.shape[0])

    # Resize the bbox_region if it exceeds the boundaries of image1
    if new_max_x - new_min_x != bbox_region.shape[1] or new_max_y - new_min_y != bbox_region.shape[0]:
        bbox_region = cv2.resize(bbox_region, (new_max_x - new_min_x, new_max_y - new_min_y))

    # Place the region in image1
    image1[new_min_y:new_max_y, new_min_x:new_max_x] = bbox_region
    return
