import os, shutil
import skimage
from process_mix.alg.config import Config
import scipy.io as sio
import numpy as np
import cv2


def pad_image_to_fit_patch_size(image, patch_size=(256, 256)):
    h, w = image.shape[:2]
    pad_h = (patch_size[1] - h % patch_size[1]) % patch_size[1]
    pad_w = (patch_size[0] - w % patch_size[0]) % patch_size[0]
    fill_value = 255 if len(image.shape) == 2 else [255, 255, 255]
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=fill_value)
    return padded_image

def extract_patches(image, patch_size=(256, 256)):
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h, patch_size[1]):
        for x in range(0, w, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)
    return patches

class CropPatches(Config):

    def __init__(self):
        super(CropPatches, self).__init__()

    def crop_diffusion_mix(self, masked_img_list=None, masks=None):
        idx = 0
        save_list = []
        for path in [self.patch_gt_path, self.patch_mask_path, self.patch_gt_path_noinpaint]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)

        if masked_img_list is None:
            sorted_name_list = sorted(os.listdir(self.masked_label_path))
        else:
            sorted_name_list = sorted(masked_img_list.keys())

        for file in sorted_name_list:
            name_stem = file.split(".")[0]
            if masked_img_list is None:
                gt_file_path = os.path.join(self.gt_path, name_stem + "_masked.png")
                gt_np = skimage.io.imread(gt_file_path)
            else:
                gt_np = np.array(masked_img_list[name_stem], dtype="uint8")
            gt_np = pad_image_to_fit_patch_size(gt_np)
            if masks is None:
                mask_file_path = os.path.join(self.mask_path, name_stem + "_mask_1.png")
                mask_np = skimage.io.imread(mask_file_path)
            else:
                mask_np = np.array(masks[name_stem], dtype="uint8")
            mask_np = pad_image_to_fit_patch_size(mask_np)
            img_patches = extract_patches(gt_np)
            msk_patches = extract_patches(mask_np)
            for i, mask_patch in enumerate(msk_patches):
                if 0 in mask_patch:
                    gt_patch_path = os.path.join(self.patch_gt_path, f"{idx}.png")
                    skimage.io.imsave(gt_patch_path, img_patches[i], check_contrast=False)
                    mask_patch_path = os.path.join(self.patch_mask_path, f"{idx}.png")
                    skimage.io.imsave(mask_patch_path, mask_patch, check_contrast=False)
                    save_list.append(idx)
                else: # no need to inpaint, different path to save
                    gt_patch_path = os.path.join(self.patch_gt_path_noinpaint, f"{idx}.png")
                    skimage.io.imsave(gt_patch_path, img_patches[i], check_contrast=False)
                idx += 1
        return save_list

    def crop_diffusion_2(self, save_list, maskm_list=None, mask2_list=None, mask3_list=None):
        for path in [self.patch_masks_m_path, self.patch_masks_2_path, self.patch_masks_3_path]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)

        idx = 0
        if maskm_list is None:
            sorted_name_list = sorted(os.listdir(self.masked_label_path))
        else:
            sorted_name_list = sorted(maskm_list.keys())
        for file in sorted_name_list:
            name_stem = file.split(".")[0]
            if maskm_list is None:
                mask_m_file_path = os.path.join(self.masks_m_path, name_stem + "_mask_nuclei2.png")
                file_np = skimage.io.imread(mask_m_file_path)
            else:
                file_np = np.array(maskm_list[name_stem], dtype="uint8")
            local_num = self.patch_loop(file_np, self.patch_masks_m_path, idx, save_list)

            if mask2_list is None:
                mask_2_file_path = os.path.join(self.masks_2_path, name_stem + "_mask_2srd.png")
                file_np = skimage.io.imread(mask_2_file_path)
            else:
                file_np = np.array(mask2_list[name_stem], dtype="uint8")
            local_num = self.patch_loop(file_np, self.patch_masks_2_path, idx, save_list)

            if mask3_list is None:
                mask_3_file_path = os.path.join(self.masks_3_path, name_stem + "_mask_3srd.png")
                file_np = skimage.io.imread(mask_3_file_path)
            else:
                file_np = np.array(mask3_list[name_stem], dtype="uint8")

            local_num = self.patch_loop(file_np, self.patch_masks_3_path, idx, save_list)
            idx += local_num

    def patch_loop(self, mask_np, patch_masks_path, start_idx, save_list):
        idx = start_idx
        mask_np = pad_image_to_fit_patch_size(mask_np)
        msk_patches = extract_patches(mask_np)
        local_idx = 0
        for i, mask_patch in enumerate(msk_patches):
            patch_path = os.path.join(patch_masks_path, f"{idx}.png")
            if idx in save_list:
                skimage.io.imsave(patch_path, mask_patch, check_contrast=False)
            idx += 1
            local_idx += 1
        return local_idx


    def count_class_num(self, masked_label_list=None):
        cls_num = 5

        if masked_label_list is None:
            list = os.listdir(self.masked_label_path)
        else:
            list = [x + ".mat" for x in masked_label_list.keys()]

        class_dist = {}
        for c in range(1, cls_num):
            class_dist[c] = 0
        for file in sorted(list):
            ann = sio.loadmat(os.path.join(self.label_path, file))
            type_map, class_arr = ann['type_map'], np.squeeze(ann['inst_type'])
            if self.dataset == "consep":
                class_arr[(class_arr == 3) | (class_arr == 4)] = 3
                class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
            elif self.dataset == "glysac":
                class_arr[(class_arr == 2) | (class_arr == 9) | (class_arr == 10)] = 1
                class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7) | (class_arr == 4)] = 2
                class_arr[(class_arr == 3) | (class_arr == 8)] = 3
            for c in range(1, cls_num):
                class_dist[c] += len(np.where(class_arr[class_arr == c])[0])
        print(class_dist)

        class_dist = {}
        for c in range(1, cls_num):
            class_dist[c] = 0
        if masked_label_list is not None:
            print("masked: ")
            for name_stem in sorted(masked_label_list.keys()):
                ann = masked_label_list[name_stem]
                type_map, class_arr = ann['type_map'], np.squeeze(ann['inst_type'])
                if self.dataset == "consep":
                    class_arr[(class_arr == 3) | (class_arr == 4)] = 3
                    class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
                elif self.dataset == "glysac":
                    class_arr[(class_arr == 2) | (class_arr == 9) | (class_arr == 10)] = 1
                    class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7) | (class_arr == 4)] = 2
                    class_arr[(class_arr == 3) | (class_arr == 8)] = 3
                for c in range(1, cls_num):
                    class_dist[c] += len(np.where(class_arr[class_arr == c])[0])
        else:
            for file in sorted(list):
                ann = sio.loadmat(os.path.join(self.masked_label_path, file))
                type_map, class_arr = ann['type_map'], np.squeeze(ann['inst_type'])
                if self.dataset == "consep":
                    class_arr[(class_arr == 3) | (class_arr == 4)] = 3
                    class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
                elif self.dataset == "glysac":
                    class_arr[(class_arr == 2) | (class_arr == 9) | (class_arr == 10)] = 1
                    class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7) | (class_arr == 4)] = 2
                    class_arr[(class_arr == 3) | (class_arr == 8)] = 3
                for c in range(1, cls_num):
                    class_dist[c] += len(np.where(class_arr[class_arr == c])[0])
        print(class_dist)


if __name__ == "__main__":
    cp = CropPatches()
    cp.count_class_num()

