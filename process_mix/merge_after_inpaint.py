import numpy as np
import skimage
import os
import scipy.io as sio
from process_mix.alg.config import Config
from process_mix.utils.other_utils import check_dir_save
patch_size=(256, 256)


def reconstruct_image_from_patches(patches, original_image_shape, patch_size=(256, 256)):
    reconstructed_image = np.zeros(original_image_shape, dtype=np.uint8)
    idx = 0
    for y in range(0, original_image_shape[0], patch_size[1]):
        for x in range(0, original_image_shape[1], patch_size[0]):
            patch = patches[idx]
            height = patch.shape[0] if y + patch.shape[0] < original_image_shape[0] else original_image_shape[0] - y
            width = x + patch.shape[1] if x + patch.shape[1] < original_image_shape[1] else original_image_shape[1] - x
            reconstructed_image[y:y + patch.shape[0], x:x + patch.shape[1]] = patch[:height, :width]
            idx += 1
    return reconstructed_image

class MergePatches(Config):

    def get_full_img_name_list(self):
        full_file_path = sorted(os.listdir(self.masked_label_path))
        return full_file_path

    def merge_patch2full_img(self, full_img_name_list):

        def loop_patches(cur_patch_idx, img_np, patch_path):
            patch_size = 256
            step_back = 8
            for h in range(4):
                for w in range(4):
                    start_point_h = h * patch_size - h * step_back
                    start_point_w = w * patch_size - w * step_back
                    cur_patch_path = os.path.join(patch_path, f"{cur_patch_idx}.png")
                    if os.path.exists(cur_patch_path):
                        patch = skimage.io.imread(cur_patch_path)
                        img_np[start_point_h: start_point_h + patch_size,
                                start_point_w: start_point_w + patch_size, :] = patch
                        cur_patch_idx += 1
                    else:
                        raise Exception(f"{cur_patch_path} does not exist")

        # patch_num_per_img = 16
        cur_patch_idx = 0
        for full_idx, file_name in enumerate(full_img_name_list):
            basename = file_name.split('.')[0]
            file_name = basename + '.png'
            img_np = skimage.io.imread(os.path.join(self.gt_path, basename + '_masked.png'))
            h, w = img_np.shape[:2]
            pad_h = (patch_size[1] - h % patch_size[1]) % patch_size[1]
            pad_w = (patch_size[0] - w % patch_size[0]) % patch_size[0]
            # if self.dataset == "monusac":
            patch_num_per_img = ((h + pad_h)//patch_size[1]) * ((w + pad_w)//patch_size[0])
            patch_list = []
            for i in range(cur_patch_idx, cur_patch_idx + patch_num_per_img):
                cur_patch_path = os.path.join(self.inpainted_patch_path, f"{i}.png")
                if os.path.exists(cur_patch_path):
                    patch = skimage.io.imread(cur_patch_path)
                    patch_list.append(patch)
                else:
                    print(f"full image {full_idx} {file_name} lack of patch")
            # new_img = reconstruct_image_from_patches(patch_list, img_np.shape)
            # check_dir_save(self.mix_img_path, file_name, new_img)
            try:
                new_img = reconstruct_image_from_patches(patch_list, img_np.shape)
                check_dir_save(self.mix_img_path, file_name, new_img)
            except:
                print(f"Skip {file_name}")

            #     try:
            #         loop_patches(cur_patch_idx, img_np, self.inpainted_patch_path)
            #     except:
            #         print(f"full image {full_idx} {file_name} lack of patch")
            #     else:
            #         check_dir_save(self.mix_img_path, file_name, img_np)
            #
            cur_patch_idx += patch_num_per_img

    def merge_mix(self):
        if not os.path.isdir(self.mix_lb_path):
            os.makedirs(self.mix_lb_path)
        for masked_lb_file in os.listdir(self.masked_label_path,):
            mat_file = sio.loadmat(os.path.join(self.masked_label_path, masked_lb_file))
            sio.savemat(os.path.join(self.mix_lb_path, masked_lb_file), mat_file)

        if not os.path.isdir(self.mix_msk_path):
            os.makedirs(self.mix_msk_path)
        for msk_file in os.listdir(self.masks_m_path,):
            img_obj = skimage.io.imread(os.path.join(self.masks_m_path, msk_file))
            skimage.io.imsave(os.path.join(self.mix_msk_path, msk_file), img_obj, check_contrast=False)

        if self.no_inpaint_path is not None:
            for lb_file in os.listdir(os.path.join(self.no_inpaint_path, "Labels")):
                mat_obj = sio.loadmat(os.path.join(self.no_inpaint_path, "Labels", lb_file))
                sio.savemat(os.path.join(self.mix_lb_path, lb_file), mat_obj)

            for img_file in os.listdir(os.path.join(self.no_inpaint_path, "Images")):
                img_obj = skimage.io.imread(os.path.join(self.no_inpaint_path, "Images", img_file))
                skimage.io.imsave(os.path.join(self.mix_img_path, img_file), img_obj)


if __name__ == "__main__":
    mp = MergePatches()
    full_img_name_list = mp.get_full_img_name_list()
    mp.merge_patch2full_img(full_img_name_list)
    mp.merge_mix()
