from alg.augments import ImageProcessor
from utils.crop_patch import *


if __name__ == "__main__":
    mix_alg = ImageProcessor()
    mix_alg.run()

    cp = CropPatches()
    save_list = cp.crop_diffusion_mix(mix_alg.masked_img_list, mix_alg.masks_list)
    cp.crop_diffusion_2(save_list, mix_alg.maskm_list, mix_alg.mask2_list, mix_alg.mask3_list)
    cp.count_class_num(mix_alg.masked_label_list)