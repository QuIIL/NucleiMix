import os


class Config(object):

    def __init__(self):
        self.prob_filter = 50
        version = 1
        self.dir_seq = f'{version}_{self.prob_filter}' # 0.5 resize + margin2
        # self.dataset = "monusac"
        # self.dataset = "glysac"
        self.dataset = "consep"
        self.seed = 22
        self.margin = 30
        self.margin2 = 30
        self.image_path = f'../data/{self.dataset}/Images/'
        self.label_path = f'../data/{self.dataset}/Labels/'
        self.paste_candidate_path = f"../data/{self.dataset}/DistTable/paste_candidate.pkl"

        # from here is for post process
        self.noinpaint_img = os.path.join("..", "post_process", self.dataset, self.dir_seq, "no_inpaint", "Images")
        self.noinpaint_lb = os.path.join("..", "post_process", self.dataset, self.dir_seq, "no_inpaint", "Labels")

        self.gt_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "masked_images")
        self.masked_label_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "masked_labels")
        self.mask_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "masks")
        self.masks_m_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "mask_middle_full")
        self.masks_2_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "mask_2_full")
        self.masks_3_path = os.path.join("..", "post_process", self.dataset, self.dir_seq, "pre_inpaint", "mask_3_full")

        self.patch_gt_path = f"../post_process/{self.dataset}/{self.dir_seq}/diffusion_mix_gt"
        self.patch_gt_path_noinpaint = f"../post_process/{self.dataset}/{self.dir_seq}/test_diffusion_mix_nuclei/inpainted"
        self.patch_mask_path = f"../post_process/{self.dataset}/{self.dir_seq}/diffusion_mix_masks"
        self.patch_masks_m_path = f"../post_process/{self.dataset}/{self.dir_seq}/mask_middle"
        self.patch_masks_2_path = f"../post_process/{self.dataset}/{self.dir_seq}/mask_2"
        self.patch_masks_3_path = f"../post_process/{self.dataset}/{self.dir_seq}/mask_3"

        self.no_inpaint_path = None
        self.inpainted_patch_path = f"../post_process/{self.dataset}/{self.dir_seq}/test_diffusion_mix_nuclei/inpainted"
        self.train_mix_path = f"../post_process/{self.dataset}/{self.dir_seq}/Train_mix/"
        self.mix_img_path = f"../post_process/{self.dataset}/{self.dir_seq}/Train_mix/Images"
        self.mix_lb_path = f"../post_process/{self.dataset}/{self.dir_seq}/Train_mix/Labels"
        self.mix_msk_path = f"../post_process/{self.dataset}/{self.dir_seq}/Train_mix/Masks"
