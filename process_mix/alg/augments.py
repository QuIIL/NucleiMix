import sys
sys.path.append('../')
from process_mix.utils.geo_utils import *
from process_mix.utils.other_utils import *
from process_mix.alg.config import Config
import numpy as np
import scipy.io as sio
import os
from scipy import ndimage
from collections import Counter
from PIL import Image
import pickle
np.random.seed(22)


class ImageProcessor(Config):
    def calculate_gaussian_probabilities(self, close_X_list, distances):
        """
        Calculate Gaussian Kernel probabilities for a given array of distances.
        The standard deviation used in the Gaussian Kernel is the standard deviation of the distances.

        :param distances: Numpy array of distances.
        :return: Numpy array of Gaussian probabilities.
        """
        # Calculate the standard deviation of the distances
        key_list = [f'{x["basename"]}_{x["inst_id"]}' for x in close_X_list]
        minor_fre = [self.minor_fre[k] for k in key_list]
        distances = np.array(distances)
        distances = distances * (np.array(minor_fre) + 1)
        sigma = np.std(distances)
        # Calculate the Gaussian probabilities (not normalized)
        # probabilities = np.exp(-distances ** 2 / (2 * sigma ** 2))
        probabilities = np.exp(-distances)
        # probabilities = 1/(np.array(minor_fre) + 1)
        # Normalize the probabilities so they sum to 1
        probabilities_normalized = probabilities / np.sum(probabilities)
        return probabilities_normalized


    def sample_and_remove(self, instances, instance_scores):
        """
        Sample one instance from 'instances' according to the probabilities 'instance_probs',
        then remove the sampled instance and update 'instance_probs'.

        :param instances: Numpy array of instances.
        :param instance_probs: Numpy array of corresponding probabilities.
        :return: Tuple of the sampled instance, updated instances array, and updated probabilities array.
        """
        # Ensure probabilities sum to 1
        # instance_probs = np.exp(instance_scores / np.std(instance_scores)) / np.sum(
        #     np.exp(instance_scores / np.std(instance_scores)))
        instance_probs = np.exp(instance_scores) / np.sum(np.exp(instance_scores))
        # Sample one instance
        sampled_index = np.random.choice(len(instances), p=instance_probs)
        sampled_instance = instances[sampled_index]
        # Remove the sampled instance and update probabilities
        return sampled_instance, instances, instance_scores, sampled_index

    def sample_minor_by_distance(self, instance, list_exp=None, list_gau=None):
        close_X_list, close_X_dist = instance["close_minor_list"], instance["close_minor_dist"]
        pick_id = np.random.choice(len(close_X_dist), p=self.calculate_gaussian_probabilities(close_X_list, close_X_dist))
        # pick_id = np.random.choice(len(close_X_dist))
        minor_key = f"{close_X_list[pick_id]['basename']}_{close_X_list[pick_id]['inst_id']}"
        # vis_dis(close_X_dist, list_exp, list_gau, key, pick_id)
        picked_minor = close_X_list[pick_id]
        basename2, inst_id2, bbox2 = picked_minor["basename"], picked_minor["inst_id"], picked_minor["inner_bbox"]
        return basename2, inst_id2, bbox2, minor_key, close_X_dist[pick_id]

    def __init__(self):
        super().__init__()
        self.info_dict = {
            "init_0": 0,
            "init_1": 0,
            "init_2": 0,
            "init_3": 0,
            "init_4": 0,
            "Minor increase totally": 0,
            "paste on background": 0,
            "replace from class 1": 0,
            "replace from class 2": 0,
            "replace from class 3": 0,
            "replace from class 4": 0,
            "pass": 0
        }
        self.masked_img_list = {}
        self.masked_label_list = {}
        self.masks_list = {}
        self.mask2_list = {}
        self.mask3_list = {}
        self.maskm_list = {}
        self.minor_size_pdf = None
        self.minor_size = None
        self.minor_fre = None
        self.instances = None
        self.storage = {}
        self.load_images(self.image_path, self.label_path)
        self.get_minor_candidate()
        check_manual_seed(self.seed)
        return

    def load_images(self, image_path, label_path):
        object_names = ['img', 'masked', 'final_mask_1', 'final_mask_nuclei2', 'final_mask_2srd', 'final_mask_3srd',
                        'inst_map', 'type_map', 'inst_centroid', 'inst_type', 'ann',
                        'inst_map_out', 'type_map_out', 'inst_type_out', 'inst_centroid_out', 'visited_list']
        for file_name in os.listdir(image_path):
            basename, ext = file_name.split('.')[0], file_name.split('.')[1]
            self.storage[basename] = {}
            # Img Read
            with Image.open(os.path.join(image_path, file_name)) as f:
                img = np.array(f)[:,:,:3]
            masked = np.array(img)
            img_h, img_w = img.shape[0], img.shape[1]
            final_mask_1 = np.ones((img_h, img_w)).astype(np.uint8)  # Remove nuclei 1
            final_mask_nuclei2 = np.ones((img_h, img_w)).astype(np.uint8)  # Region without nuclei_2
            final_mask_2srd = np.ones((img_h, img_w)).astype(np.uint8)
            final_mask_3srd = np.ones((img_h, img_w)).astype(np.uint8)
            visited_list = []
            # Label Read : Label contains 'inst_map', 'type_map', 'inst_type', 'inst_centroid'
            ann = sio.loadmat(os.path.join(label_path, basename + '.mat'))
            inst_map, type_map, inst_centroid, inst_type = ann['inst_map'], ann['type_map'], ann[
                'inst_centroid'], np.squeeze(ann['inst_type'])
            inst_map_out, type_map_out, inst_type_out, inst_centroid_out = np.array(inst_map), np.array(type_map), np.array(
                inst_type), np.array(inst_centroid)
            for name in object_names:
                self.storage[basename][name] = locals()[name]

    def get_minor_candidate(self):
        with open(self.paste_candidate_path, 'rb') as f:
            self.instances = pickle.load(f)
        minor_list = np.squeeze(self.instances["void_instances"])[0]["close_minor_list"]
        minor_id_list = [f'{x["basename"]}_{x["inst_id"]}' for x in minor_list]
        minor_size_list = [x["size"] for x in minor_list]
        size_count = Counter(minor_size_list)
        total = sum(size_count.values())
        size_pdf = [count / total for size, count in size_count.items()]
        self.minor_size_pdf = size_pdf
        self.minor_size = minor_size_list
        self.minor_fre = {key: 0 for key in minor_id_list}

    def img_update(self, instances, instance_scores):
        count = 0
        dis_avg = 0

        while count < self.prob_filter:
            print(count)
            instance, instances, instance_scores, sampled_index = self.sample_and_remove(instances, instance_scores)
            # major or background
            basename1, inst_id1, bbox1 = instance["basename"], instance["inst_id"], instance["inner_bbox"]
            # minor
            basename2, inst_id2, bbox2, minor_key, cur_dist = self.sample_minor_by_distance(instance)
            key = f"{basename1}_{basename2}_{int(inst_id1)}_{int(inst_id2)}"
            img2 = self.storage[basename2]["img"]
            ann2 = self.storage[basename2]["ann"]
            inst_map2 = self.storage[basename2]['inst_map']
            msk2 = (inst_map2 == inst_id2).astype(np.uint8)
            msk2_r2 = binary_dilation(msk2, iterations=2).astype(np.uint8)
            cent2 = get_geo_info(msk2_r2)[0]
            cent2 = (int(cent2[0]), int(cent2[1]))

            img = self.storage[basename1]["img"]
            masked = self.storage[basename1]["masked"]
            final_mask_nuclei2 = self.storage[basename1]["final_mask_nuclei2"]
            final_mask_1 = self.storage[basename1]["final_mask_1"]
            final_mask_2srd = self.storage[basename1]["final_mask_2srd"]
            final_mask_3srd = self.storage[basename1]["final_mask_3srd"]
            inst_map = self.storage[basename1]["inst_map"]
            inst_type = self.storage[basename1]["inst_type"]
            inst_map_out = self.storage[basename1]["inst_map_out"]
            type_map_out = self.storage[basename1]["type_map_out"]
            inst_centroid_out = self.storage[basename1]["inst_centroid_out"]
            inst_type_out = self.storage[basename1]["inst_type_out"]
            visited_list = self.storage[basename1]['visited_list']

            # process 1
            try:
                if inst_id1 == 0:  # paste minor on bg
                    msk1 = np.zeros_like(inst_map_out).astype(np.uint8)
                    msk1[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]] = 1

                    empty_ratio = len(np.where(inst_map_out[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]] == 0)[0]) / ((bbox1[3]-bbox1[1]) * (bbox1[2]-bbox1[0]))
                    if empty_ratio < 0.8:
                        print("pass due to no space")
                        instances = np.delete(instances, sampled_index)
                        instance_scores = np.delete(instance_scores, sampled_index)
                        continue
                    msk1[inst_map_out > 0] = 0
                    connectivity = 8
                    output = cv2.connectedComponentsWithStats(msk1, connectivity, cv2.CV_32S)[1]
                    largest_area_label_list = np.unique(output)
                    largest_area_list = []
                    for lal in largest_area_label_list:
                        largest_area_list.append((len(output[output == lal]), lal))
                    largest_area_list.sort(reverse=True)
                    if len(largest_area_list) <= 1:
                        print("pass due to no space")
                        instances = np.delete(instances, sampled_index)
                        instance_scores = np.delete(instance_scores, sampled_index)
                        continue
                    msk1[output != largest_area_list[1][1]] = 0
                    msk1_r2 = ndimage.binary_dilation(msk1, iterations=2).astype(np.uint8)
                    center_of_mass = ndimage.center_of_mass(msk1_r2)
                    target_center_mass = np.array((center_of_mass[1], center_of_mass[0]))
                    cent1 = target_center_mass

                    while (np.sum(msk1_r2) < np.sum(msk2_r2)):
                        msk1 = ndimage.binary_dilation(msk1, iterations=2).astype(np.uint8)
                        msk1[inst_map_out > 0] = 0
                        output = cv2.connectedComponentsWithStats(msk1, connectivity, cv2.CV_32S)[1]
                        largest_area_label_list = np.unique(output)
                        largest_area_list = []
                        for lal in largest_area_label_list:
                            largest_area_list.append((len(output[output == lal]), lal))
                        largest_area_list.sort(reverse=True)
                        msk1[output != largest_area_list[1][1]] = 0
                        msk1_r2 = ndimage.binary_dilation(msk1, iterations=2).astype(np.uint8)
                        center_of_mass = ndimage.center_of_mass(msk1_r2)
                        target_center_mass = np.array((center_of_mass[1], center_of_mass[0]))
                        cent1 = target_center_mass
                else: # replace major with minor
                    if inst_id1 in visited_list:
                        print("pass due to visited")
                        continue
                    msk1 = (inst_map_out == inst_id1).astype(np.uint8)
                    msk1_r2 = binary_dilation(msk1, iterations=2).astype(np.uint8)
                    cent1 = get_geo_info(msk1_r2)[0]

                cent1 = (int(cent1[0]), int(cent1[1]))
                if cent1[0] < self.margin or cent1[1] < self.margin or \
                        cent1[0] > img.shape[1] - self.margin or cent1[1] > img.shape[0] - self.margin :
                    print("pass due to margin1")
                    instances = np.delete(instances, sampled_index)
                    instance_scores = np.delete(instance_scores, sampled_index)
                    continue
                if cent2[0] < self.margin2 or cent2[1] < self.margin2 or \
                        cent2[0] > img2.shape[1] - self.margin2 or cent2[1] > img2.shape[0] - self.margin2 :
                    print("pass due to margin2")
                    self.minor_fre[minor_key] += 1
                    continue

                # try:
                visited_list, inst_map_out, type_map_out, inst_centroid_out, inst_type_out, masked = paste_idx2_on_img1(
                    inst_id1,
                    img,
                    msk1,
                    cent1,
                    bbox1,
                    inst_id2,
                    img2,
                    ann2,
                    masked,
                    final_mask_nuclei2,
                    final_mask_1,
                    final_mask_2srd,
                    final_mask_3srd,
                    visited_list,
                    inst_map,
                    inst_map_out,
                    type_map_out,
                    inst_centroid_out,
                    inst_type_out,
                    self.minor_size,
                    inst_id1 != 0,
                )
                dis_avg += cur_dist

                # punish frequent minor
                self.minor_fre[minor_key] += 1
                instances = np.delete(instances, sampled_index)
                instance_scores = np.delete(instance_scores, sampled_index)
                count += 1
                update_names = ['masked', 'final_mask_1', 'final_mask_nuclei2', 'final_mask_2srd', 'final_mask_3srd',
                                'inst_map_out', 'type_map_out', 'inst_type_out', 'inst_centroid_out', 'visited_list']
                for name in update_names:
                    self.storage[basename1][name] = locals()[name]
            except Exception as e:
                print(f"pass due to error: {e}")

        print(dis_avg / self.prob_filter)

    def run(self):

        major_instances = np.squeeze(self.instances["major_instances"])
        major_scores = np.squeeze(self.instances["major_scores"])

        void_instances = np.squeeze(self.instances["void_instances"])
        void_scores = np.squeeze(self.instances["void_scores"])
        instances = np.concatenate([major_instances, void_instances], axis=0)
        scores = np.concatenate([major_scores, void_scores], axis=0)

        self.img_update(instances, scores)
        print("finished nuclei processing")

        for basename in self.storage.keys():
            print(f"saving {basename}")
            inst_map_out, inst_type_out, inst_centroid_out, type_map_out = self.storage[basename]['inst_map_out'], \
                                                                           self.storage[basename]['inst_type_out'], \
                                                                           self.storage[basename]['inst_centroid_out'], \
                                                                           self.storage[basename]['type_map_out']
            inst_map_out, inst_type_out, inst_centroid_out = remap_label(inst_map_out, inst_type_out, inst_centroid_out)
            masked = self.storage[basename]['masked']
            final_mask_1, final_mask_nuclei2, final_mask_2srd, final_mask_3srd = self.storage[basename]['final_mask_1'], \
                                                                                 self.storage[basename]['final_mask_nuclei2'], \
                                                                                 self.storage[basename]['final_mask_2srd'], \
                                                                                 self.storage[basename]['final_mask_3srd']

            assert inst_centroid_out.shape[0] == inst_type_out.shape[0]
            assert inst_centroid_out.shape[0] == np.unique(inst_map_out).shape[0] - 1
            self.masked_img_list[basename] = masked
            self.masked_label_list[basename] = {
                'inst_map': inst_map_out,
                'type_map': type_map_out,
                'inst_type': np.array(inst_type_out[:, None]),
                'inst_centroid': inst_centroid_out
            }
            self.masks_list[basename] = final_mask_1 * 255
            self.maskm_list[basename] = final_mask_nuclei2 * 255
            self.mask2_list[basename] = final_mask_2srd * 255
            self.mask3_list[basename] = final_mask_3srd * 255

            check_dir_save(
                self.gt_path,
                basename + '_masked.png',
                self.masked_img_list[basename]
            )
            check_dir_save(
                self.masked_label_path,
                basename + '.mat',
                self.masked_label_list[basename]
            )
            check_dir_save(
                self.mask_path,
                basename + '_mask_1.png',
                self.masks_list[basename]
            )
            check_dir_save(
                self.masks_m_path,
                basename + '_mask_nuclei2.png',
                self.maskm_list[basename]
            )
            check_dir_save(
                self.masks_2_path,
                basename + '_mask_2srd.png',
                self.mask2_list[basename]
            )
            check_dir_save(
                self.masks_3_path,
                basename + '_mask_3srd.png',
                self.mask3_list[basename]
            )


if __name__ == "__main__":

    mix_alg = ImageProcessor()
    mix_alg.run()
    print(mix_alg)


