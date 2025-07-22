import numpy as np
import os
import skimage
import scipy.io as sio
import random


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    return


def check_dir_save(path, name, obj):
    if not os.path.isdir(path):
        os.makedirs(path)
    suffix = name.split('.')[1]
    save_path = os.path.join(path, name)
    if suffix == "npz":
        np.savez(save_path, obj)
    elif suffix == "png":
        skimage.io.imsave(save_path, obj, check_contrast=False)
        print(f"save {save_path}")
    elif suffix == "mat":
        sio.savemat(save_path, obj)
    else:
        assert "Doesn't know suffix"


def remap_label(pred, label_list=None, pred_centroid=None):
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

    new_label = []
    new_centroid = []
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
        new_label.append(label_list[inst_id - 1])
        new_centroid.append(pred_centroid[inst_id - 1])

    new_label = np.array(new_label).astype("int32")
    new_centroid = np.array(new_centroid)
    return new_pred, new_label, new_centroid

