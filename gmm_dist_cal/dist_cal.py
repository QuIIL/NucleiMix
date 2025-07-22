import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skimage
import random
import pickle
import cv2 as cv
import torch
import scipy.io as sio
import numpy as np
from sklearn.decomposition._pca import PCA
import albumentations as A
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns
from element import process_nuclei_patches, process_void_patches
from gmm_dist_cal.TransPath.ctran import ctranspath
from torchvision import transforms
import torch.nn as nn


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    return


def model_selection_plot(X):
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)
    param_grid = {
        "n_components": range(1, 7),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X)
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    df.sort_values(by="BIC score").head()
    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    plt.show()


if __name__ == "__main__":
    dataset = "consep"
    # dataset = "glysac"
    # dataset = "monusac"

    check_manual_seed(888)
    model = ctranspath()
    model.cuda()
    model.head = nn.Identity()
    td = torch.load(r'./TransPath/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformer_val = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    feature_vector_dim = 16
    crop_size = 224
    minor_class = [1] if dataset == "glysac" or dataset == "consep" else [4]
    if dataset == "glysac":
        major_class = [2,3]
    elif dataset == "consep":
        major_class = [2,3,4]
    else:
        major_class = [1,2]

    image_folder = f"../data/{dataset}/Images"
    label_folder = f"../data/{dataset}/Labels"
    yaml_folder = f"../data/{dataset}/DistTable"
    os.makedirs(yaml_folder, exist_ok=True)

    minor_feature_np, minor_datapoints, minor_labels = [], [], []
    major_feature_np, major_datapoints, major_labels = [], [], []
    void_datapoints, void_feature_np, void_labels = [], [], []
    # init datapoint
    for file_idx, file_name in enumerate(sorted(os.listdir(image_folder))):
        if '.DS' in file_name:
            continue
        print(file_name)
        basename = file_name.split('.')[0]
        input_image_path = os.path.join(image_folder, file_name)
        input_lb_path = os.path.join(label_folder, basename+'.mat')
        img_np = skimage.io.imread(input_image_path)[:,:,:3]
        load_size_h, load_size_w = img_np.shape[0], img_np.shape[1]
        img_padding = A.augmentations.geometric.transforms.PadIfNeeded(load_size_h + crop_size, load_size_w + crop_size, border_mode=cv.BORDER_REFLECT)
        msk_padding = A.augmentations.geometric.transforms.PadIfNeeded(load_size_h + crop_size, load_size_w + crop_size, border_mode=cv.BORDER_CONSTANT,
                                                                       mask_value=0)
        ann = sio.loadmat(input_lb_path)
        inst_map, inst_type, inst_cent = ann["inst_map"], ann["inst_type"], ann["inst_centroid"]
        if dataset == "consep":
            inst_type[(inst_type == 3) | (inst_type == 4)] = 3
            inst_type[(inst_type == 5) | (inst_type == 6) | (inst_type == 7)] = 4
        elif dataset == "glysac":
            inst_type[(inst_type == 2) | (inst_type == 9) | (inst_type == 10)] = 1
            inst_type[(inst_type == 4) | (inst_type == 5) | (inst_type == 7) | (inst_type == 6)] = 2
            inst_type[(inst_type == 3) | (inst_type == 8)] = 3
        inst_list = list(np.unique(inst_map).astype(int))
        inst_list.remove(0)

        # collect minor instances
        process_nuclei_patches(model, transformer_val, inst_list,
                        img_padding(image=img_np)['image'], msk_padding, ann,
                        minor_datapoints, minor_feature_np, minor_labels,
                        crop_size, basename, dataset, minor_class, minor_class)
        # collect major instances
        process_nuclei_patches(model, transformer_val, inst_list,
                        img_padding(image=img_np)['image'], msk_padding, ann,
                        major_datapoints, major_feature_np, major_labels,
                        crop_size, basename, dataset, major_class, minor_class)
        # collect void instances
        process_void_patches(model, transformer_val, inst_list,
                        img_padding(image=img_np)['image'], msk_padding, ann,
                        void_datapoints, void_feature_np, void_labels,
                        crop_size, basename, dataset, minor_class)

    minor_datapoints, minor_feature_np, minor_labels = np.array(minor_datapoints), np.array(minor_feature_np), np.array(minor_labels)
    major_datapoints, major_feature_np, major_labels = np.array(major_datapoints), np.array(major_feature_np), np.array(major_labels)
    void_datapoints, void_feature_np, void_labels = np.array(void_datapoints), np.array(void_feature_np), np.array(void_labels)

    # PCA dimension reduction
    pca = PCA(n_components=feature_vector_dim)
    pca.fit(minor_feature_np)
    plt.scatter(np.arange(feature_vector_dim), pca.singular_values_[:feature_vector_dim])
    plt.xlabel("SVD basis")
    plt.ylabel(f"{dataset} Singuler values")
    plt.show()

    minor_archive, major_archive, void_archive = pca.transform(minor_feature_np), \
                                                 pca.transform(major_feature_np),\
                                                 pca.transform(void_feature_np)
    X = minor_archive
    model_selection_plot(minor_archive) # choose the best model
    model = GaussianMixture(n_components=1, random_state=8, covariance_type="diag")
    model.fit(minor_archive)

    other_archive = np.concatenate([major_archive, void_archive], axis=0)
    other_scores = model.score_samples(other_archive)
    major_scores, void_scores = other_scores[: len(major_archive)], other_scores[len(major_archive):]

    for idx, pred_datapoint in enumerate(major_datapoints):
        pred_feature = major_archive[idx]
        # k_closest_x_idx = np.argpartition(np.linalg.norm(train_final[:] - pred_feature, axis=1), top_k)[:top_k]
        k_closest_x_dist = np.linalg.norm(minor_archive[:] - pred_feature, axis=1)
        k_closest_x_datapoint = minor_datapoints
        pred_datapoint["close_minor_list"], pred_datapoint["close_minor_dist"] = k_closest_x_datapoint.tolist(), k_closest_x_dist.tolist()

    for idx, pred_datapoint in enumerate(void_datapoints):
        pred_feature = void_archive[idx]
        k_closest_x_dist = np.linalg.norm(minor_archive[:] - pred_feature, axis=1)
        k_closest_x_datapoint = minor_datapoints
        pred_datapoint["close_minor_list"], pred_datapoint["close_minor_dist"] = k_closest_x_datapoint.tolist(), k_closest_x_dist.tolist()

    result = dict()
    result = {
        "major_instances": major_datapoints,
        "void_instances": void_datapoints,
        "major_scores": major_scores,
        "void_scores": void_scores
    }

    with open(f'{yaml_folder}/paste_candidate.pkl', 'wb') as f:
        pickle.dump(result, f)



