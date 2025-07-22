
## NucleiMix (Beta)
## Realistic Data Augmentation for Nuclei Instance Segmentation

[Journal Link](https://arxiv.org/pdf/2410.16671?)

### Prepare Dataset and Environment
 Download datasets [ConSeP](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/), [GLySAC](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/) and [MoNuSAC](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/)
under the `data/` directory to have the following structure:
```
data
    ├── consep
    │   ├── Images
    │   ├── Labels
    │   ...
    ├── glysac
    │   ├── Images
        ├── Labels
```
Download [TransPath](https://github.com/Xiyue-Wang/TransPath) repository under the `gmm_dist_cal/` directory and install the requirements:
```
conda create -f environment.yml
```
### Augment Dataset

#### 1. Obtain the Distance Table 
Excute the following command to obtain the distance table between rare-type nuclei and major-type nuclei or backgound patches:
```
python gmm_dist_cal/dist_cal.py
```
Will have a following structure:
```
data
    ├── consep
    │   ├── Images
    │   ├── Labels
    │   ├── DistTable
    │   │   ├── paste_candidate.pkl
    
```

#### 2. Generate Augmented Dataset before Inpainting
For example, to augment 50 new rare-type nuclei for ConSeP dataset, change the config file `process_mix/alg/config.py`:

```python
self.prob_filter = 50
self.dataset = "consep"
```
Then run the following command:
```
cd process_mix
python src.py
```
Will have a following structure:
```
post_process
    ├── consep
    │   ├── 1_50
    │   ├── pre_inpaint
    │   ├── inpainted
    │   ├── crops_to_inpaint
    │   │   ├── diffusion_mix_gt
    │   │   ├── diffusion_mix_masks
    │   │   ├── mask_2
    │   │   ├── mask_3
    │   │   ├── mask_middle
    
```
The `pre_inpaint/` directory contains the augmented images and masks in the original size without inpainting, to 
use a pretrained diffusion model (256x256), we generate crops in the `crops_to_inpaint` directory.

#### 3. Inpaint
We used [MCG](https://github.com/hyungjin-chung/MCG_diffusion) inpainting method. Please download the [pretrained weight](https://drive.google.com/drive/folders/1DPRqsIRjJupCdb3_4H2u7jz4J1565TDz?usp=sharing).

#### 4. Merge the inpainted crops back to the original images
We need to merge the inpainted crops with the crops in `inpainted` directory, run the following command:
```
cd process_mix
python merge_after_inpaint.py
```


