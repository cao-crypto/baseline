# Generated and Pseudo Content guided Prototype Refinement for Few-shot Point Cloud Segmentation [[pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/377d0752059d3d4686aa021b664a25dd-Paper-Conference.pdf)]


## Overview

![framework](framework.png)



## Running 

**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**

**Update torch version:** pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 https://download.pytorch.org/whl/torch_stable.html


### Pretrain

Pretrain the segmentor which includes feature extractor module on the available training set:

```bash
bash scripts/pretrain_segmentor.sh
```

### Train

Train our method under few-shot setting:

```bash
bash scripts/train_GPCPR.sh
```

### Evaluation

Test our method under few-shot setting:

```bash
bash scripts/eval_GPCPR.sh
```


## Checkpoints and Log Files


<details><summary>📋 Example Evaluation Log on S3DIS 2-way 1-shot $S^0$ (click to expand)</summary>
<pre style="white-space: pre-wrap;">
[Eval] Iter: 100 | Loss: 0.6116 | Acc: 0.93 | 2024-05-01 06:50:34.924039
[Eval] Iter: 200 | Loss: 0.7918 | Acc: 0.87 | 2024-05-01 06:50:43.659885
[Eval] Iter: 300 | Loss: 0.4648 | Acc: 0.91 | 2024-05-01 06:50:52.470970
[Eval] Iter: 400 | Loss: 0.4312 | Acc: 0.97 | 2024-05-01 06:51:01.385772
[Eval] Iter: 500 | Loss: 0.8366 | Acc: 0.83 | 2024-05-01 06:51:10.089211
[Eval] Iter: 600 | Loss: 0.3792 | Acc: 0.92 | 2024-05-01 06:51:18.904245
[Eval] Iter: 700 | Loss: 0.8277 | Acc: 0.84 | 2024-05-01 06:51:27.654968
[Eval] Iter: 800 | Loss: 0.5387 | Acc: 0.90 | 2024-05-01 06:51:36.446670
[Eval] Iter: 900 | Loss: 0.6179 | Acc: 0.97 | 2024-05-01 06:51:45.185131
[Eval] Iter: 1000 | Loss: 0.5811 | Acc: 0.87 | 2024-05-01 06:51:53.024703
[Eval] Iter: 1100 | Loss: 0.9381 | Acc: 0.79 | 2024-05-01 06:51:59.957046
[Eval] Iter: 1200 | Loss: 0.3135 | Acc: 0.91 | 2024-05-01 06:52:05.937542
[Eval] Iter: 1300 | Loss: 0.2423 | Acc: 0.93 | 2024-05-01 06:52:13.156202
[Eval] Iter: 1400 | Loss: 0.4104 | Acc: 0.97 | 2024-05-01 06:52:20.758485
[Eval] Iter: 1500 | Loss: 1.0394 | Acc: 0.83 | 2024-05-01 06:52:29.347236
*****Test Classes: [3, 11, 10, 0, 8, 4]*****
----- [class 0]  IoU: 0.773986 -----
----- [class 1]  IoU: 0.635013 -----
----- [class 2]  IoU: 0.770267 -----
----- [class 3]  IoU: 0.787329 -----
----- [class 4]  IoU: 0.610820 -----
----- [class 5]  IoU: 0.871807 -----
----- [class 6]  IoU: 0.767085 -----

=====[TEST] Loss: 0.6994 | Mean IoU: 0.740387 =====
</pre>
</details>


<details><summary>📋 Example Evaluation Log on S3DIS 2-way 1-shot $S^1$     (click to expand)</summary>
<pre style="white-space: pre-wrap;">
[Eval] Iter: 100 | Loss: 0.3545 | Acc: 0.98 | 2024-05-01 06:52:26.173780
[Eval] Iter: 200 | Loss: 0.3451 | Acc: 0.94 | 2024-05-01 06:52:33.936398
[Eval] Iter: 300 | Loss: 0.4708 | Acc: 0.97 | 2024-05-01 06:52:41.070406
[Eval] Iter: 400 | Loss: 0.6633 | Acc: 0.81 | 2024-05-01 06:52:49.392353
[Eval] Iter: 500 | Loss: 2.1276 | Acc: 0.57 | 2024-05-01 06:52:56.535450
[Eval] Iter: 600 | Loss: 0.6927 | Acc: 0.89 | 2024-05-01 06:53:03.686981
[Eval] Iter: 700 | Loss: 0.7142 | Acc: 0.83 | 2024-05-01 06:53:10.838412
[Eval] Iter: 800 | Loss: 1.0077 | Acc: 0.68 | 2024-05-01 06:53:17.949161
[Eval] Iter: 900 | Loss: 0.5528 | Acc: 0.91 | 2024-05-01 06:53:25.107347
[Eval] Iter: 1000 | Loss: 0.3311 | Acc: 0.97 | 2024-05-01 06:53:32.250927
[Eval] Iter: 1100 | Loss: 0.6496 | Acc: 0.82 | 2024-05-01 06:53:39.485327
[Eval] Iter: 1200 | Loss: 0.5423 | Acc: 0.98 | 2024-05-01 06:53:46.672769
[Eval] Iter: 1300 | Loss: 0.7527 | Acc: 0.79 | 2024-05-01 06:53:53.884703
[Eval] Iter: 1400 | Loss: 0.5532 | Acc: 0.86 | 2024-05-01 06:54:01.128012
[Eval] Iter: 1500 | Loss: 0.4483 | Acc: 0.97 | 2024-05-01 06:54:08.372825
*****Test Classes: [6, 1, 9, 7, 2, 5]*****
----- [class 0]  IoU: 0.799077 -----
----- [class 1]  IoU: 0.839004 -----
----- [class 2]  IoU: 0.690794 -----
----- [class 3]  IoU: 0.893977 -----
----- [class 4]  IoU: 0.723927 -----
----- [class 5]  IoU: 0.673990 -----
----- [class 6]  IoU: 0.824688 -----

=====[TEST] Loss: 0.6656 | Mean IoU: 0.774397 =====
</pre>
</details>


More checkpoints and log files will be released soon. Please stay tuned!

## Citation
Please cite our paper if it is helpful to your research:

    @inproceedings{NEURIPS2024_377d0752,
     author = {Wei, Lili and Lang, Congyan and Chen, Ziyi and Wang, Tao and Li, Yidong and Liu, Jun},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
     pages = {31103--31123},
     publisher = {Curran Associates, Inc.},
     title = {Generated and Pseudo Content guided Prototype Refinement for Few-shot Point Cloud Segmentation},
     url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/377d0752059d3d4686aa021b664a25dd-Paper-Conference.pdf},
     volume = {37},
     year = {2024}
    }





## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch), [attMPTI](https://github.com/Na-Z/attMPTI), [QGPA](https://github.com/heshuting555/PAP-FZS3D) for sharing their source code.
