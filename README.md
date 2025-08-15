# LG-MOT
### **Multi-Granularity Language-Guided Training for Multi-Object Tracking**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Yuhao Li](), [Jiale Cao](https://jialecao001.github.io/), [Muzammal Naseer](https://muzammal-naseer.com/), [Yu Zhu](), [Jinqiu Sun](https://teacher.nwpu.edu.cn/m/2009010133), [Yanning Zhang](https://jszy.nwpu.edu.cn/1999000059) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)


#### **Northwestern Polytechnical University， Mohamed bin Zayed University of AI,  TianJin University, Khalifa University, Linköping University**

<!-- [![Website](https://img.shields.io/badge/Project-Website-87CEEB)](site_url) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.04844.pdf)

## Latest 
- `2025/08/15`: We released our annotated [DanceTrack-L](https://huggingface.co/datasets/UHow/DanceTrack-L), [MOT17-L](https://huggingface.co/datasets/UHow/MOT17-L), and [SportsMOT](https://huggingface.co/datasets/UHow/SportsMOT-L).
- `2025/05/22`: Our work has been accepted by IEEE TCSVT.
- `2024/06/14`: We released our trained models.
- `2024/06/12`: We released our code.
- `2024/06/11`: We released our technical report on [arxiv](https://arxiv.org/abs/2406.04844.pdf). Our code and models are coming soon!

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Most existing multi-object tracking methods typically learn visual tracking features via maximizing dis-similarities of different instances and minimizing similarities of the same instance. While such a feature learning scheme achieves promising performance, learning discriminative features solely based on visual information is challenging especially in case of environmental interference such as occlusion, blur and domain variance. In this work, we argue that multi-modal language-driven features provide complementary information to classical visual features, thereby aiding in improving the robustness to such environmental interference. To this end, we propose a new multi-object tracking framework, named LG-MOT, that explicitly leverages language information at different levels of granularity (scene-and instance-level) and combines it with standard visual features to obtain discriminative representations. To develop LG-MOT, we annotate existing MOT datasets with scene-and instance-level language descriptions. We then encode both instance-and scene-level language information into high-dimensional embeddings, which are utilized to guide the visual features during training. At inference, our LG-MOT uses the standard visual features without relying on annotated language descriptions. Extensive experiments on three benchmarks, MOT17, DanceTrack and SportsMOT, reveal the merits of the proposed contributions leading to state-of-the-art performance. On the DanceTrack test set, our LG-MOT achieves an absolute gain of 2.2% in terms of target object association (IDF1 score), compared to the baseline using only visual features. Further, our LG-MOT exhibits strong cross-domain generalizability. 
</details>

## Intro

- **LG-MOT**  first annotate
the training sets and validation sets of commonly used MOT datasets including
[MOT17](https://motchallenge.net/data/MOT17/), [DanceTrack](https://dancetrack.github.io/) and [SportsMOT](https://github.com/MCG-NJU/SportsMOT?tab=readme-ov-file) with language descriptions
at both scene and instance levels.



| Dataset|Videos (Scenes) |Annotated Scenes |Tracks (Instances) | Annotated Instances|Annotated Boxes |Frames|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MOT17-L |  7 | 7 | 796 |796 | 614,103 | 110,407 |
| DanceTrack-L| 65   | 65| 682 | 682 | 576,078 | 67,304|
|SportsMOT-L |90| 90 | 1,280 | 1,280 |608,152|55,544 |
|Total | 162|162|2,758|2,758|1798,333|233,255|

- **LG-MOT** is a new multi-object tracking framework which leverages language information at different granularity during training to enhance object association capabilities. During training, our **ISG** module aligns each node embedding $\phi(b_i^k)$ with instance-level descriptions embeddings $\varphi_i$, while our **SPG** module aligns edge embeddings $\hat{E}_{(u,v)}$ with scene-level descriptions embeddings $\varphi_s$ to guide correlation estimation after message passing. Our approach does not require language description during inference.

<div align=center>
	<img src="source/LG-MOT_overview.png" width="90%"/>
</div>

- **LG-MOT**  increases 1.2% in terms of IDF1 over the baseline SUSHI intra-domain, while significantly improves 11.2% in cross-domain.

<div align=center>
	<img src="source/performance.png" width="50%"/>
</div>
<!-- <img src="source/performance.png" width="50%"/> -->

## Visualization
<div align=center>
<img src="source/visualization.png" width="90%"/>
</div>

## Performance on Benchmarks
### MOT17 Challenge - Test Set
| Dataset    |  IDF1 | HOTA | MOTA | ID Sw. | model|
|:---:|:---:|:---:|:---:|:---:|:---:|
|MOT17 |81.7 |65.6 |81.0 |1161 |[checkpoint.pth](https://drive.google.com/file/d/1LVFQ-i9cuilGX82dIlArXmm1TuvcxNvx/view?usp=drive_link)|



### DanceTrack - Test Set
| Dataset    |  IDF1 | HOTA | MOTA | AssA | DetA | model|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|DanceTrack | 60.5 | 61.8| 89.0| 80.0| 47.8 |[checkpoint.pth](https://drive.google.com/file/d/1yD-k-rXsqhu8c9mxjCYG41OuHYeDSpWa/view?usp=drive_link)|

### SportsMOT - Test Set
| Dataset    |  IDF1 | HOTA | MOTA | ID Sw. | model|
|:---:|:---:|:---:|:---:|:---:|:---:|
|SportsMOT | 77.1 |75.0 |91.0 |2847|[checkpoint.pth](https://drive.google.com/file/d/1e-xfa2yeASdWLjWKGSzrn46_hkl6L8vO/view?usp=drive_link)|


## Setup

1. Clone and enter this repository
    ```
    git clone https://github.com/WesLee88524/LG-MOT.git
    cd LG-MOT
    ```

3. Create an [Anaconda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for this project:
    ```
    conda env create -f environment.yml
    conda activate LG-MOT
    ```

3. Clone [fast-reid](https://github.com/JDAI-CV/fast-reid) (latest version should be compatible but we use [this](https://github.com/JDAI-CV/fast-reid/tree/afe432b8c0ecd309db7921b7292b2c69813d0991) version) and install its dependencies. The `fast-reid` repo should be inside `LG-MOT` root:
    ```
    LG-MOT
    ├── src
    ├── fast-reid
    └── ...
    ```

4. Download [re-identification model we use from fast-reid](https://drive.google.com/file/d/1MovixfOLwnnXet05JLIGPy-Ag67fdW-2/view?usp=share_link) and move it inside `LG-MOT/fastreid-models/model_weights/`

5. Download [MOT17](https://motchallenge.net/data/MOT17/), [SPORTSMOT](https://github.com/MCG-NJU/SportsMOT?tab=readme-ov-file) and [DanceTrack](https://dancetrack.github.io/) datasets. In addition, prepare seqmaps to run evaluation (for details see [TrackEval](https://github.com/JonathonLuiten/TrackEval)). We provide an [example seqmap](https://drive.google.com/drive/folders/1LYRYPuNWIWWz-HXSjuqyX8Fz9fDunTRI?usp=sharing). Overall, the expected folder structure is: 

    ```
    DATA_PATH
    ├── DANCETRACK
    │   └── ...
    ├── SPORTSMOT
    │   └── ...
    └── MOT17
        └── seqmaps
        │    ├── seqmap_file_name_matching_split_name.txt
        │    └── ...
        └── train
        │    ├── MOT17-02
        │    │   ├── det
        │    │   │   └── det.txt
        │    │   └── gt
        │    │   │   └── gt.txt
        │    │   └── img1 
        │    │   │   └── ...
        │    │   └── seqinfo.ini
        │    └── ...
        └── test
             └── ...

    ```



<!-- 6. (OPTIONAL) Download [our detections](https://drive.google.com/drive/folders/1bxw1Hz77LCCW3cWizhg_q03vk4aXUriz?usp=share_link) and [pretrained models](https://drive.google.com/drive/folders/1cU7LeTAeKxS-nvxqrUdUstNpR-Wpp5NV?usp=share_link) and arrange them according to the expected file structure (See step 5). -->


## Training
You can launch a training from the command line. An example command for MOT17 training:

```
RUN=example_mot17_training
REID_ARCH='fastreid_msmt_BOT_R50_ibn'

DATA_PATH=YOUR_DATA_PATH

python scripts/main.py --experiment_mode train --cuda --train_splits MOT17-train-all --val_splits MOT17-train-all --run_id ${RUN}_${REID_ARCH} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH}  --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --inst_KL_loss 1 --KL_loss 1 --num_epoch 200 --mpn_use_prompt_edge 1 1 1 1 --use_instance_prompt 
```


## Testing
You can test a trained model from the command line. An example for testing on MOT17 training set:
```
RUN=example_mot17_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'

DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=your_pretrained_model_path

python scripts/main.py --experiment_mode test --cuda --test_splits MOT17-test-all --run_id ${RUN}_${REID_ARCH} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH}  --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --hicl_model_path ${PRETRAINED_MODEL_PATH}  --inst_KL_loss 1 --KL_loss 1 --mpn_use_prompt_edge 1 1 1 1 

```




## Citation
if you use our work, please consider citing us:
```BibTeX
@ARTICLE{11010839,
  author={Li, Yuhao and Cao, Jiale and Naseer, Muzammal and Zhu, Yu and Sun, Jinqiu and Zhang, Yanning and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Multi-Granularity Language-Guided Training for Multi-Object Tracking}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Visualization;Training;Feature extraction;Standards;Object detection;Accuracy;Representation learning;Current transformers;Circuits and systems;Target tracking;Multi-object tracking;Language-guided features;Cross-domain generalizability},
  doi={10.1109/TCSVT.2025.3572810}}

```


## License
This project is released under the Apache license. See [LICENSE](LICENSE) for additional details.
## Acknowledgement
The code is mainly based on [SUSHI](https://github.com/dvl-tum/SUSHI). Thanks for their wonderful works.
