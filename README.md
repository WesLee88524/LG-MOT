# LG-MOT
### **Multi-Granularity Language-Guided Multi-Object Tracking**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Yuhao Li](), [Muzammal Naseer](https://muzammal-naseer.com/), [Jiale Cao](https://jialecao001.github.io/), [Yu Zhu](), [Jinqiu Sun](https://teacher.nwpu.edu.cn/m/2009010133), [Yanning Zhang](https://jszy.nwpu.edu.cn/1999000059) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)


#### **Northwestern Polytechnical University， Mohamed bin Zayed University of AI, TianJin University, Linköping University**

<!-- [![Website](https://img.shields.io/badge/Project-Website-87CEEB)](site_url) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.04844.pdf)

## Latest 
- `2024/06/11`: We released our technical report on [arxiv](https://arxiv.org/abs/2406.04844.pdf). Our code and models are coming soon!

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Most existing multi-object tracking methods typically learn visual tracking features via maximizing dis-similarities of different instances and minimizing similarities of the same instance. While such a feature learning scheme achieves promising performance, learning discriminative features solely based on visual information is challenging especially in case of environmental interference such as occlusion, blur and domain variance. In this work, we argue that multi-modal language-driven features provide complementary information to classical visual features, thereby aiding in improving the robustness to such environmental interference. To this end, we propose a new multi-object tracking framework, named LG-MOT, that explicitly leverages language information at different levels of granularity (scene-and instance-level) and combines it with standard visual features to obtain discriminative representations. To develop LG-MOT, we annotate existing MOT datasets with scene-and instance-level language descriptions. We then encode both instance-and scene-level language information into high-dimensional embeddings, which are utilized to guide the visual features during training. At inference, our LG-MOT uses the standard visual features without relying on annotated language descriptions. Extensive experiments on three benchmarks, MOT17, DanceTrack and SportsMOT, reveal the merits of the proposed contributions leading to state-of-the-art performance. On the DanceTrack test set, our LG-MOT achieves an absolute gain of 2.2% in terms of target object association (IDF1 score), compared to the baseline using only visual features. Further, our LG-MOT exhibits strong cross-domain generalizability.
</details>

## Intro

- **MAVOS** is a transformer-based VOS method that achieves real-time FPS and reduced GPU memory for long videos. We introduce a Modulated Cross-Attention (MCA) memory, designed to efficiently propagate information from past frames to the target. Our approach utilizes a novel fusion operator capable of effectively managing both local and global features across diverse levels of detail.


<img src="source/MAVOS_overview.png" width="80%"/>

- **MAVOS**  increases the speed by 7.6x over the baseline DeAOT, while significantly reducing the GPU memory by 87% on long videos with comparable segmentation performance on short and long video datasets. 

<img src="source/Intro_figure.png" width="70%"/>

## Examples

<img src="source/basket_players.gif" width="80%"/>

<img src="source/dancing_girls.gif" width="80%"/>

<br>


## Results on Long videos benchmarks
### LVOS val set
<img src="source/LVOS.png" width="90%"/>

### LTV 
<img src="source/LTV.png" width="90%"/>


## Acknowledgment
The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at Alvis partially funded by the Swedish Research Council through grant agreement no. 2022-06725, the LUMI supercomputer hosted by CSC (Finland) and the LUMI consortium, and by the Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre.

## Citation
if you use our work, please consider citing us:
```BibTeX
@article{Shaker2024MAVOS,
  title={Efficient Video Object Segmentation via Modulated Cross-Attention Memory},
  author={Shaker, Abdelrahman and Wasim, Syed and Danelljan, Martin and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
  journal={arXiv:2403.17937},
  year={2024}
}

```


## License
This project is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for additional details.
# LG-MOT
.

# Latest
- **2024/06/11** We released our technical report on [arxiv](https://arxiv.org/abs/2406.04844 "arxiv"). Our code and models are coming soon.

## Abstract
Most existing multi-object tracking methods typically learn visual tracking features via maximizing dis-similarities of different instances and minimizing similarities of the same instance. While such a feature learning scheme achieves promising performance, learning discriminative features solely based on visual information is challenging especially in case of environmental interference such as occlusion, blur and domain variance. In this work, we argue that multi-modal language-driven features provide complementary information to classical visual features, thereby aiding in improving the robustness to such environmental interference. To this end, we propose a new multi-object tracking framework, named LG-MOT, that explicitly leverages language information at different levels of granularity (scene-and instance-level) and combines it with standard visual features to obtain discriminative representations. To develop LG-MOT, we annotate existing MOT datasets with scene-and instance-level language descriptions. We then encode both instance-and scene-level language information into high-dimensional embeddings, which are utilized to guide the visual features during training. At inference, our LG-MOT uses the standard visual features without relying on annotated language descriptions. Extensive experiments on three benchmarks, MOT17, DanceTrack and SportsMOT, reveal the merits of the proposed contributions leading to state-of-the-art performance. On the DanceTrack test set, our LG-MOT achieves an absolute gain of 2.2% in terms of target object association (IDF1 score), compared to the baseline using only visual features. Further, our LG-MOT exhibits strong cross-domain generalizability.
