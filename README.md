# Learning Affordance Grounding from Exocentric Images
PyTorch implementation of our one-shot affordance detection models. This repository contains PyTorch evaluation code, training code.
1. [📎 Paper Link](#1)
2. [💡 Abstract](#2)
3. [📖 Method](#3)
4. [📂 Dataset](#4)
5. [📃 Requirements](#5)
6. [✏️ Usage](#6)
7. [📊 Experimental Results](#7)
8. [✉️ Statement](#8)
9. [🔍 Citation](#9)

## 📎 Paper Link <a name="1"></a> 
* Learning Affordance Grounding from Exocentric Images (CVPR2022) [[pdf]()] [[Supplementary Material]()]
> Authors:
> Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao

## 💡 Abstract <a name="2"></a> 
Affordance grounding, a task to ground (i.e., localize) action possibility region in objects, which faces the challenge of establishing an explicit link with object parts due to the diversity of interactive affordance. Human has the ability that transform the various exocentric interactions to invariant egocentric affordance so as to counter the impact of interactive diversity. To empower an agent with such an ability, this paper proposes a task of affordance grounding from exocentric view, i.e., given exocentric human-object interaction and egocentric object images, learning the affordance knowledge of the object and transferring it to the egocentric image using only the affordance label as supervision. To this end, we devise a cross-view knowledge transfer framework that extracts affordance-specific features from exocentric interactions and enhances the perception of affordance regions by preserving affordance correlation. Specifically, an Affordance Invariance Mining module is devised to extract specific clues by minimizing the intra-class differences originated from interaction habits in exocentric images. Furthermore, an Affordnace Co-relation Preserving strategy is presented to perceive and localize affordance by aligning the co-relation matrix of predicted results between the two views. Particularly, an affordance grounding dataset named AGD20K is constructed by collecting and labeling over 20K images from 36 affordance categories. Experimental results demonstrate that our method outperforms the representative methods in terms of objective metrics and visual quality.

<p align="center">
    <img src="./img/fig1.png" width="600"/> <br />
    <em> 
    </em>
</p>

**Observation.** By observing the exocentric diverse interactions, the human learns affordance knowledge determined by the
object’s intrinsic properties and transfer it to the egocentric view.

<p align="center">
    <img src="./img/Motivation.png" width="700"/> <br />
    <em> 
    </em>
</p>

**Motivation.** (a) Exocentric interactions can be decomposed into affordance-specific features M and differences in individual
habits E. (b) There are co-relations between affordances, e.g.“Cut with” inevitably accompanies “Hold” and is independent of the object
category (knife and scissors). Such co-relation is common between objects. In this paper, we mainly consider extracting affordance-specific
cues M from diverse interactions while preserving the affordance co-relations to enhance the perceptual capability of the network.


## 📖 Method <a name="3"></a> 

<p align="center">
    <img src="./img/Method.png" width="800"/> <br />
    <em> 
    </em>
</p>

**Overview of the proposed cross-view knowledge transfer affordance grounding framework.** It mainly consists of an
Affordance Invariance Mining (AIM) module and an Affordance Co-relation Preservation (ACP) strategy. The AIM module (see in Sec.
3.1) aims to obtain invariant affordance representations from diverse exocentric interactions. The ACP strategy (see in Sec. 3.2) enhances
the network’s affordance perception by aligning the co-relation of the outputs of the two views.

## 📂 Dataset <a name="4"></a> 

<p align="center">
    <img src="./img/dataset.png" width="800"/> <br />
    <em> 
    </em>
</p>

**The properties of the AGD20K dataset.** (a) Some examples from the dataset. (b) The distribution of categories in AGD20K.
(c) The word cloud distribution of affordances in AGD20K. (d) Confusion matrix between the affordance category and the object category
in AGD20K, where the horizontal axis denotes the object category and the vertical axis denotes the affordance category.

## 📃 Requirements <a name="5"></a> 
  - python 3.7 
  - pytorch 1.1.0
  - opencv



## ✏️ Usage <a name="6"></a> 



## 📊 Experimental Results <a name="7"></a> 


## ✉️ Statement <a name="8"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [lhc12@mail.ustc.edu.cn](lhc12@mail.ustc.edu.cn) or [wzhai056@mail.ustc.edu.cn](wzhai056@mail.ustc.edu.cn).

## 🔍 Citation <a name="9"></a> 

```
@inproceedings{Learningluo,
  title={Learning Affordance Grounding from Exocentric Images},
  author={Hongchen Luo and Wei Zhai and Jing Zhang and Yang Cao and Dacheng Tao},
  booktitle={CVPR},
  year={2022}
}
```
