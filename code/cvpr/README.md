# Learning Affordance Grounding from Exocentric Images
PyTorch implementation of our one-shot affordance detection models. This repository contains PyTorch evaluation code, training code.
**The properties of the AGD20K dataset.** (a) Some examples from the dataset. (b) The distribution of categories in AGD20K. (c) The word cloud distribution of affordances in AGD20K. (d) Confusion matrix between the affordance category and the object category in AGD20K, where the horizontal axis denotes the object category and the vertical axis denotes the affordance category.

## 📃 Requirements <a name="5"></a> 
  - python 3.7 
  - pytorch 
  - opencv
## ✏️ Usage <a name="6"></a> 

```bash  
git clone https://github.com/lhc1224/Cross-View-AG.git
cd Cross-View-AG
```
### Download AGD20K <a name="41"></a> 
- You can download the AGD20K from [ [Google Drive](https://drive.google.com/file/d/1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1IRfho7xDAT0oJi5_mvP1sg) (g23n) ].
Download the dataset and place it in the dataset/ folder
### Train <a name="61"></a> 
You can download the pretrained model from [ [Google Drive](https://drive.google.com/file/d/16OYi8kAxHosfCo8E4gmFIhwemW1FaCEB/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1HbsvNctWd6XLXFcbIoq1ZQ) (xjk5) ], then move it to the `weights` folder
To train the Cross-View-AG model, run `bash run.sh` with the desired model architecture:
```bash  
 bash run.sh   
```
You can process the label data by running `process_gt.py` in process_data.
```bash  
 cd process_data
 python process_gt.py  
```
You can test the trained model by running `test.py`.

```bash  
 python test.py  
```

## 🔍 Citation <a name="9"></a> 

```
@inproceedings{Learningluo,
  title={Learning Affordance Grounding from Exocentric Images},
  author={Luo, Hongchen and Zhai, Wei and Zhang, Jing and Cao, Yang and Tao, Dacheng},
  booktitle={CVPR},
  year={2022}
}
```
