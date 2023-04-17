# Grounded Affordance from Exocentric View
PyTorch implementation of our Cross-view-AG models. This repository contains PyTorch evaluation code, training code.

## Requirements <a name="5"></a> 
  - python 3.7 
  - pytorch 
  - opencv
## ️ Usage <a name="6"></a> 

```bash  
git clone https://github.com/lhc1224/Cross-View-AG.git
cd Cross-View-AG
```
### Download AGD20K <a name="41"></a> 
- You can download the AGD20K from [ [Google Drive](https://drive.google.com/file/d/1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1IRfho7xDAT0oJi5_mvP1sg) (g23n) ].
Download the dataset and place it in the dataset/ folder
- You can download the testsetv2 from [ [Google Drive]() | [Baidu Pan]() () ].
Download the dataset and place it in the dataset/AGD20K folder
### Train <a name="61"></a> 
You can download the pretrained model from [ [Google Drive](https://drive.google.com/file/d/1TqnkTTw0W5Kbx9PArHMvyzlmCBr9K5zZ/view?usp=share_link)], then move it to the `weights` folder
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

