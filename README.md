# VN-GAN
An tensor flow implementation of VN-GAN. The framework is as follow,
* Framework of VN-GAN

<div align=center>
<img src="./model/framework.png" width = "600" height = "300" alt="Framework of VN-GAN" align=center />
</div>

* Two-stage structure
<div align=center>
<img src="./model/two-stage.png" width = "600" height = "300" alt="Framework of VN-GAN" align=center />
</div>
## Requirements
* tensorflow 
* PIL
* matplotlib
* numpy
* logging
* shutil

## Dataset

Download the CASIA gait dataset B from http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp and put it in directory './data'.
## Usage

For training, run

```
python run_triplet_block.py --is_train --batch_size 100 --resume_iters None
```

For generation, run

```
python run_triplet_block.py --batch_size 1
```
## Examples

Attached are some examples, the first column are inputs, the first row are references, and the bottom 11 x 11 are generated gaits,

<img src="./sample/sample.png" width = "500" height = "500" alt="An example of generated gaits" align=center />


