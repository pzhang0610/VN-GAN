# VN-GAN
An tensorflow implementation of VN-GAN. The framework is as follow,
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
python run.py --is_train --batch_size 100
```

For generation, run

```
python run.py --batch_size 1
```
## Examples

An example of view normalized gaits in terms of input gaits in arbitrary views and their corresponding reference gaits. Each three columns are input gaits, reference gaits and synthesized gaits in normal view.

<div align=center>
<img src="./sample/sample.png" alt="Framework of VN-GAN" align=center />
</div>



