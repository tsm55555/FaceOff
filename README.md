# FaceOff
### Steps towards physical adversarial attacks on facial recognition

<p align="center">
  <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/input-face-example.png" width="175"> 
  <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/delta-example.png" width="175"> 
  <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/combined-face-example.png" width="175"> 
  <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/target-face-example.png" width="175">
</p>

Input image on the left is detected as the target image on the right after the mask has been applied.

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Table of contents

* [Table of contents](#table-of-contents)
* [Description](#description)
* [Installation](#installation)
  + [Requirements](#requirements)
  + [Instructions](#instructions)
* [Citation](#citation)
* [References](#references)

## Description
The purpose of this library is to create adversarial attacks against the FaceNet face recognizer.  This is the preliminary work towards creating a more robust physical attack using a mask that a person could wear over their face.

For more details, please check out my [research poster](https://github.com/392781/FaceOff/blob/master/rlencevicius_poster.pdf).

The current pipeline consists of an aligned input image with a calculated mask.  This is then fed into a face detector using dlib's histogram of oriented gradients detector to test whether the face is still detected.  This is then passed to FaceNet where which ouputs a face embedding and a loss which is then calculated and propagated back.  This perturbs the input mask which generates enough of a disturbance to affect the loss.

The loss function maximizes the Euclidean distance between the inputs' true identity and minimizes the distance between the adversarial input and the target image.

An image of this process can be seen below.


<p align="center">
  <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/procedure.png" width=55%>
</p>

## Installation

### Requirements

This project works on Linux (Ubuntu 20.04).  Windows and Mac are not supported but may work.

### Instructions

1. Create a virtual environment

```bash
conda create -n facial_recognition python=3.8.5
conda activate facial_recognition
```

2. Clone the repo 

```git
git clone https://github.com/392781/FaceOff.git
```

3. Install the required libraries 

```bash
pip install -r requirements.txt
```

4. Install FaceOff from inside the folder where `setup.py` is located

```bash
pip install -e .
```

5. Import and use!

```python
from FaceOff.AFR import load_data, Attack
```

For training instructions look at [`example.py`](https://github.com/392781/FaceOff/blob/master/examples/example.py) to get started in less than 30 lines.

## Citation
Please cite `FaceOff` if used in your research:

```tex
@misc{FaceOff,
  author = {Ronald Lencevičius},
  howpublished = {GitHub},
  title = {Face-Off: Steps towards physical adversarial attacks on facial recognition},
  URL = {https://github.com/392781/FaceOff},
  month = {Aug}
  year = {2019},
}
```

## References
* Sharif, Mahmood, et al. "Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016.
* Wang, Mei, and Weihong Deng. "Deep face recognition: A survey." arXiv preprint arXiv:1804.06655 (2018).
* MacDonald, Bruce. “Fooling Facial Detection with Fashion.” Towards Data Science, Towards Data Science, 4 June 2019, towardsdatascience.com/fooling-facial-detection-with-fashion-d668ed919eb.
* Thys, Simen, et al. "Fooling automated surveillance cameras: adversarial patches to attack person detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2019.

Used the [PyTorch FaceNet implementation](https://github.com/timesler/facenet-pytorch) by Tim Esler
