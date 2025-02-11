# RF-MPC

Representation-Free Model Predictive Control (RF-MPC) is a MATLAB simulation framework for dynamic legged robots. RF-MPC represents the orientation using the rotation matrix and thus does not have the singularity issue associated with the Euler angles. The linear dynamics on the rotation matrix is derived using variation-based linearization (VBL).

![](https://i.imgur.com/mvZZUCj.gif)

video available at: [YouTube Video](https://www.youtube.com/watch?v=iMacEwQisoQ&t=101s)

## Notice

The source code is written in Matlab and published on https://github.com/YanranDing/RF-MPC.

What here do is to convert the source code to Python and try to use the source code to control the robot in MuJoCo.

## Requirement
Basic: MATLAB and MATLAB optimization toolbox

Optional: qpSWIFT (can be obtained from https://github.com/qpSWIFT; !! I have already downloaded this package in the folder with name "qpSWIFT-main", so it's no need to download the document again, but it still need to be installed on the computer following the tips in qpSWIFT github).

## Citation
    @ARTICLE{9321699,
    author={Y. {Ding} and A. {Pandala} and C. {Li} and Y. -H. {Shin} and H. -W. {Park}},
    journal={IEEE Transactions on Robotics}, 
    title={Representation-Free Model Predictive Control for Dynamic Motions in Quadrupeds}, 
    year={2021},
    volume={},
    number={},
    pages={1-18},
    doi={10.1109/TRO.2020.3046415}}

## References
This code is based on the following publications:
* Yanran Ding, Abhishek Pandala, Chuanzheng Li, Young-Ha Shin, Hae-Won Park "Representation-Free Model Predictive Control for Dynamic Motions in Quadrupeds". In IEEE Transactions on Robotics. [PDF](https://ieeexplore.ieee.org/document/9321699)
* Yanran Ding, Abhishek Pandala, and Hae-Won Park. "Real-time model predictive control for versatile dynamic motions in quadrupedal robots". In IEEE 2019 International Conference on Robotics and Automation (ICRA). [PDF](https://ieeexplore.ieee.org/abstract/document/8793669)


## Source Code Authors
[Yanran Ding](https://sites.google.com/view/yanranding/home) - Initial Work/Maintainer

