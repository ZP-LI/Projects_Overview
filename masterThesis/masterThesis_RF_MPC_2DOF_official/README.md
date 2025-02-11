# Representation-Free Model Predictive Control (RF-MPC)

Representation-Free Model Predictive Control (RF-MPC) is a advance control algorithm for quadruped robots. RF-MPC represents the orientation using the rotation matrix and thus does not have the singularity issue associated with the Euler angles. The linear dynamics on the rotation matrix is derived using variation-based linearization (VBL)

## Author of this project

Zhiping Li, Email: zhiping.li@tum.de

## Notice

1. The source code is written in Matlab and published on https://github.com/YanranDing/RF-MPC.

2. What in this project do is to convert the source code to Python and modify the source code to control the quadruped robot in MuJoCo.

3. This project preserves the Matlab source code with folder/file name in the form of "XXX_matlab". The other folder/files are the real codes of this project.

4. For debugging propose, the timestep is 0.001 sec now. But for real-time simulation, the timestep need to be changed to 0.01 sec. To implement this change, the timestep setting in xml-file, the time-related parameters in "fcns/get_params.py" and the PD gain values (Kp, Kd) need to be modified.

5. The entire project contains "main.py" for high-level control, Class in "mujoco_robot.py" for interactions with MuJoCo, Class in "plot_results.py" for result plotting and other python files in folders "fcns" & "fcns_MPC" for other low-level functions.

## Requirement
Basic: Ubuntu 20.04 / MuJoCo 210 + mujoco-py / Python (Other versions may work!)

Optional but recommendable: qpSWIFT (can be obtained from https://github.com/qpSWIFT) 
(!! I have already downloaded this package in the folder with name "qpSWIFT-main", so it's no need to download the document again, but it still need to be installed on the computer following the tips in qpSWIFT GitHub).

## Source Code Citation
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

