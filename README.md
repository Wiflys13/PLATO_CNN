# Convolucional Neural Network for PLATO ESA Mission
![PLATO conceptual Image](images/Conceptual_Image.png)

![Badge In development](https://img.shields.io/badge/STATUS-In%20development-green)
![Last Update](https://img.shields.io/badge/Last_Update-October_2024-blue)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Author
<table>
<tr>
<td align="left">

<img src="images/plato_logo_320x320.png" alt="PLATO Logo Image" width="150">

</td>
<td align="left">

**Guillermo Mercant**  

AIT Engineer at the CSIC Astrobiology Centre (CAB)  
[GitHub Profile](https://github.com/Wiflys13)  
[guillermomercant@gmail.com](mailto:guillermomercant@gmail.com)

</td>
</tr>
</table>
Feel free to reach out for questions, collaborations, or feedback on this project!


# PLATO Mission Focus Temperature Prediction using CNN
This repository contains the development of a Convolutional Neural Network (CNN) designed to predict the optimal focus temperature for the PLATO space mission, led by the European Space Agency (ESA). The primary scientific goal of the PLATO mission is to detect and characterize exoplanets around Sun-like stars. The mission's success depends on achieving precise focus, which is regulated by the thermal conditions of its opto-mechanical systems.

## About the PLATO Mission
The PLATO mission (PLAnetary Transits and Oscillations of stars) aims to identify and study exoplanets by observing planetary transits and stellar oscillations. To achieve high-precision measurements, PLATO’s optical system requires optimal focus, which is adjusted based on the thermal state of its mechanisms.

During ground tests, operational space conditions are simulated to identify the ideal focus temperature. This is done by analyzing image quality at different temperatures, thus enabling calibration of the instruments prior to launch.

## Project Objective
The objective of this project is to develop a CNN-based method for determining PLATO’s focus temperature, providing an alternative approach to traditional methods. The CNN will be trained using images obtained during ground tests to facilitate:

* Pre-launch calibration based on environmental simulations,
* In-orbit calibration for ongoing adjustments during the mission.

By automating focus temperature calibration, this project aims to improve the reliability and efficiency of the mission's focus system

## Repository Structure

```
├── images/                     # Sample images from ground tests
├── data/                       # Data used for training and validation
├── notebooks/                  # Jupyter notebooks for experiments and analyses
├── src/                        # Source code for CNN model and preprocessing
│   ├── preprocessing           # Data preprocessing scripts
│   ├── train                   # Training pipeline
│   └── model                   # CNN architecture and training functions
├── README.md                   # Project documentation
└── requirements.txt            # Required packages and dependencies
```

## Getting Started

### Prerequisites
### Running the model
### Results


