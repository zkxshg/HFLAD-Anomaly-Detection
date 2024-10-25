# HFLAD

This repository contains code for the paper, HFLAD: Anomaly Detection for Multivariate Time
Series Data Based on Hierarchical Fusion Learning, by Zhide Chen, Nan Chen, Kexin Zhu, Xu Yang, Xun Yi, Ibrahim Kahlil, and Albert Y. Zomaya.

HFLAD provides an effective framework for multivariate time series anomaly detection. The hierarchical time encoder captures multi-resolution temporal dynamics.

(We are still working on this topic, and upload the completed version after publishing.)

## Overview

HFLAD (Hierarchical Fusion Learning Anomaly Detection) is an advanced anomaly detection model designed specifically for multivariate time series data. Given the high dimensionality and complex temporal dependencies inherent in multivariate data, traditional anomaly detection models often fall short. HFLAD addresses these challenges by integrating both feature and temporal correlations, providing a robust solution for detecting anomalies in dynamic systems such as industrial control, IT infrastructure, and more.

### Key Features

- **Hierarchical Time Encoding**: Captures temporal patterns across multiple time scales, combining short-term and long-term dependencies to enhance the model's understanding of the data's time-related characteristics.
- **Feature Encoding**: Extracts essential inter-variable relationships and reduces dimensionality, enabling the model to identify correlations between multiple variables within the time series.
- **Reconstruction-Based Detection**: Utilizes a Hierarchical Variational Auto-Encoder (HVAE) to reconstruct the input data. Anomalies are detected based on reconstruction errors, with larger errors indicating deviations from normal patterns.



## TFAD model architecture

![fig 1](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/1.jpg)

## Main Results

![2](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/2.JPG)

![3](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/3.JPG)

![4](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/4.JPG)

![6](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/6.jpg)

![7](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/7.jpg)

![8](https://github.com/zkxshg/HFLAD-Anomaly-Detection/blob/main/Pic/8.jpg)

### Contributions

HFLAD demonstrates superior performance on various real-world datasets, including those from water treatment systems, NASA sensors, and network intrusion detection. Extensive experiments reveal that HFLAD outperforms current state-of-the-art models, showcasing its capability in accurately identifying anomalies while maintaining a low false positive rate.

## Datasets

- SWaT:  Mathur, Aditya P., and Nils Ole Tippenhauer. "SWaT: A water treatment testbed for research and training on ICS security." *2016 international workshop on cyber-physical systems for smart water networks (CySWater)*. IEEE, 2016.
- MSL: Hundman, Kyle, et al. "Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding." *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 2018.
- KDD-Cup99: Lippmann, Richard, et al. "The 1999 DARPA off-line intrusion detection evaluation." *Computer networks* 34.4 (2000): 579-595.
- ASD: Li, Zhihan, et al. "Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding." *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining*. 2021.

## Contact
If you have any question or want to use the code, please contact zkxshg@gmail.com .

