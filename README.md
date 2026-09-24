# SDN Network Traffic Prediction using Machine Learning & Lagged Features

This repository contains the code, datasets, and presentation for the project **"SDN Traffic Prediction with Lagged Features and Machine Learning Techniques"**, developed for the *Machine Learning for Smart Cities Automation* course.

The project focuses on predicting IP packet traffic over a Software Defined Networking (SDN) infrastructure using supervised learning techniques on real-world time-series data.

---

## 📌 Project Overview

Accurate prediction of network traffic in SDN routers/switches is crucial for enabling real-time Model Predictive Control (MPC) and optimal bandwidth allocation. This project aims to forecast incoming IP traffic packets ($\text{IN}$) at $h$ time steps ahead ($h = 3 \implies 15$ minutes, $h = 6 \implies 30$ minutes.

To capture temporal dynamics and long-term dependencies, we construct an extended **Lag Matrix** by shifting features across past time steps ($\text{lag} = \Delta x$).

### Key Features:
- **Real-World Dataset:** Collected from an Italian Internet Service Provider (**Sonicatel S.r.l.**) using Cacti monitoring software with a 5-minute sampling interval.
- **Traffic Breakdown:** Measures overall incoming (`IN`) and outgoing (`OUT`) traffic, as well as specific service streams: `VOIP`, `NETFLIX`, and `DAZN`.
- **Time Features:** Includes temporal attributes (`YEAR`, `MONTH`, `DAY`, `HOUR`, `MIN`, `DAYWEEK`) to capture network seasonality.
- **Benchmarked Models:**
  - **Least Squares (LS)** Batch / Pseudo-inverse
  - **Stochastic Gradient Descent (SGD / LMS)**
  - **Kernel Ridge Regression** (RBF & Polynomial Kernels)
  - **Support Vector Regression (SVR)** (RBF & Polynomial Kernels)
  - **Regression Trees (RT)**
  - **Random Forests (RF)** (Ensemble with dynamic tree scaling)

---

## Repository Structure

.
|-- SONICATEL traffic train.csv      # Training set (~8,291+ samples, 5-min intervals)
|-- SONICATEL traffic test.csv       # Test set (600+ samples)
|-- Progetto_SDN_lag.m               # MATLAB script with lagged features integration
|-- progetto_SDN_statico.m           # MATLAB script for standard horizon predictions
|-- KernelRidgeRegression.m          # Custom KRR implementation
|-- KernelPrediction.m               # Prediction function for Kernel models
|-- kernelmatrix.m                   # RBF & Polynomial Kernel Gram matrix calculation
|-- main_Globecom.pdf                # Reference scientific paper (E. Reticcioli et al.)
|-- README.md                        # Project documentation        
