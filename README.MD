# Privacy-Preserving Face Recognition using Federated Learning

## Project Overview

This project implements a privacy-preserving face recognition system where student biometric data 
never leaves their personal devices.

Instead of collecting all face images on a centralized server, 
the system uses Federated Learning (FL) to collaboratively
train a shared global model while keeping raw images stored locally.


**On Each Student Device:**

* Store face images locally
* Perform local training
* Send model updates (weights/gradients) to server
* Never share raw images

**On the Server:**

* Aggregate client updates using Federated Averaging (FedAvg)
* Update the global model
* Send updated model back to clients


**Libraries Used**

* PyTorch
* Flower

