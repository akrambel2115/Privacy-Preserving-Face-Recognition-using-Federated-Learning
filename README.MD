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
* facenet_pytorch 


## The main workflow of federated learning  :

1. A global model is initialized either randomly  or with a pre-trained model. 
2. The aggregating server sends the current version of the global model to the available clients
3. The clients train the global model on their local data for a few iterations and send the model updates back
4. The server aggregates the model updates to update the global model
5. Steps 2-4 are repeated until the global model converges or achieves the desired performance.
<img src="federated%20learning%20.png" alt="federated learning" width="600"/>


**Our approach**
<br>
each client will have a single identity, we will use a pre-trained model that will be fine-tuned on the local data of each client. The server will aggregate the model updates using FedAvg to create a global model that can recognize all identities across clients.