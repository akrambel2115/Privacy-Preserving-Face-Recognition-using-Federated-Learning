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

1. Pre-training & Mean Feature Initialization
Global Pre-training: The server starts with a global face feature extractor(Facenet), parameterized by $\theta_{0}$, which is pre-trained on a publicly available dataset
Initialization on Clients: In the first communication round, this global model is sent to all clients. Instead of initializing their personal class embedding randomly, each client generates their initial class embedding ($w_{0}^{i}$) by taking the mean of the features extracted from their local images using the pre-trained model
Mathematical Representation: $w_{0}^{i}=\frac{1}{n_{i}}\sum_{j=1}^{n_{i}}f_{\theta_{0}}(x_{j}^{i})$
(provides a significantly better starting point than a random guess and prevents the embeddings from disrupting the already learned feature space)

2. Local Client Updates (Positive-Only Loss)Local Optimization: During each communication round, the client updates the model parameters ($\theta_{t}$) and its personal class embedding ($w_{t}^{i}$) using only its local data.The Loss Function: Because the client has no access to other people's faces (negative embeddings), it optimizes a "positive-only" loss function. Specifically a squared hinge loss with cosine distance:
$$l_{pos}(f_{\theta_{t}}(x),i)=max(0,m-(w_{t}^{i})^{T}f_{\theta_{t}}(x))^{2}$$
This simply encourages the model to make the user's face images highly similar to their own class embedding

3. Server Aggregation Transmitting Updates: After local training, clients send their updated parameters back to the central server. Importantly, they only send their model weights and their single class embedding, never the raw face images
Averaging: The server aggregates the feature extractor parameters ($\theta$) by taking a weighted average from all participating clients
4. Spreadout Regularization (The Server-Side Fix)Preventing Collapse: Because the clients only trained on their own faces, the embeddings are at risk of collapsing into each other.The Regularizer: To fix this, the server applies a geometric technique called a Spreadout Regularizer directly to the concatenated class embedding matrix $W_{t}$.How it Works: This regularizer actively pushes the different class embeddings away from each other in the feature space. It penalizes the model if any two different class embeddings are too close to one another, enforcing a minimum margin of separation



