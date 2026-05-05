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


## Secure aggregation with Flower SecAgg+

The project also exposes a Flower `ServerApp`/`ClientApp` path that uses
Flower's built-in SecAgg+ protocol. SecAgg+ masks client model updates so the
server receives only the aggregated update, not any individual client's update.


For this face-recognition workflow, the secure path exchanges only the shared
FaceNet backbone. Each client's personal embedding row stays local on that
client and is never sent to the server. Secure runs use equal client weighting
and do not report per-client loss/sample metrics in the clear. Optional local
differential privacy uses Flower's `LocalDpMod` to clip and noise the backbone
update before it enters SecAgg+.

Run with differential privacy disabled:

```bash
python scripts/run_simulation.py --data-dir data/your_dataset --num-clients 5
```

Run with client-side DP enabled for backbone updates:

```bash
python scripts/run_simulation.py --data-dir data/your_dataset --num-clients 5 --local-dp-enabled --local-dp-clipping-norm 1.0 --local-dp-sensitivity 1.0 --local-dp-epsilon 5.0 --local-dp-delta 0.00001
```

For a local deployment runtime, start one SuperLink and one SuperNode per
client, then submit the app to the `local-deployment` federation:

```bash
python scripts/run_server.py --insecure
python scripts/run_client.py --client-id 0 --num-clients 5 --insecure
python scripts/run_client.py --client-id 1 --num-clients 5 --insecure --clientappio-api-address 127.0.0.1:9095
python scripts/run_simulation.py --data-dir data/your_dataset --num-clients 5 --federation local-deployment
```

secure aggregation flow:

1. The server runs a Flower ServerApp with SecAggPlusWorkflow.
2. Each client runs a ClientApp with secaggplus_mod.
3. During training, clients mask/encrypt their model updates before sending them.
4. The server cannot inspect individual client updates. It only receives the unmasked aggregate after Flower completes the SecAgg+ protocol.
5. Optional Flower `LocalDpMod` clips/noises the backbone update before masking.
6. The shared FaceNet backbone is aggregated securely across clients.
7. Client-specific embeddings remain local and are not part of the secure payload.


## The main workflow of federated learning  :

1. A global model is initialized either randomly  or with a pre-trained model. 
2. The aggregating server sends the current version of the global model to the available clients
3. The clients train the global model on their local data for a few iterations and send the model updates back
4. The server aggregates the model updates to update the global model
5. Steps 2-4 are repeated until the global model converges or achieves the desired performance.
<img src="federated%20learning%20.png" alt="federated learning" width="600"/>


**Our approach**
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



