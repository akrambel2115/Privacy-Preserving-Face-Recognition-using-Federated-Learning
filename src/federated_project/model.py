# model definition
# model definition
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FedFaceModel(nn.Module):
    """
    The global model broadcasted to all clients.
    It contains the FaceNet feature extractor and the full classification matrix W.
    """
    def __init__(self, num_clients, embedding_dim=512, pretrained='vggface2'):
        super(FedFaceModel, self).__init__()
        
        # 1. The Global Feature Extractor (f_theta)
        # We use FaceNet pre-trained on VGGFace2 to provide a robust initial embedding space.
        self.feature_extractor = InceptionResnetV1(pretrained=pretrained)
        
        # Optional: Freeze early layers to save memory and compute on local clients.
        self._freeze_early_layers()
        
        # 2. The Global Classification Matrix (W)
        # Shape: (C, d) where C = num_clients and d = embedding_dim.
        # This is initialized randomly here, but the client will overwrite its specific row 
        # using Mean Feature Initialization before its first local training round.
        self.W_matrix = nn.Parameter(torch.randn(num_clients, embedding_dim))

    def forward(self, x):
        """
        The forward pass extracts and normalizes the facial features.
        
        Note: The actual slicing of W_matrix and the cosine similarity 
        computation happen in the training loop (train.py) because each 
        client only needs to compute the loss against its own specific row.
        """
        # Extract features from the image: f_theta(x)
        features = self.feature_extractor(x)
        
        # L2-normalize the instance embeddings to constrain them to a hypersphere
        features = F.normalize(features, p=2, dim=1)
        
        return features

    def _freeze_early_layers(self):
        """
        Freezes the early convolutional blocks of the FaceNet backbone.
        Since clients are only fine-tuning the model for their specific face, 
        updating the entire deep network is usually unnecessary and computationally heavy.
        """
        for name, param in self.feature_extractor.named_parameters():
            # Freeze everything except the final blocks and fully connected layers
            if not name.startswith('block8') and not name.startswith('last_linear') and not name.startswith('logits'):
                param.requires_grad = False