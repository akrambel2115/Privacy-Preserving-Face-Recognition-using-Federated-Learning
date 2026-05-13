# model definition
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FedFaceModel(nn.Module):
    """
    The global model broadcasted to all clients.
    Contains the FaceNet feature extractor and the full classification matrix W.

    Note on backbone freezing:
    The FedFace paper (Algorithm 1, line 6) jointly updates all of theta
    without any freezing. We default to that behavior. The legacy partial-
    freeze mode is still reachable via ``freeze_backbone=True`` for users
    who need to reduce per-client compute / memory at the cost of fidelity
    to the paper.
    """

    def __init__(
        self,
        num_clients: int,
        embedding_dim: int = 512,
        pretrained: str = "vggface2",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # 1. The Global Feature Extractor (f_theta).
        # Pre-trained on VGGFace2 to provide a robust initial embedding space.
        self.feature_extractor = InceptionResnetV1(pretrained=pretrained)

        if freeze_backbone:
            self._freeze_early_layers()

        # 2. The Global Classification Matrix (W).
        # Shape: (C, d) where C = num_clients and d = embedding_dim.
        # Randomly initialized here; each client's row will be overwritten
        # by Mean Feature Initialization (paper Eq. 6) before the first
        # local training round.
        self.W_matrix = nn.Parameter(torch.randn(num_clients, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract and L2-normalize facial features.

        The actual cosine similarity against W happens in the training loop
        because each client only computes loss against its own row of W.
        """
        features = self.feature_extractor(x)
        features = F.normalize(features, p=2, dim=1)
        return features

    def _freeze_early_layers(self) -> None:
        """
        Freeze the early convolutional blocks of the FaceNet backbone.

        Disabled by default. The paper updates the full backbone; we keep
        this method available for compute-constrained deployments.
        """
        for name, param in self.feature_extractor.named_parameters():
            if (
                not name.startswith("block8")
                and not name.startswith("last_linear")
                and not name.startswith("logits")
            ):
                param.requires_grad = False