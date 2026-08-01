import numpy as np
import torch

from src.metrics.calculate_eer import compute_eer


def calculate_eer(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    eer, _ = compute_eer(
        bonafide_scores=np.asarray(bonafide_scores),
        other_scores=np.asarray(spoof_scores),
    )

    return float(eer * 100)