"""DNN 모델의 단일 정의원(single source of truth).

학습 코드와 추후 SHAP 설명 코드가 이 클래스를 함께 사용해야 모델 구조와
저장된 가중치가 어긋나지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """인수인계서에서 확정한 MLP 하이퍼파라미터."""

    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 4096
    max_epochs: int = 7
    early_stopping_patience: int = 5
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        return result


class DeepMLP(nn.Module):
    """Input -> 256 -> 128 -> 64 -> 1 구조의 이진 분류 DNN.

    마지막 값은 확률이 아니라 logit이다. 학습 손실은 수치적으로 안정적인
    ``BCEWithLogitsLoss``를 사용하고, 추론할 때만 sigmoid를 적용한다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim은 1 이상이어야 합니다.")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims에는 1 이상의 은닉층 크기가 필요합니다.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """각 행에 대한 logit을 ``[batch]`` 형태로 반환한다."""

        return self.network(features).squeeze(-1)


def set_global_seed(seed: int) -> None:
    """Python, NumPy, PyTorch의 난수 상태를 가능한 범위에서 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
