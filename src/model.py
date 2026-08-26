"""
AeroSurrogate: Physics-Informed MLP for NACA aerodynamic coefficient prediction.

Architecture
------------
Input(5) -> Dense(128, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Output(2)

Outputs
-------
[0] Cl  lift coefficient
[1] Cd  drag coefficient

L/D is derived as Cl/Cd after inference to guarantee physical consistency.

Input features (passed in normalised form — apply z-score before calling):
  [0] m_frac    max camber fraction          (0.00 – 0.09)
  [1] p_frac    max-camber position fraction  (0.10 – 0.90)
  [2] t_frac    max thickness fraction        (0.06 – 0.24)
  [3] alpha_deg angle of attack in degrees    (-10 – 20)
  [4] log10_Re  log_10 of Reynolds number     (5.0 – 7.0)

Normalisation statistics (mean/std computed on training data) are stored
outside the model, in the checkpoint and in model_weights.json.
"""

import torch
import torch.nn as nn


class AeroSurrogate(nn.Module):
    """
    MLP surrogate for aerodynamic coefficient prediction.

    Input/output scaling is applied externally (train.py and export_weights.py).
    This keeps the model weights independent of the specific dataset statistics,
    making it straightforward to retrain on different data ranges.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),     # [Cl, Cd]
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 5) normalised input features

        Returns
        -------
        (N, 2) normalised outputs [Cl_norm, Cd_norm]
        """
        return self.net(x)
