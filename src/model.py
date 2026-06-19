from torch import nn


class FootballPredictor(nn.Module):
    def __init__(self, n_inputs=10, n_outputs=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, n_outputs),
        )

    def forward(self, inputs):
        return self.network(inputs)
