import torch
import torch.nn as nn

LATENT_DIM = 32
L = 1000   # resampled curve length (must match dataset.RESAMPLE_LEN)

# Encoder flattened size: 64 channels × 125 timesteps after 3× MaxPool1d(2)
_ENC_FLAT = 64 * (L // 8)   # = 8000


class ConvAutoencoder(nn.Module):
    """
    1D convolutional autoencoder for fluorescence curve shapes.

    Input/output: (batch, 1, L)  — channel-first, L = 1000.
    Latent:       (batch, latent_dim)
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # → (B, 16, 500)

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # → (B, 32, 250)

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # → (B, 64, 125)
        )
        self.encoder_fc = nn.Linear(_ENC_FLAT, latent_dim)

        # ── Decoder ──────────────────────────────────────────────
        self.decoder_fc = nn.Linear(latent_dim, _ENC_FLAT)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                          # → (B, 32, 250)

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),                          # → (B, 16, 500)

            nn.ConvTranspose1d(16,  1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),                       # → (B,  1, 1000)
        )

    def encode(self, x):
        h = self.encoder_conv(x)               # (B, 64, 125)
        h = h.flatten(1)                       # (B, 8000)
        return self.encoder_fc(h)              # (B, latent_dim)

    def decode(self, z):
        h = self.decoder_fc(z)                 # (B, 8000)
        h = h.view(h.size(0), 64, L // 8)     # (B, 64, 125)
        return self.decoder_conv(h)            # (B, 1, 1000)

    def forward(self, x):
        return self.decode(self.encode(x))