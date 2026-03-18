import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FluorescenceDataset, load_all_curves
from model import ConvAutoencoder

DATA_DIR    = "compressed data"
MODELS_DIR  = "models"
EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 1e-3
LATENT_DIM  = 32


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────
    dataset = FluorescenceDataset(DATA_DIR)
    print(f"Loaded {len(dataset)} curves from '{DATA_DIR}'")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ── Model ────────────────────────────────────────────────────
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs ...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}")

    # ── Save weights ─────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    weights_path = os.path.join(MODELS_DIR, "cae.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"\nWeights saved → {weights_path}")

    # ── Encode all curves → latent vectors ───────────────────────
    curves, metadata = load_all_curves(DATA_DIR)
    model.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(curves).unsqueeze(1).to(device)   # (N, 1, 1000)
        latent = model.encode(x_all).cpu().numpy()                 # (N, 32)

    latent_path = os.path.join(MODELS_DIR, "latent_vectors.npy")
    np.save(latent_path, latent)
    print(f"Latent vectors saved → {latent_path}  shape={latent.shape}")

    meta_path = os.path.join(MODELS_DIR, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "well_id"])
        writer.writerows(metadata)
    print(f"Metadata saved → {meta_path}")


if __name__ == "__main__":
    train()
