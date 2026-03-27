import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_all_curves
from model import ConvAutoencoder

DATA_DIR    = "compressed data"
MODELS_DIR  = "models"
EPOCHS               = 300
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 0.005
BATCH_SIZE           = 32
LR                   = 1e-3
LATENT_DIM           = 32


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = os.path.join(MODELS_DIR, "cae.pt")
    meta_path    = os.path.join(MODELS_DIR, "metadata.csv")

    # ── Load all curves from disk ─────────────────────────────────
    print("Loading curves …")
    curves, metadata = load_all_curves(DATA_DIR)
    print(f"Found {len(curves)} curves total in '{DATA_DIR}'")

    # ── Decide: fresh train or incremental fine-tune ──────────────
    existing_keys = set()
    if os.path.exists(meta_path) and os.path.exists(weights_path):
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["prefix"], row["well_id"]))

    new_idx = [i for i, (p, w) in enumerate(metadata) if (p, w) not in existing_keys]

    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)

    if existing_keys:
        if not new_idx:
            print("No new curves found — nothing to train. Exiting.")
            return
        print(f"{len(new_idx)} new curves found (out of {len(curves)} total). Fine-tuning existing model …")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        train_curves = curves[np.array(new_idx)]
        lr = LR * 0.1   # lower LR to avoid overwriting existing knowledge
    else:
        print(f"No existing model found. Training from scratch on {len(curves)} curves …")
        train_curves = curves
        lr = LR

    # ── Build loader from the training subset ─────────────────────
    from torch.utils.data import TensorDataset
    tensor   = torch.from_numpy(train_curves).unsqueeze(1)
    loader   = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────
    print(f"\nTraining on {len(train_curves)} curves for up to {EPOCHS} epochs "
          f"(early stop patience={EARLY_STOP_PATIENCE}) ...\n")
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(train_curves)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}")

        if avg_loss < best_loss * (1 - EARLY_STOP_MIN_DELTA):
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stop at epoch {epoch}  "
                      f"(no >{EARLY_STOP_MIN_DELTA*100:.1f}% improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    # ── Save weights ─────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"\nWeights saved → {weights_path}")

    # ── Encode ALL curves (fast single forward pass) ──────────────
    print("Encoding all curves …")
    model.eval()
    with torch.no_grad():
        x_all  = torch.from_numpy(curves).unsqueeze(1).to(device)
        latent = model.encode(x_all).cpu().numpy()

    np.save(os.path.join(MODELS_DIR, "latent_vectors.npy"), latent)
    print(f"Latent vectors saved  shape={latent.shape}")

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "well_id"])
        writer.writerows(metadata)
    print(f"Metadata saved → {meta_path}  ({len(metadata)} curves)")


if __name__ == "__main__":
    train()