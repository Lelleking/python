import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dataset import load_all_curves
from model import ConvAutoencoder

DATA_DIR   = "compressed data"
MODELS_DIR = "models"
PLOTS_DIR  = "plots"
RESULTS_DIR = "results"
LATENT_DIM = 32
K_MIN, K_MAX = 10, 25


def load_latent():
    latent = np.load(os.path.join(MODELS_DIR, "latent_vectors.npy"))
    metadata = []
    with open(os.path.join(MODELS_DIR, "metadata.csv"), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append((row["prefix"], row["well_id"]))
    return latent, metadata


def cluster_sweep(latent):
    ks = range(K_MIN, K_MAX + 1)
    inertias = []
    silhouettes = []
    labels_by_k = {}
    for k in ks:
        print(f"  k={k} …", end=" ", flush=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(latent)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(latent, labels))
        labels_by_k[k] = labels
        print("done")
    return list(ks), inertias, silhouettes, labels_by_k


def plot_elbow_silhouette(ks, inertias, silhouettes, best_k):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(ks, inertias, "o-", color="#1f77b4", label="Inertia")
    ax2.plot(ks, silhouettes, "s--", color="#ff7f0e", label="Silhouette")

    ax1.axvline(best_k, color="grey", linestyle=":", linewidth=1.5)
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia", color="#1f77b4")
    ax2.set_ylabel("Silhouette score", color="#ff7f0e")
    ax1.set_title(f"Elbow & silhouette  (best k={best_k})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "elbow_silhouette.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_reconstructions(curves, model, device, n=8):
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(curves), size=min(n, len(curves)), replace=False)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(curves[idxs]).unsqueeze(1).to(device)
        recon = model(x).squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for i, ax in enumerate(axes.flat):
        if i >= len(idxs):
            ax.axis("off")
            continue
        ax.plot(curves[idxs[i]], color="#1f77b4", linewidth=0.8, label="Original")
        ax.plot(recon[i], color="#ff7f0e", linewidth=0.8, linestyle="--", label="Reconstruction")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Reconstructions (original vs autoencoder)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "reconstructions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cluster_representatives(curves, labels, best_k):
    n_cols = min(best_k, 4)
    n_rows = (best_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)

    colours = plt.cm.tab10.colors

    for k in range(best_k):
        row, col = divmod(k, n_cols)
        ax = axes[row][col]
        members = curves[labels == k]
        for curve in members:
            ax.plot(curve, color="lightgrey", linewidth=0.5, alpha=0.6)
        median_curve = np.median(members, axis=0)
        ax.plot(median_curve, color=colours[k % 10], linewidth=2.0)
        ax.set_title(f"Cluster {k}  (n={len(members)})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused subplots
    for k in range(best_k, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(f"Cluster representatives  (k={best_k})", fontsize=11)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "cluster_representatives.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def save_assignments(metadata, labels, best_k):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "cluster_assignments.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "well_id", "cluster"])
        for (prefix, well_id), label in zip(metadata, labels):
            writer.writerow([prefix, well_id, label])
    print(f"Saved {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load latent vectors ──────────────────────────────────────
    latent, metadata = load_latent()
    print(f"Loaded {len(latent)} latent vectors  shape={latent.shape}")

    # ── Load model for reconstructions ──────────────────────────
    print("Loading model …")
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "cae.pt"), map_location=device, weights_only=True
    ))

    # ── Load raw curves for plots ────────────────────────────────
    print("Loading curves …")
    curves, _ = load_all_curves(DATA_DIR)   # (N, 1000)
    print(f"Loaded {len(curves)} curves")

    # ── Cluster sweep ────────────────────────────────────────────
    print(f"Running cluster sweep k={K_MIN}…{K_MAX} …")
    ks, inertias, silhouettes, labels_by_k = cluster_sweep(latent)
    best_k = ks[int(np.argmax(silhouettes))]
    print(f"Best k by silhouette: {best_k}  (score={max(silhouettes):.4f})")

    # ── Plots ────────────────────────────────────────────────────
    plot_elbow_silhouette(ks, inertias, silhouettes, best_k)
    plot_reconstructions(curves, model, device)
    plot_cluster_representatives(curves, labels_by_k[best_k], best_k)

    # ── Save assignments ─────────────────────────────────────────
    save_assignments(metadata, labels_by_k[best_k], best_k)

    print(f"\nDone. Cluster assignments written to {RESULTS_DIR}/cluster_assignments.csv")

    import subprocess
    subprocess.run([
        "osascript", "-e",
        'display notification "predict.py finished" with title "1D-CNN" sound name "Glass"'
    ], check=False)


if __name__ == "__main__":
    main()