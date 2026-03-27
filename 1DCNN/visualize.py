"""
Interactive dashboard for 1D-CNN autoencoder clustering results.
Run:  python visualize.py
Then open: http://localhost:5001
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import csv
import json
import numpy as np
import torch
from flask import Flask, render_template_string, jsonify, request
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from dataset import load_all_curves
from model import ConvAutoencoder

MODELS_DIR  = "models"
DATA_DIR    = "compressed data"
LATENT_DIM  = 32
K_MIN, K_MAX = 10, 25
PORT = 5001

app = Flask(__name__)

def find_knee(ks, inertias):
    """Geometric knee: max perpendicular distance from the line first→last."""
    x = np.array(ks, dtype=float)
    y = np.array(inertias, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    dists = np.abs(x_n + y_n - 1) / np.sqrt(2)
    return ks[int(np.argmax(dists))]

# ── Load everything once at startup ──────────────────────────────────────────

def _load():
    print("Loading latent vectors …")
    latent = np.load(os.path.join(MODELS_DIR, "latent_vectors.npy"))

    metadata = []
    with open(os.path.join(MODELS_DIR, "metadata.csv"), newline="") as f:
        for row in csv.DictReader(f):
            metadata.append((row["prefix"], row["well_id"]))

    print("Loading curves …")
    curves, _ = load_all_curves(DATA_DIR)

    print("Loading model …")
    model = ConvAutoencoder(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "cae.pt"), map_location="cpu", weights_only=True
    ))
    model.eval()

    print("Running cluster sweep …")
    ks = list(range(K_MIN, K_MAX + 1))
    inertias, silhouettes, ch_scores, db_scores, labels_by_k = [], [], [], [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        lbl = km.fit_predict(latent)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(latent, lbl)))
        ch_scores.append(float(calinski_harabasz_score(latent, lbl)))
        db_scores.append(float(davies_bouldin_score(latent, lbl)))
        labels_by_k[k] = lbl.tolist()

    best_k_sil   = ks[int(np.argmax(silhouettes))]
    best_k_ch    = ks[int(np.argmax(ch_scores))]
    best_k_db    = ks[int(np.argmin(db_scores))]
    best_k_elbow = find_knee(ks, inertias)
    best_k = Counter([best_k_sil, best_k_ch, best_k_db, best_k_elbow]).most_common(1)[0][0]

    print(f"  Silhouette → k={best_k_sil}  |  CH → k={best_k_ch}  |  DB → k={best_k_db}  |  Elbow → k={best_k_elbow}")
    print(f"  Consensus best k = {best_k}")
    print("Ready.\n")

    return (latent, metadata, curves, model, ks, inertias, silhouettes,
            ch_scores, db_scores, labels_by_k,
            best_k_sil, best_k_ch, best_k_db, best_k_elbow, best_k)


STATE = _load()
(LATENT, METADATA, CURVES, MODEL, KS, INERTIAS, SILHOUETTES,
 CH_SCORES, DB_SCORES, LABELS_BY_K,
 BEST_K_SIL, BEST_K_CH, BEST_K_DB, BEST_K_ELBOW, BEST_K) = STATE


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/elbow")
def api_elbow():
    return jsonify(
        ks=KS, inertias=INERTIAS, silhouettes=SILHOUETTES,
        ch_scores=CH_SCORES, db_scores=DB_SCORES,
        best_k_sil=BEST_K_SIL, best_k_ch=BEST_K_CH,
        best_k_db=BEST_K_DB, best_k_elbow=BEST_K_ELBOW,
        best_k=BEST_K,
    )


@app.get("/api/pca/<int:k>")
def api_pca(k):
    k = max(K_MIN, min(K_MAX, k))
    labels = LABELS_BY_K[k]
    colours = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    ]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(LATENT)
    clusters = []
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        clusters.append(dict(
            id=c,
            color=colours[c % len(colours)],
            x=[float(coords[i, 0]) for i in idx],
            y=[float(coords[i, 1]) for i in idx],
            text=[f"{METADATA[i][0]} / {METADATA[i][1]}" for i in idx],
        ))
    return jsonify(
        clusters=clusters,
        var_explained=round(float(pca.explained_variance_ratio_.sum() * 100), 1),
    )


@app.get("/api/clusters/<int:k>")
def api_clusters(k):
    k = max(K_MIN, min(K_MAX, k))
    labels = np.array(LABELS_BY_K[k])
    colours = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    ]
    clusters = []
    rng = np.random.default_rng(42)
    for c in range(k):
        idx = np.where(labels == c)[0]
        members = CURVES[idx]          # (n, 1000)
        median  = np.median(members, axis=0).tolist()
        # sample up to 40 background curves to keep payload small
        sample_idx = rng.choice(len(idx), size=min(40, len(idx)), replace=False)
        sample = members[sample_idx].tolist()
        meta = [(METADATA[i][0], METADATA[i][1]) for i in idx]
        clusters.append(dict(
            id=c,
            color=colours[c % len(colours)],
            n=int(len(idx)),
            median=median,
            sample=sample,
            meta=meta,
        ))
    return jsonify(k=k, clusters=clusters)


@app.get("/api/reconstructions")
def api_reconstructions():
    n = int(request.args.get("n", 8))
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(CURVES), size=min(n, len(CURVES)), replace=False)
    with torch.no_grad():
        x = torch.from_numpy(CURVES[idxs]).unsqueeze(1)
        recon = MODEL(x).squeeze(1).numpy()
    items = []
    for i, idx in enumerate(idxs):
        prefix, well = METADATA[idx]
        items.append(dict(
            original=CURVES[idx].tolist(),
            recon=recon[i].tolist(),
            label=f"{prefix} / {well}",
        ))
    return jsonify(items=items)


@app.get("/api/assignments")
def api_assignments():
    k = int(request.args.get("k", BEST_K))
    k = max(K_MIN, min(K_MAX, k))
    labels = LABELS_BY_K[k]
    rows = [
        dict(prefix=METADATA[i][0], well=METADATA[i][1], cluster=int(labels[i]))
        for i in range(len(METADATA))
    ]
    return jsonify(rows=rows, k=k)


# ── Main page ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>1D-CNN Clustering Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f4f6f9; color: #222; }
  header { background: #1a1a2e; color: #fff; padding: 14px 24px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.2rem; font-weight: 600; }
  header .badge { background: #e94560; border-radius: 12px;
                  padding: 2px 10px; font-size: 0.78rem; }
  nav { display: flex; gap: 0; border-bottom: 2px solid #ddd;
        background: #fff; padding: 0 24px; }
  nav button { background: none; border: none; padding: 12px 20px;
               font-size: 0.92rem; cursor: pointer; color: #555;
               border-bottom: 3px solid transparent; margin-bottom: -2px; }
  nav button.active { color: #1a1a2e; border-bottom-color: #e94560; font-weight: 600; }
  nav button:hover:not(.active) { color: #222; }
  .tab { display: none; padding: 24px; }
  .tab.active { display: block; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px #0001;
          padding: 20px; margin-bottom: 20px; }
  .card h2 { font-size: 1rem; margin-bottom: 14px; color: #444; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; }
  .row .card { flex: 1; min-width: 300px; }
  .slider-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
  .slider-row label { font-weight: 600; white-space: nowrap; }
  input[type=range] { flex: 1; accent-color: #e94560; }
  #k-display { font-size: 1.1rem; font-weight: 700; color: #e94560; min-width: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { background: #1a1a2e; color: #fff; padding: 8px 10px; text-align: left; }
  td { padding: 7px 10px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f0f4ff; }
  .search-row { display: flex; gap: 10px; margin-bottom: 12px; align-items: center; }
  .search-row input, .search-row select {
    padding: 7px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.88rem; }
  .search-row input { flex: 1; }
  .cluster-pill { display: inline-block; border-radius: 12px; padding: 2px 10px;
                  font-size: 0.78rem; font-weight: 700; color: #fff; }
  #loading { position: fixed; inset: 0; background: #fff9; display: flex;
             align-items: center; justify-content: center; font-size: 1.4rem;
             z-index: 999; pointer-events: none; transition: opacity .3s; }
</style>
</head>
<body>

<div id="loading">Loading …</div>

<header>
  <h1>1D-CNN Clustering Dashboard</h1>
  <span class="badge" id="hdr-best-k">best k = ?</span>
  <span class="badge" style="background:#16213e">{{ total_wells }} wells</span>
</header>

<nav>
  <button class="active" onclick="showTab('overview')">Overview</button>
  <button onclick="showTab('clusters')">Clusters</button>
  <button onclick="showTab('latent')">Latent Space</button>
  <button onclick="showTab('recon')">Reconstructions</button>
  <button onclick="showTab('table')">Assignments</button>
</nav>

<!-- Overview tab -->
<div class="tab active" id="tab-overview">
  <div class="card" id="consensus-card" style="margin-bottom:20px">
    <h2>Consensus recommendation</h2>
    <div id="consensus-body" style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:10px"></div>
  </div>
  <div class="row">
    <div class="card">
      <h2>Elbow (inertia) — knee marked</h2>
      <div id="plot-inertia" style="height:260px"></div>
    </div>
    <div class="card">
      <h2>Silhouette score ↑ higher is better</h2>
      <div id="plot-silhouette" style="height:260px"></div>
    </div>
  </div>
  <div class="row">
    <div class="card">
      <h2>Calinski-Harabasz index ↑ higher is better</h2>
      <div id="plot-ch" style="height:260px"></div>
    </div>
    <div class="card">
      <h2>Davies-Bouldin index ↓ lower is better</h2>
      <div id="plot-db" style="height:260px"></div>
    </div>
  </div>
</div>

<!-- Latent Space tab -->
<div class="tab" id="tab-latent">
  <div class="card">
    <div class="slider-row">
      <label>Colour by k =</label>
      <span id="pca-k-display">?</span>
      <input type="range" id="pca-k-slider" min="{{ k_min }}" max="{{ k_max }}" step="1" value="{{ best_k }}"
             oninput="onPcaKChange(this.value)">
    </div>
    <p id="pca-var" style="font-size:0.82rem;color:#888;margin-bottom:12px"></p>
    <div id="plot-pca" style="height:560px"></div>
  </div>
</div>

<!-- Clusters tab -->
<div class="tab" id="tab-clusters">
  <div class="card">
    <div class="slider-row">
      <label>Number of clusters k =</label>
      <span id="k-display">?</span>
      <input type="range" id="k-slider" min="{{ k_min }}" max="{{ k_max }}" step="1" value="{{ best_k }}"
             oninput="onKChange(this.value)">
    </div>
    <div id="cluster-summary" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px"></div>
    <div id="plot-clusters" style="height:600px"></div>
  </div>
</div>

<!-- Reconstructions tab -->
<div class="tab" id="tab-recon">
  <div class="card">
    <h2>Original vs reconstructed (8 random wells)</h2>
    <div id="plot-recon" style="height:500px"></div>
  </div>
</div>

<!-- Assignments table tab -->
<div class="tab" id="tab-table">
  <div class="card">
    <div class="search-row">
      <input id="tbl-search" type="text" placeholder="Filter by experiment or well…" oninput="filterTable()">
      <select id="tbl-cluster" onchange="filterTable()"><option value="">All clusters</option></select>
      <select id="tbl-k" onchange="reloadTable()">
        {% for k in ks %}
        <option value="{{ k }}" {% if k == best_k %}selected{% endif %}>k = {{ k }}</option>
        {% endfor %}
      </select>
    </div>
    <table>
      <thead><tr><th>Experiment</th><th>Well</th><th>Cluster</th></tr></thead>
      <tbody id="tbl-body"></tbody>
    </table>
    <p id="tbl-count" style="margin-top:8px;font-size:0.82rem;color:#888"></p>
  </div>
</div>

<script>
const CLUSTER_COLOURS = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5"
];

let bestK = {{ best_k }};
let allRows = [];
let currentLabels = [];
const x1000 = Array.from({length:1000}, (_,i) => i);

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ── Overview ──────────────────────────────────────────────────────────────────
async function loadOverview() {
  const d = await fetch('/api/elbow').then(r => r.json());
  bestK = d.best_k;
  document.getElementById('hdr-best-k').textContent = 'consensus k = ' + bestK;
  document.getElementById('k-slider').value = bestK;
  document.getElementById('k-display').textContent = bestK;
  document.getElementById('pca-k-slider').value = bestK;
  document.getElementById('pca-k-display').textContent = bestK;

  // consensus card
  const methods = [
    {label:'Silhouette', k: d.best_k_sil, color:'#ff7f0e'},
    {label:'Calinski-Harabasz', k: d.best_k_ch, color:'#2ca02c'},
    {label:'Davies-Bouldin', k: d.best_k_db, color:'#9467bd'},
    {label:'Elbow knee', k: d.best_k_elbow, color:'#1f77b4'},
  ];
  document.getElementById('consensus-body').innerHTML =
    methods.map(m =>
      `<span style="background:${m.color};color:#fff;border-radius:8px;padding:5px 14px;font-size:0.85rem;font-weight:600">
        ${m.label}: k = ${m.k}
      </span>`
    ).join('') +
    `<span style="background:#e94560;color:#fff;border-radius:8px;padding:5px 18px;
                  font-size:1rem;font-weight:700;margin-left:8px">
      ★ Consensus: k = ${d.best_k}
    </span>`;

  const L = (extra) => ({margin:{t:10,b:40,l:55,r:20}, plot_bgcolor:'#fafafa',
                          paper_bgcolor:'#fff', font:{size:12}, ...extra});
  const star = (ks, vals, k) => ({
    x:[k], y:[vals[ks.indexOf(k)]],
    mode:'markers', marker:{color:'#e94560', size:13, symbol:'star'},
    showlegend:false, hoverinfo:'skip',
  });
  const line = (ks, vals, color) => ({
    x:ks, y:vals, mode:'lines+markers',
    marker:{color, size:6}, line:{color}, showlegend:false,
  });

  Plotly.newPlot('plot-inertia',
    [line(d.ks, d.inertias,'#1f77b4'), star(d.ks, d.inertias, d.best_k_elbow)],
    L({xaxis:{title:'k'}, yaxis:{title:'Inertia'}}), {responsive:true});

  Plotly.newPlot('plot-silhouette',
    [line(d.ks, d.silhouettes,'#ff7f0e'), star(d.ks, d.silhouettes, d.best_k_sil)],
    L({xaxis:{title:'k'}, yaxis:{title:'Silhouette'}}), {responsive:true});

  Plotly.newPlot('plot-ch',
    [line(d.ks, d.ch_scores,'#2ca02c'), star(d.ks, d.ch_scores, d.best_k_ch)],
    L({xaxis:{title:'k'}, yaxis:{title:'CH index'}}), {responsive:true});

  Plotly.newPlot('plot-db',
    [line(d.ks, d.db_scores,'#9467bd'), star(d.ks, d.db_scores, d.best_k_db)],
    L({xaxis:{title:'k'}, yaxis:{title:'DB index'}}), {responsive:true});
}

// ── Latent Space (PCA) ────────────────────────────────────────────────────────
let pcaLoaded = {};
async function loadPca(k) {
  if (!pcaLoaded[k]) {
    const d = await fetch('/api/pca/' + k).then(r => r.json());
    pcaLoaded[k] = d;
  }
  renderPca(k);
}

function renderPca(k) {
  const d = pcaLoaded[k];
  if (!d) return;
  document.getElementById('pca-var').textContent =
    `PCA — top 2 components explain ${d.var_explained}% of variance`;
  const traces = d.clusters.map(cl => ({
    x: cl.x, y: cl.y, text: cl.text,
    mode: 'markers', type: 'scatter',
    marker: {color: cl.color, size: 6, opacity: 0.75},
    name: 'Cluster ' + cl.id,
    hovertemplate: '%{text}<extra>Cluster ' + cl.id + '</extra>',
  }));
  Plotly.react('plot-pca', traces, {
    margin:{t:20,b:50,l:50,r:20},
    paper_bgcolor:'#fff', plot_bgcolor:'#fafafa',
    xaxis:{title:'PC 1'}, yaxis:{title:'PC 2'},
    legend:{orientation:'h', y:-0.12},
  }, {responsive:true});
}

function onPcaKChange(val) {
  document.getElementById('pca-k-display').textContent = val;
  loadPca(parseInt(val));
}

// ── Clusters ──────────────────────────────────────────────────────────────────
let clusterLoaded = {};
async function loadClusters(k) {
  if (clusterLoaded[k]) return;
  const d = await fetch('/api/clusters/' + k).then(r => r.json());
  clusterLoaded[k] = d;
  renderClusters(k);
}

function renderClusters(k) {
  const d = clusterLoaded[k];
  if (!d) return;

  // cluster count summary
  const total = d.clusters.reduce((s, cl) => s + cl.n, 0);
  document.getElementById('cluster-summary').innerHTML =
    d.clusters.map(cl =>
      `<span style="background:${cl.color};color:#fff;border-radius:12px;padding:3px 12px;font-size:0.82rem;font-weight:600">
        Cluster ${cl.id}: ${cl.n} wells (${(cl.n/total*100).toFixed(1)}%)
      </span>`
    ).join('');

  const nc = d.clusters.length;
  const ncols = Math.min(nc, 4);
  const nrows = Math.ceil(nc / ncols);

  const traces = [];
  const layout = {
    grid: {rows: nrows, columns: ncols, pattern: 'independent'},
    margin: {t: 40, b: 20, l: 40, r: 20},
    paper_bgcolor: '#fff',
    showlegend: false,
    height: Math.max(300, nrows * 200),
  };

  d.clusters.forEach((cl, ci) => {
    const ax = ci === 0 ? '' : (ci + 1);
    // background curves
    cl.sample.forEach((curve, si) => {
      traces.push({
        x: x1000, y: curve,
        xaxis: 'x' + ax, yaxis: 'y' + ax,
        mode: 'lines',
        line: {color: '#ccc', width: 0.6},
        opacity: 0.5,
        hoverinfo: 'skip',
        showlegend: false,
      });
    });
    // median
    traces.push({
      x: x1000, y: cl.median,
      xaxis: 'x' + ax, yaxis: 'y' + ax,
      mode: 'lines',
      line: {color: cl.color, width: 2.5},
      name: 'Cluster ' + cl.id,
      hoverinfo: 'name',
    });

    layout['xaxis' + ax] = {visible: false};
    layout['yaxis' + ax] = {visible: false};
    layout['annotations'] = layout['annotations'] || [];
    // title annotation per subplot
    const col = ci % ncols;
    const row = Math.floor(ci / ncols);
    layout['annotations'].push({
      text: `Cluster ${cl.id}  (n=${cl.n})`,
      xref: 'paper', yref: 'paper',
      x: (col + 0.5) / ncols,
      y: 1 - row / nrows,
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: {size: 11, color: cl.color},
    });
  });

  Plotly.react('plot-clusters', traces, layout, {responsive: true});
}

function onKChange(val) {
  document.getElementById('k-display').textContent = val;
  const k = parseInt(val);
  if (clusterLoaded[k]) {
    renderClusters(k);
  } else {
    loadClusters(k);
  }
}

// ── Reconstructions ───────────────────────────────────────────────────────────
async function loadRecon() {
  const d = await fetch('/api/reconstructions?n=8').then(r => r.json());
  const nc = 4, nr = 2;
  const traces = [];
  const layout = {
    grid: {rows: nr, columns: nc, pattern: 'independent'},
    margin: {t: 30, b: 10, l: 10, r: 10},
    paper_bgcolor: '#fff',
    showlegend: false,
    height: 460,
  };

  d.items.forEach((item, ci) => {
    const ax = ci === 0 ? '' : (ci + 1);
    traces.push({
      x: x1000, y: item.original,
      xaxis: 'x' + ax, yaxis: 'y' + ax,
      mode: 'lines', line: {color: '#1f77b4', width: 1},
      name: 'Original', hoverinfo: 'name',
    });
    traces.push({
      x: x1000, y: item.recon,
      xaxis: 'x' + ax, yaxis: 'y' + ax,
      mode: 'lines', line: {color: '#e94560', width: 1, dash: 'dash'},
      name: 'Recon', hoverinfo: 'name',
    });
    layout['xaxis' + ax] = {visible: false};
    layout['yaxis' + ax] = {visible: false};
    layout['annotations'] = layout['annotations'] || [];
    const col = ci % nc;
    const row = Math.floor(ci / nc);
    layout['annotations'].push({
      text: item.label,
      xref: 'paper', yref: 'paper',
      x: (col + 0.5) / nc,
      y: 1 - row / nr,
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: {size: 9, color: '#555'},
    });
  });

  Plotly.newPlot('plot-recon', traces, layout, {responsive: true});
}

// ── Assignments table ─────────────────────────────────────────────────────────
async function reloadTable() {
  const k = document.getElementById('tbl-k').value;
  const d = await fetch('/api/assignments?k=' + k).then(r => r.json());
  allRows = d.rows;
  currentLabels = [...new Set(allRows.map(r => r.cluster))].sort((a,b)=>a-b);

  const sel = document.getElementById('tbl-cluster');
  sel.innerHTML = '<option value="">All clusters</option>' +
    currentLabels.map(c => `<option value="${c}">Cluster ${c}</option>`).join('');

  filterTable();
}

function filterTable() {
  const search = document.getElementById('tbl-search').value.toLowerCase();
  const clusterFilter = document.getElementById('tbl-cluster').value;
  const k = parseInt(document.getElementById('tbl-k').value);

  const filtered = allRows.filter(r => {
    const text = (r.prefix + ' ' + r.well).toLowerCase();
    const matchText = !search || text.includes(search);
    const matchCluster = !clusterFilter || r.cluster === parseInt(clusterFilter);
    return matchText && matchCluster;
  });

  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = filtered.slice(0, 500).map(r => {
    const col = CLUSTER_COLOURS[r.cluster % CLUSTER_COLOURS.length];
    return `<tr>
      <td>${r.prefix}</td>
      <td>${r.well}</td>
      <td><span class="cluster-pill" style="background:${col}">Cluster ${r.cluster}</span></td>
    </tr>`;
  }).join('');

  const total = allRows.length;
  const shown = Math.min(filtered.length, 500);
  document.getElementById('tbl-count').textContent =
    `Showing ${shown} of ${filtered.length} wells (${total} total)`;
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  await loadOverview();
  loadClusters(bestK);
  loadPca(bestK);
  loadRecon();
  reloadTable();
  document.getElementById('loading').style.opacity = '0';
}

init();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(
        HTML, best_k=BEST_K, ks=KS, k_min=K_MIN, k_max=K_MAX, total_wells=len(METADATA)
    )


if __name__ == "__main__":
    print(f"\nDashboard running at http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
