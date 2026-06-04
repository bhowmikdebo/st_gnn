# Production Deployment Guide

This project is a Streamlit dashboard backed by a PyTorch ST-GNN model. The app requires these generated artifacts at runtime:

- `st_gnn_waste/data/processed/adjacency.npy`
- `st_gnn_waste/data/processed/scaler.pkl`
- `st_gnn_waste/models/saved/best_stgnn.pt`

## 1. Prepare Locally

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python st_gnn_waste/data/preprocess.py
python st_gnn_waste/train.py
streamlit run st_gnn_waste/app.py
```

Open `http://localhost:8501` and run one prediction from the dashboard.

## 2. Commit Required Runtime Files

For this small project, commit the model artifacts with the app:

```bash
git add requirements.txt Dockerfile Procfile .streamlit/config.toml DEPLOYMENT.md
git add st_gnn_waste/data/processed/adjacency.npy
git add st_gnn_waste/data/processed/scaler.pkl
git add st_gnn_waste/models/saved/best_stgnn.pt
git commit -m "prepare app for production deployment"
```

If the checkpoint becomes large later, use Git LFS or download the model from object storage during deployment.

## 3. Recommended Deployment: Docker

Build and run:

```bash
docker build -t st-gnn-waste .
docker run -p 8501:8501 st-gnn-waste
```

Then open `http://localhost:8501`.

Deploy the same Docker image to Render, Fly.io, Railway, AWS ECS, GCP Cloud Run, or Azure Container Apps.

## 4. Streamlit Community Cloud

Use:

- Main file path: `st_gnn_waste/app.py`
- Python dependencies: root `requirements.txt`
- Runtime artifacts: commit `best_stgnn.pt`, `adjacency.npy`, and `scaler.pkl`

Streamlit Cloud will run:

```bash
streamlit run st_gnn_waste/app.py
```

## 5. Render Without Docker

Use these settings:

- Build command: `pip install -r requirements.txt && python st_gnn_waste/data/preprocess.py && python st_gnn_waste/train.py`
- Start command: `streamlit run st_gnn_waste/app.py --server.port=$PORT --server.address=0.0.0.0`

For faster deploys, commit the trained artifacts and remove the training commands from the build command.

## 6. Production Checklist

- Pin dependencies in `requirements.txt`.
- Keep model/data paths relative to the package, not the shell working directory.
- Generate or ship `best_stgnn.pt`, `adjacency.npy`, and `scaler.pkl`.
- Run the app in CPU mode unless you deploy to a GPU host.
- Use Docker for reproducible builds.
- Add monitoring for app health at `/_stcore/health`.
- Retrain the model through CI or a scheduled job when the dataset changes.

## 7. Model Quality Note

The app can be deployed once the artifacts are present, but deployment readiness is not the same as model readiness. After training, compare `st_gnn_waste/models/saved/metrics.json` against the moving-average and random-forest baselines. If ST-GNN underperforms the baselines, improve the dataset, features, split strategy, or model tuning before using predictions operationally.
