# order-matters-randomness-too

## Data

`data/` and `results/` are not tracked in this repo (see `.gitignore`) — they're regenerated locally by running the experiment scripts, which fetch/build the datasets on demand (e.g. via `sklearn.datasets.fetch_rcv1`, `load_svmlight_file`, etc.) and write results back into `results/`.
