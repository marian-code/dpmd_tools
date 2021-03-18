from datetime import date
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

WD = Path.cwd()


used = np.loadtxt(WD / "used.raw")
clusters = np.loadtxt(WD / "clusters.raw")
print(used.shape)

fig = go.Figure()
for i in range(used.shape[1]):
    selection = np.argwhere(np.array(~(used[:, :i] == 0).all(axis=1), dtype=int)).flatten()
    unique, counts = np.unique(clusters[selection], return_counts=True)

    fig.add_trace(go.Bar(x=unique, y=counts, name=f"iteration {i}"))


unique, counts = np.unique(clusters, return_counts=True)
clst = zip(unique.astype(int), counts.astype(int))
fig.update_layout(
    title=f"cluster labels and counts: {', '.join([f'c{u}:{c}' for u, c in clst])}",
    barmode='group')
fig.write_html("clusters_distrib.html", include_plotlyjs="cdn")