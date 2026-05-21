from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
	source_file: Path
	output_file: Path
	dendrogram_file: Path
	n_clusters: int
	silhouette: float | None


def choose_cluster_count(data: pd.DataFrame, max_clusters: int = 10) -> tuple[int, float | None]:
	sample_count = len(data)
	if sample_count < 3:
		return 1, None

	upper_bound = min(max_clusters, sample_count - 1)
	best_clusters = 2
	best_score = -1.0

	for n_clusters in range(2, upper_bound + 1):
		model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
		labels = model.fit_predict(data)

		if len(set(labels)) < 2:
			continue

		score = silhouette_score(data, labels)
		if score > best_score:
			best_score = score
			best_clusters = n_clusters

	if best_score < 0:
		return 1, None

	return best_clusters, best_score


def cluster_single_csv(csv_path: Path, output_dir: Path, plots_dir: Path) -> ClusteringResult:
	df = pd.read_csv(csv_path)
	numeric_df = df.select_dtypes(include="number").dropna(axis=0).copy()

	if numeric_df.empty:
		raise ValueError(f"{csv_path.name} has no valid numeric rows for clustering.")

	n_clusters, silhouette = choose_cluster_count(numeric_df)

	if n_clusters == 1:
		labels = [0] * len(numeric_df)
	else:
		model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
		labels = model.fit_predict(numeric_df)

	clustered = numeric_df.copy()
	clustered["cluster"] = labels

	output_file = output_dir / f"{csv_path.stem}_hierarchical_clusters.csv"
	clustered.to_csv(output_file, index=False)

	link_matrix = linkage(numeric_df, method="ward", metric="euclidean")
	plt.figure(figsize=(12, 6))
	dendrogram(link_matrix, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10)
	plt.title(f"Hierarchical Dendrogram - {csv_path.name}")
	plt.xlabel("Clustered Samples")
	plt.ylabel("Distance")
	plt.tight_layout()

	dendrogram_file = plots_dir / f"{csv_path.stem}_dendrogram.png"
	plt.savefig(dendrogram_file, dpi=150)
	plt.close()

	return ClusteringResult(
		source_file=csv_path,
		output_file=output_file,
		dendrogram_file=dendrogram_file,
		n_clusters=n_clusters,
		silhouette=silhouette,
	)


def run_hierarchical_clustering() -> list[ClusteringResult]:
	project_root = Path(__file__).resolve().parents[1]
	data_dir = project_root / "data"
	output_dir = data_dir
	plots_dir = project_root / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	csv_files = sorted(data_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError("No CSV files found in the data directory.")

	results: list[ClusteringResult] = []
	for csv_path in csv_files:
		results.append(cluster_single_csv(csv_path, output_dir, plots_dir))

	return results


if __name__ == "__main__":
	all_results = run_hierarchical_clustering()
	for result in all_results:
		sil_txt = f"{result.silhouette:.4f}" if result.silhouette is not None else "N/A"
		print(
			f"Processed {result.source_file.name}: "
			f"clusters={result.n_clusters}, silhouette={sil_txt}, "
			f"output={result.output_file.name}, dendrogram={result.dendrogram_file.name}"
		)
