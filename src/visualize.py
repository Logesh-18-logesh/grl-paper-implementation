# visualize.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

def reduce_and_plot(X, y, method, title, filename):
    """
    Reduce dimensionality and save visualization to the 'output/' folder.
    """
    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(X)
    else:
        reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")

def visualize_embeddings_comparison(base_path, final_path, label_path):
    """
    Visualize and save PCA and t-SNE for base and final embeddings.
    """
    print("\n📊 Loading embeddings and labels...")
    base_df = pd.read_csv(base_path)
    final_df = pd.read_csv(final_path)
    label_df = pd.read_csv(label_path)

    base_merged = pd.merge(base_df, label_df, on='id')
    final_merged = pd.merge(final_df, label_df, on='id')

    X_base = base_merged.drop(['id', 'target'], axis=1).values
    X_final = final_merged.drop(['id', 'target'], axis=1).values
    y = base_merged['target'].values  # same for both

    # Normalize
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_base)
    X_final = scaler.fit_transform(X_final)

    print("📈 Running PCA and t-SNE on base and final embeddings...")

    # PCA
    reduce_and_plot(X_base, y, 'pca', "PCA - Base Embeddings", "base_pca.png")
    reduce_and_plot(X_final, y, 'pca', "PCA - Final Embeddings", "final_pca.png")

    # t-SNE
    reduce_and_plot(X_base, y, 'tsne', "t-SNE - Base Embeddings", "base_tsne.png")
    reduce_and_plot(X_final, y, 'tsne', "t-SNE - Final Embeddings", "final_tsne.png")
