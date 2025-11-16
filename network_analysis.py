import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Load data
RETURNS_FILE = "nifty100_logreturns.csv"
logret = pd.read_csv(RETURNS_FILE, index_col=0)
stocks = logret.columns.tolist()
corr = logret.corr()

# Helper: Build graph for threshold
def build_graph(theta):
    G = nx.Graph()
    G.add_nodes_from(stocks)

    for i in stocks:
        for j in stocks:
            if i != j and corr.loc[i, j] >= theta:
                G.add_edge(i, j, weight=float(corr.loc[i, j]))

    # remove isolated nodes
    Gcc = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()
    return G, Gcc

# Helper: Plot and save graph
def save_graph_plot(G, folder, theta):
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=80, with_labels=True, font_size=6, edge_color='gray')
    plt.title(f"Network (theta = {theta})")
    plt.savefig(f"{folder}/graph.png", dpi=300, bbox_inches='tight')
    plt.close()

# Helper: Analyze network
def analyze_network(G, folder, theta):

    # --- Centralities ---
    degree_c = nx.degree_centrality(G)
    between_c = nx.betweenness_centrality(G, normalized=True)
    close_c = nx.closeness_centrality(G)

    df_cent = pd.DataFrame({
        "degree_centrality": degree_c,
        "betweenness": between_c,
        "closeness": close_c
    })
    df_cent.to_csv(f"{folder}/centrality.csv")

    #  Clustering coefficients 
    clustering = nx.clustering(G)
    pd.DataFrame({"clustering": clustering}).to_csv(f"{folder}/clustering.csv")

    #  Similarity scores (Jaccard) 
    jaccard = list(nx.jaccard_coefficient(G))
    df_jaccard = pd.DataFrame(jaccard, columns=["node1", "node2", "jaccard"])
    df_jaccard.to_csv(f"{folder}/similarity.csv", index=False)

    #  Degree assortativity 
    assort = nx.degree_assortativity_coefficient(G)
    with open(f"{folder}/assortativity.txt", "w") as f:
        f.write(f"Degree assortativity (theta={theta}) = {assort}")

    #  Degree distribution 
    degrees = [G.degree(n) for n in G.nodes()]
    df_deg = pd.DataFrame({"degree": degrees})
    df_deg.to_csv(f"{folder}/degree_distribution.csv", index=False)

    # Plot degree histogram 
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=20)
    plt.title(f"Degree Distribution (theta={theta})")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(f"{folder}/degree_distribution.png", dpi=300)
    plt.close()

# RUN EXPERIMENTS
thetas = [0.3, 0.4, 0.5, 0.6, 0.7]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\nRunning network analysis for thresholds:", thetas)

for theta in thetas:
    print(f"\n=== Processing theta = {theta} ===")
    
    folder = f"{RESULTS_DIR}/theta_{theta}"
    os.makedirs(folder, exist_ok=True)

    # Build graph
    G, Gcc = build_graph(theta)

    # Save graph visualization
    save_graph_plot(Gcc, folder, theta)

    # Compute stats
    analyze_network(Gcc, folder, theta)

print("\nAll results saved inside /results/")
