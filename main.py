import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
from collections import Counter

# CONFIG
RETURNS_FILE = "nifty100_logreturns.csv"
THETA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
SEED = 42
SECTOR_FILE = "ind_nifty100list.csv"

# Color scheme 
COLOR_PALETTE = {
    'Financial Services': '#1f77b4',      # Blue
    'Information Technology': '#ff7f0e',  # Orange
    'Healthcare': '#2ca02c',              # Green
    'Oil Gas & Consumable Fuels': '#d62728', # Red
    'Automobile and Auto Components': '#9467bd', # Purple
    'Fast Moving Consumer Goods': '#8c564b', # Brown
    'Power': '#e377c2',                   # Pink
    'Metals & Mining': '#7f7f7f',         # Gray
    'Construction Materials': '#bcbd22',  # Yellow-green
    'Capital Goods': '#17becf',           # Cyan
    'Consumer Durables': '#ff9896',       # Light red
    'Services': '#c5b0d5',                # Light purple
    'Realty': '#c49c94',                  # Light brown
    'Telecommunication': '#f7b6d2',       # Light pink
    'Chemicals': '#dbdb8d'                # Light yellow
}

# LOAD SECTOR INFORMATION
def load_sector_info():
    sector_df = pd.read_csv(SECTOR_FILE)
    return dict(zip(sector_df['Symbol'], sector_df['Industry']))

# NETWORK PLOT
def plot_network(G, theta, outdir, sectors):
    # Remove isolated nodes for clean visualization
    G2 = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()
    if G2.number_of_nodes() == 0:
        print(f"No connected nodes to plot for θ = {theta}")
        return

    # layout
    pos = nx.spring_layout(
        G2,
        seed=SEED,
        k=2.5,              
        iterations=100,      
        scale=2,            
        weight='weight'     
    )

    # Calculate metrics for visualization
    degrees = dict(G2.degree())
    betweenness = nx.betweenness_centrality(G2)
    
    # Node sizes based on degree (log scale for better visualization)
    node_sizes = [np.log(degrees[n] + 1) * 150 + 50 for n in G2.nodes()]
    
    # Node colors based on sector
    node_colors = [COLOR_PALETTE.get(sectors.get(n, 'Other'), '#cccccc') for n in G2.nodes()]
    
    # Edge properties
    edge_weights = [G2[u][v]['weight'] * 2 for u, v in G2.edges()]
    edge_alphas = [0.3 + (G2[u][v]['weight'] - theta) / (1 - theta) * 0.4 for u, v in G2.edges()]

    plt.figure(figsize=(16, 12))
    
    # Draw edges with varying transparency based on correlation strength
    edges = nx.draw_networkx_edges(
        G2,
        pos,
        width=edge_weights,
        alpha=edge_alphas,
        edge_color='gray',
        style='solid'
    )

    # Draw nodes with sector-based coloring
    nodes = nx.draw_networkx_nodes(
        G2,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=0.8,
        alpha=0.9
    )

    # Label only high-degree nodes to reduce clutter 
    high_degree_nodes = [n for n in G2.nodes() if degrees[n] >= np.percentile(list(degrees.values()), 75)]
    
    for node in high_degree_nodes:
        x, y = pos[node]
        plt.text(
            x,
            y + 0.05,
            node,
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='none')
        )

    # Add legend for sectors
    legend_elements = []
    sector_counts = Counter([sectors.get(n, 'Other') for n in G2.nodes()])
    for sector, color in COLOR_PALETTE.items():
        if sector_counts.get(sector, 0) > 0:
            legend_elements.append(
                Patch(facecolor=color, edgecolor='black', label=f'{sector} ({sector_counts[sector]})')
            )
    
    if legend_elements:
        plt.legend(handles=legend_elements, 
                  loc='upper left', 
                  bbox_to_anchor=(1.05, 1),
                  fontsize=9,
                  title="Sectors")

    plt.title(f"Stock Correlation Network (θ = {theta})\n"
              f"N={G2.number_of_nodes()}, E={G2.number_of_edges()}, "
              f"Density={nx.density(G2):.4f}",
              fontsize=14, pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{outdir}/network_theta_{theta}.png", 
                dpi=350, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

# ENHANCED DEGREE DISTRIBUTION
def plot_degree_distribution(G, theta, outdir):
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return

    plt.figure(figsize=(10, 6))
    
    # Log-log plot for power-law analysis 
    plt.subplot(1, 2, 1)
    degree_counts = Counter(degrees)
    x = list(degree_counts.keys())
    y = list(degree_counts.values())
    
    plt.scatter(x, y, alpha=0.7, s=50, color='steelblue')
    plt.xlabel('Degree (k)')
    plt.ylabel('Count P(k)')
    plt.title(f'Degree Distribution\n(θ = {theta})')
    plt.grid(True, alpha=0.3)
    
    # Log-log scale for power law inspection
    plt.subplot(1, 2, 2)
    plt.loglog(x, y, 'o', alpha=0.7, markersize=6)
    plt.xlabel('Degree (k) - log scale')
    plt.ylabel('Count P(k) - log scale')
    plt.title('Log-Log Degree Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/degree_distribution_theta_{theta}.png", dpi=300)
    plt.close()

# CENTRALITY COMPARISON PLOTS
def plot_centrality_comparison(all_metrics, outdir):
    """Compare centrality measures across different theta values"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    centrality_measures = ['DegreeCentrality', 'BetweennessCentrality', 
                          'ClosenessCentrality', 'ClusteringCoefficient']
    
    for idx, measure in enumerate(centrality_measures):
        ax = axes[idx]
        for theta, metrics_df in all_metrics.items():
            if measure in metrics_df.columns:
                values = metrics_df[measure].dropna()
                if len(values) > 0:
                    ax.hist(values, alpha=0.6, label=f'θ={theta}', bins=20, density=True)
        
        ax.set_xlabel(measure)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {measure}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/centrality_comparison.png", dpi=300)
    plt.close()

# NETWORK METRICS TREND PLOT
def plot_network_metrics_trend(metrics_over_theta, outdir):
    """Plot how network metrics change with theta"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('num_nodes', 'Number of Nodes'),
        ('num_edges', 'Number of Edges'), 
        ('density', 'Network Density'),
        ('avg_clustering', 'Average Clustering'),
        ('avg_degree', 'Average Degree'),
        ('assortativity', 'Assortativity')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            thetas = []
            values = []
            
            for theta, metrics in metrics_over_theta.items():
                if metric in metrics:
                    thetas.append(theta)
                    values.append(metrics[metric])
            
            if thetas and values:
                ax.plot(thetas, values, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Threshold (θ)')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs θ')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/network_metrics_trend.png", dpi=300)
    plt.close()

# SECTOR ANALYSIS PLOTS
def plot_sector_analysis(G, theta, sectors, outdir):
    """Analyze network properties by sector"""
    sector_degrees = {}
    sector_betweenness = {}
    betweenness = nx.betweenness_centrality(G)
    
    for node in G.nodes():
        sector = sectors.get(node, 'Other')
        if sector not in sector_degrees:
            sector_degrees[sector] = []
            sector_betweenness[sector] = []
        
        sector_degrees[sector].append(G.degree(node))
        sector_betweenness[sector].append(betweenness.get(node, 0))
    
    # Plot sector-wise average degree
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    avg_degrees = {s: np.mean(degs) for s, degs in sector_degrees.items() if degs}
    sectors_sorted = sorted(avg_degrees.keys(), key=lambda x: avg_degrees[x], reverse=True)
    
    colors = [COLOR_PALETTE.get(s, '#cccccc') for s in sectors_sorted]
    bars = ax1.bar(range(len(sectors_sorted)), [avg_degrees[s] for s in sectors_sorted], 
                   color=colors, alpha=0.8)
    
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Average Degree')
    ax1.set_title(f'Sector-wise Average Degree (θ = {theta})')
    ax1.set_xticks(range(len(sectors_sorted)))
    ax1.set_xticklabels(sectors_sorted, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot sector-wise average betweenness
    avg_betweenness = {s: np.mean(bet) for s, bet in sector_betweenness.items() if bet}
    sectors_sorted_bet = sorted(avg_betweenness.keys(), key=lambda x: avg_betweenness[x], reverse=True)[:10]  # Top 10
    
    colors_bet = [COLOR_PALETTE.get(s, '#cccccc') for s in sectors_sorted_bet]
    bars_bet = ax2.bar(range(len(sectors_sorted_bet)), [avg_betweenness[s] for s in sectors_sorted_bet], 
                      color=colors_bet, alpha=0.8)
    
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Average Betweenness Centrality')
    ax2.set_title(f'Sector-wise Betweenness Centrality (θ = {theta})')
    ax2.set_xticks(range(len(sectors_sorted_bet)))
    ax2.set_xticklabels(sectors_sorted_bet, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/sector_analysis_theta_{theta}.png", dpi=300)
    plt.close()

# ENHANCED METRICS COMPUTATION
def compute_and_save_metrics(G, theta, outdir, sectors):
    # Centrality measures
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    clo_cent = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    # Similarity scores (Jaccard)
    jaccard = list(nx.jaccard_coefficient(G))

    # Network-level metrics
    network_metrics = {
        'theta': theta,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
        'avg_clustering': nx.average_clustering(G),
        'assortativity': assortativity,
        'connected_components': nx.number_connected_components(G)
    }

    df_cent = pd.DataFrame({
        "DegreeCentrality": deg_cent,
        "BetweennessCentrality": bet_cent,
        "ClosenessCentrality": clo_cent,
        "ClusteringCoefficient": clustering
    })

    # Add sector information
    df_cent['Sector'] = df_cent.index.map(sectors)
    df_cent.to_csv(f"{outdir}/centrality_clustering_theta_{theta}.csv")

    # Jaccard similarity
    sim_df = pd.DataFrame(jaccard, columns=["Node1", "Node2", "Jaccard"])
    sim_df.to_csv(f"{outdir}/similarity_theta_{theta}.csv", index=False)

    # Save network metrics
    pd.DataFrame([network_metrics]).to_csv(f"{outdir}/network_metrics_theta_{theta}.csv", index=False)
    
    # Assortativity
    with open(f"{outdir}/assortativity_theta_{theta}.txt", "w") as f:
        f.write(f"Degree assortativity (theta = {theta}): {assortativity}\n")
    
    return network_metrics, df_cent

# UPDATED MAIN LOOP
def main():
    logret = load_log_returns()
    corr = logret.corr()
    sectors = load_sector_info()
    
    all_metrics = {}
    all_centrality_data = {}

    for theta in THETA_VALUES:
        outdir = f"results/theta_{theta}"
        os.makedirs(outdir, exist_ok=True)

        print(f"\n====== Building network for theta = {theta} ======")

        G = build_network(corr, theta)

        # Save statistics
        network_metrics, centrality_data = compute_and_save_metrics(G, theta, outdir, sectors)
        all_metrics[theta] = network_metrics
        all_centrality_data[theta] = centrality_data

        # Enhanced plots
        plot_network(G, theta, outdir, sectors)
        plot_degree_distribution(G, theta, outdir)
        plot_sector_analysis(G, theta, sectors, outdir)

        print(f"Saved results in: {outdir}")

    # Comparative plots across all theta values
    overall_outdir = "results/comparative_analysis"
    os.makedirs(overall_outdir, exist_ok=True)
    
    plot_centrality_comparison(all_centrality_data, overall_outdir)
    plot_network_metrics_trend(all_metrics, overall_outdir)

    print(f"\nComparative analysis saved in: {overall_outdir}")

# Keep your existing functions (load_log_returns, build_network) unchanged
def load_log_returns():
    df = pd.read_csv(RETURNS_FILE, index_col=0)
    return df

def build_network(corr, theta):
    G = nx.Graph()
    stocks = corr.columns.tolist()
    G.add_nodes_from(stocks)

    for i in stocks:
        for j in stocks:
            if i != j and corr.loc[i, j] >= theta:
                G.add_edge(i, j, weight=float(corr.loc[i, j]))

    return G

if __name__ == "__main__":
    main()