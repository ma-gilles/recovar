import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def plot_junk_detection_results(zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, 
                               particle_usage, output_folder, zdim_key):
    """
    Create plots for junk particle detection results with improved styling, clarity, and UMAP visualization.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up seaborn styling for better-looking plots
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Extract FSC scores
    halfmap_fscs = [fsc_scores[i]['halfmap_fsc'] for i in range(len(cluster_centers))]
    vs_mean_fscs = [fsc_scores[i]['vs_mean_fsc'] for i in range(len(cluster_centers))]
    halfmap_aucs = [fsc_auc_scores[i]['halfmap_auc'] for i in range(len(cluster_centers))]
    vs_mean_aucs = [fsc_auc_scores[i]['vs_mean_auc'] for i in range(len(cluster_centers))]

    # --- Save all FSC curves ---
    all_fsc_curves = [fsc_scores[i]['halfmap_curve'] for i in range(len(cluster_centers))]
    all_vs_mean_curves = [fsc_scores[i]['vs_mean_curve'] for i in range(len(cluster_centers))]
    
    with open(os.path.join(output_folder, f'all_fsc_curves_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(all_fsc_curves, f)
    with open(os.path.join(output_folder, f'all_vs_mean_fsc_curves_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(all_vs_mean_curves, f)

    # --- Compute UMAP embedding ---
    logger.info("Computing UMAP embedding...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_embedding = reducer.fit_transform(zs)
        # Also transform cluster centers to UMAP space
        umap_cluster_centers = reducer.transform(cluster_centers)
        logger.info("UMAP embedding computed successfully")
    except Exception as e:
        logger.warning(f"UMAP computation failed: {e}. Using first two dimensions of latent space.")
        umap_embedding = zs[:, :2]
        umap_cluster_centers = cluster_centers[:, :2]

    # --- Ensure UMAP and cluster centers are finite ---
    umap_embedding = np.asarray(umap_embedding)
    umap_cluster_centers = np.asarray(umap_cluster_centers)
    if not (np.isfinite(umap_embedding).all() and np.isfinite(umap_cluster_centers).all()):
        logger.warning("Non-finite values detected in UMAP embedding or cluster centers. Filtering...")
        mask = np.isfinite(umap_embedding).all(axis=1)
        umap_embedding = umap_embedding[mask]
        cluster_indices = np.asarray(cluster_indices)[mask]
        # cluster_centers should be finite by construction, but check anyway
        umap_cluster_centers = umap_cluster_centers[np.isfinite(umap_cluster_centers).all(axis=1)]

    # --- Ensure all index arrays are integer type ---
    cluster_indices = np.asarray(cluster_indices, dtype=np.intp)

    # --- Plot all half-map FSC curves with improved styling ---
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(all_fsc_curves[0]))
    
    # Create frequency axis (assuming Nyquist frequency)
    freq_axis = x / (2 * len(all_fsc_curves[0]))
    
    # Color code curves by FSC score with better transparency
    colors = plt.cm.viridis(np.array(halfmap_fscs))
    for i, curve in enumerate(all_fsc_curves):
        ax.plot(freq_axis, curve, color=colors[i], alpha=0.3, linewidth=0.6)
    
    # Mean and IQR with better styling
    all_fsc_array = np.array(all_fsc_curves)
    mean_curve = np.mean(all_fsc_array, axis=0)
    q25 = np.percentile(all_fsc_array, 25, axis=0)
    q75 = np.percentile(all_fsc_array, 75, axis=0)
    ax.plot(freq_axis, mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25, q75, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('Fourier Shell Correlation')
    ax.set_title(f'Half-map FSC Curves for All Clusters (n={len(cluster_centers)})')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(halfmap_fscs), vmax=max(halfmap_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Half-map FSC Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'all_halfmap_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot all vs-mean FSC curves with improved styling ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color code curves by vs-mean FSC score
    colors = plt.cm.viridis(np.array(vs_mean_fscs))
    for i, curve in enumerate(all_vs_mean_curves):
        ax.plot(freq_axis, curve, color=colors[i], alpha=0.3, linewidth=0.6)
    
    # Mean and IQR
    all_vs_mean_array = np.array(all_vs_mean_curves)
    mean_vs_mean_curve = np.mean(all_vs_mean_array, axis=0)
    q25_vs_mean = np.percentile(all_vs_mean_array, 25, axis=0)
    q75_vs_mean = np.percentile(all_vs_mean_array, 75, axis=0)
    ax.plot(freq_axis, mean_vs_mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25_vs_mean, q75_vs_mean, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('Fourier Shell Correlation')
    ax.set_title(f'vs-Mean FSC Curves for All Clusters (n={len(cluster_centers)})')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(vs_mean_fscs), vmax=max(vs_mean_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('vs-Mean FSC Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'all_vs_mean_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Individual cluster FSC plots (top 10 and bottom 10) with better layout ---
    sorted_indices = np.argsort(halfmap_fscs)
    top_10 = sorted_indices[-10:]
    bottom_10 = sorted_indices[:10]
    
    # Plot top 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Top 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98)
    
    for i, cluster_idx in enumerate(top_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {len(sorted_indices)-i})', fontsize=10)
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'top_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot bottom 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Bottom 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98)
    
    for i, cluster_idx in enumerate(bottom_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {i+1})', fontsize=10)
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'bottom_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Create main summary plot with UMAP and hexbin visualizations ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Junk Particle Detection Summary (zdim={zdim_key}, n_clusters={len(cluster_centers)})', 
                 fontsize=20, y=0.95)

    # 1. UMAP colored by cluster
    ax = axes[0, 0]
    # Use a color palette that works well for many categories
    n_clusters = len(cluster_centers)
    colors_cluster = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    if n_clusters > 20:
        # Repeat colors if we have more clusters
        colors_cluster = np.tile(colors_cluster, (int(np.ceil(n_clusters/20)), 1))[:n_clusters]
    
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=cluster_indices, 
                        cmap=matplotlib.colors.ListedColormap(colors_cluster), 
                        s=0.5, alpha=0.7, rasterized=True)
    ax.scatter(umap_cluster_centers[:, 0], umap_cluster_centers[:, 1], c='red', marker='x', s=100, 
               linewidth=2, label='Cluster Centers', zorder=10)
    ax.set_title('UMAP Embedding by Cluster ID')
    ax.set_xlabel('UMAP₁')
    ax.set_ylabel('UMAP₂')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. UMAP colored by Half-map FSC AUC (hexbin)
    ax = axes[0, 1]
    particle_halfmap_aucs = np.array([halfmap_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], C=particle_halfmap_aucs, gridsize=50, 
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean Half-map FSC AUC')
    ax.set_title('UMAP Embedding by Half-map FSC AUC')
    ax.set_xlabel('UMAP₁')
    ax.set_ylabel('UMAP₂')
    ax.grid(True, alpha=0.3)

    # 3. UMAP colored by FSC vs Mean AUC (hexbin)
    ax = axes[0, 2]
    particle_vs_mean_aucs = np.array([vs_mean_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], C=particle_vs_mean_aucs, gridsize=50, 
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean FSC vs Mean AUC')
    ax.set_title('UMAP Embedding by FSC vs Mean AUC')
    ax.set_xlabel('UMAP₁')
    ax.set_ylabel('UMAP₂')
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram of particle counts per cluster
    ax = axes[1, 0]
    cluster_counts = np.bincount(cluster_indices)
    sns.histplot(data=cluster_counts, bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(cluster_counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cluster_counts):.1f}')
    ax.set_title('Particle Counts per Cluster')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Number of Clusters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. FSC Score Distribution
    ax = axes[1, 1]
    sns.histplot(data=halfmap_fscs, bins=20, ax=ax, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. FSC vs Mean FSC comparison
    ax = axes[1, 2]
    scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC')
    ax.set_ylabel('vs-Mean FSC')
    ax.set_title('Half-map FSC vs vs-Mean FSC')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Particle Usage Visualization (simplified and cleaner) ---
    if zs.shape[1] >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Particle Usage Analysis', fontsize=18, y=0.95)
        
        # Plot 1: UMAP particle density
        ax = axes[0, 0]
        hb = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], gridsize=60, cmap='Blues', mincnt=1)
        ax.scatter(umap_cluster_centers[:, 0], umap_cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Particle Density')
        ax.set_title('Particle Density in UMAP Space')
        ax.set_xlabel('UMAP₁')
        ax.set_ylabel('UMAP₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Particles used for FSC calculation (colored by cluster FSC score)
        ax = axes[0, 1]
        all_used_particles = []
        for cluster_idx in range(len(cluster_centers)):
            all_used_particles.extend(particle_usage[cluster_idx]['all_particles'])
        all_used_particles = np.array(all_used_particles)
        
        if len(all_used_particles) > 0:
            used_particle_fsc_scores = []
            for particle_idx in all_used_particles:
                cluster_idx = cluster_indices[particle_idx]
                used_particle_fsc_scores.append(halfmap_fscs[cluster_idx])
            
            hb = ax.hexbin(umap_embedding[all_used_particles, 0], umap_embedding[all_used_particles, 1], 
                          C=used_particle_fsc_scores, gridsize=50, cmap='viridis', 
                          reduce_C_function=np.mean, mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label('Mean Cluster FSC Score')
        
        ax.scatter(umap_cluster_centers[:, 0], umap_cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        ax.set_title('Particles Used for FSC Calculation\n(colored by cluster FSC score)')
        ax.set_xlabel('UMAP₁')
        ax.set_ylabel('UMAP₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Particle usage frequency
        ax = axes[0, 2]
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        hb = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], C=usage_counts, gridsize=60, 
                      cmap='plasma', reduce_C_function=np.mean, mincnt=1)
        ax.scatter(umap_cluster_centers[:, 0], umap_cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Usage Count')
        ax.set_title('Particle Usage Frequency\n(how many clusters use each particle)')
        ax.set_xlabel('UMAP₁')
        ax.set_ylabel('UMAP₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Top 5 clusters - particles used for FSC
        ax = axes[1, 0]
        top_5 = sorted_indices[-5:]
        colors_top5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(top_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(umap_embedding[used_particles, 0], umap_embedding[used_particles, 1], 
                      c=[colors_top5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(umap_cluster_centers[top_5, 0], umap_cluster_centers[top_5, 1], 
                  c=colors_top5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Top 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('UMAP₁')
        ax.set_ylabel('UMAP₂')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Bottom 5 clusters - particles used for FSC
        ax = axes[1, 1]
        bottom_5 = sorted_indices[:5]
        colors_bottom5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(bottom_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(umap_embedding[used_particles, 0], umap_embedding[used_particles, 1], 
                      c=[colors_bottom5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(umap_cluster_centers[bottom_5, 0], umap_cluster_centers[bottom_5, 1], 
                  c=colors_bottom5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Bottom 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('UMAP₁')
        ax.set_ylabel('UMAP₂')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Usage statistics
        ax = axes[1, 2]
        
        # Calculate usage statistics
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        # Create histogram of usage counts
        unique_counts, count_frequencies = np.unique(usage_counts, return_counts=True)
        
        bars = ax.bar(unique_counts, count_frequencies, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Clusters Using Particle')
        ax.set_ylabel('Number of Particles')
        ax.set_title('Particle Usage Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_usage = np.mean(usage_counts)
        median_usage = np.median(usage_counts)
        unused_particles = np.sum(usage_counts == 0)
        total_particles = len(usage_counts)
        
        stats_text = f'Mean usage: {mean_usage:.1f}\nMedian usage: {median_usage:.1f}\nUnused particles: {unused_particles}\n({unused_particles/total_particles*100:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(os.path.join(output_folder, f'particle_usage_visualization_{zdim_key}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # --- Create comprehensive analysis plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FSC Score Analysis', fontsize=16, y=0.95)
    
    # Plot 1: Half-map FSC histogram
    ax = axes[0, 0]
    sns.histplot(data=halfmap_fscs, bins=20, ax=ax, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: vs-Mean FSC histogram
    ax = axes[0, 1]
    sns.histplot(data=vs_mean_fscs, bins=20, ax=ax, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(vs_mean_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(vs_mean_fscs):.3f}')
    ax.set_xlabel('vs-Mean FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of vs-Mean FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Half-map FSC AUC histogram
    ax = axes[0, 2]
    sns.histplot(data=halfmap_aucs, bins=20, ax=ax, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_aucs):.3f}')
    ax.set_xlabel('Half-map FSC AUC')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: vs-Mean FSC AUC histogram
    ax = axes[1, 0]
    sns.histplot(data=vs_mean_aucs, bins=20, ax=ax, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(vs_mean_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(vs_mean_aucs):.3f}')
    ax.set_xlabel('vs-Mean FSC AUC')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of vs-Mean FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Half-map FSC vs Half-map FSC AUC scatter
    ax = axes[1, 1]
    scatter = ax.scatter(halfmap_fscs, halfmap_aucs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Half-map FSC AUC')
    ax.set_title('Half-map FSC Score vs Half-map FSC AUC')
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Half-map FSC vs vs-Mean FSC comparison
    ax = axes[1, 2]
    scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC')
    ax.set_ylabel('vs-Mean FSC')
    ax.set_title('Half-map FSC vs vs-Mean FSC')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(output_folder, f'fsc_analysis_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate cluster sizes
    cluster_sizes = [np.sum(cluster_indices == i) for i in range(len(cluster_centers))]
    
    # Save detailed results
    results = {
        'cluster_centers': cluster_centers,
        'cluster_indices': cluster_indices,
        'fsc_scores': fsc_scores,
        'fsc_auc_scores': fsc_auc_scores,
        'particle_usage': particle_usage,
        'cluster_sizes': cluster_sizes,
        'halfmap_fscs': halfmap_fscs,
        'halfmap_aucs': halfmap_aucs,
        'vs_mean_fscs': vs_mean_fscs,
        'vs_mean_aucs': vs_mean_aucs,
        'umap_embedding': umap_embedding
    }
    
    with open(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Identify potential junk clusters using both metrics
    # Half-map FSC based outlier detection
    halfmap_fsc_threshold = np.percentile(halfmap_fscs, 25)  # Bottom 25%
    halfmap_auc_threshold = np.percentile(halfmap_aucs, 25)  # Bottom 25%
    
    # vs-Mean FSC based outlier detection
    vs_mean_fsc_threshold = np.percentile(vs_mean_fscs, 25)  # Bottom 25%
    vs_mean_auc_threshold = np.percentile(vs_mean_aucs, 25)  # Bottom 25%
    
    # Combined outlier detection (clusters that are outliers in either metric)
    combined_junk_clusters = []
    for i in range(len(cluster_centers)):
        if (halfmap_fscs[i] < halfmap_fsc_threshold or halfmap_aucs[i] < halfmap_auc_threshold or
            vs_mean_fscs[i] < vs_mean_fsc_threshold or vs_mean_aucs[i] < vs_mean_auc_threshold):
            combined_junk_clusters.append(i)
    
    # Half-map FSC only outlier detection
    halfmap_junk_clusters = []
    for i in range(len(cluster_centers)):
        if halfmap_fscs[i] < halfmap_fsc_threshold or halfmap_aucs[i] < halfmap_auc_threshold:
            halfmap_junk_clusters.append(i)
    
    # vs-Mean FSC only outlier detection
    vs_mean_junk_clusters = []
    for i in range(len(cluster_centers)):
        if vs_mean_fscs[i] < vs_mean_fsc_threshold or vs_mean_aucs[i] < vs_mean_auc_threshold:
            vs_mean_junk_clusters.append(i)
    
    # Save junk cluster indices for each method
    combined_junk_particle_indices = np.where(np.isin(cluster_indices, combined_junk_clusters))[0]
    halfmap_junk_particle_indices = np.where(np.isin(cluster_indices, halfmap_junk_clusters))[0]
    vs_mean_junk_particle_indices = np.where(np.isin(cluster_indices, vs_mean_junk_clusters))[0]
    
    with open(os.path.join(output_folder, f'combined_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(combined_junk_particle_indices, f)
    with open(os.path.join(output_folder, f'halfmap_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(halfmap_junk_particle_indices, f)
    with open(os.path.join(output_folder, f'vs_mean_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(vs_mean_junk_particle_indices, f)
    
    # Save detailed junk cluster information
    junk_info = {
        'combined_junk_clusters': combined_junk_clusters,
        'halfmap_junk_clusters': halfmap_junk_clusters,
        'vs_mean_junk_clusters': vs_mean_junk_clusters,
        'combined_junk_particle_indices': combined_junk_particle_indices,
        'halfmap_junk_particle_indices': halfmap_junk_particle_indices,
        'vs_mean_junk_particle_indices': vs_mean_junk_particle_indices,
        'halfmap_fsc_threshold': halfmap_fsc_threshold,
        'halfmap_auc_threshold': halfmap_auc_threshold,
        'vs_mean_fsc_threshold': vs_mean_fsc_threshold,
        'vs_mean_auc_threshold': vs_mean_auc_threshold,
        'total_particles': len(zs),
        'combined_junk_particles': len(combined_junk_particle_indices),
        'halfmap_junk_particles': len(halfmap_junk_particle_indices),
        'vs_mean_junk_particles': len(vs_mean_junk_particle_indices),
        'combined_junk_percentage': len(combined_junk_particle_indices) / len(zs) * 100,
        'halfmap_junk_percentage': len(halfmap_junk_particle_indices) / len(zs) * 100,
        'vs_mean_junk_percentage': len(vs_mean_junk_particle_indices) / len(zs) * 100
    }
    
    with open(os.path.join(output_folder, f'junk_cluster_info_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(junk_info, f)
    
    # Print summary statistics
    logger.info(f"=== Junk Detection Summary ===")
    logger.info(f"Total clusters: {len(cluster_centers)}")
    logger.info(f"Total particles: {len(zs)}")
    logger.info(f"")
    logger.info(f"Half-map FSC based detection:")
    logger.info(f"  - Junk clusters: {len(halfmap_junk_clusters)} ({len(halfmap_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(halfmap_junk_particle_indices)} ({len(halfmap_junk_particle_indices)/len(zs)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"vs-Mean FSC based detection:")
    logger.info(f"  - Junk clusters: {len(vs_mean_junk_clusters)} ({len(vs_mean_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(vs_mean_junk_particle_indices)} ({len(vs_mean_junk_particle_indices)/len(zs)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Combined detection:")
    logger.info(f"  - Junk clusters: {len(combined_junk_clusters)} ({len(combined_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(combined_junk_particle_indices)} ({len(combined_junk_particle_indices)/len(zs)*100:.1f}%)")
    
    return junk_info 

def create_junk_detection_visualizations(halfmap_fscs, vs_mean_fscs, halfmap_aucs, vs_mean_aucs,
                                       cluster_centers, cluster_indices, particle_usage, 
                                       umap_embedding, output_folder, zdim_key):
    """Create visualizations for junk detection results with error handling."""
    try:
        # Check if we have valid data
        if len(cluster_centers) == 0 or len(cluster_indices) == 0:
            logger.warning("No cluster data available for visualization")
            return
        
        # Ensure cluster_indices are integers
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        
        # Create a simple summary plot instead of complex visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Junk Detection Summary', fontsize=14)
        
        # Plot 1: Half-map FSC histogram
        ax = axes[0, 0]
        ax.hist(halfmap_fscs, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(halfmap_fscs):.3f}')
        ax.set_xlabel('Half-map FSC Score')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Half-map FSC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: vs-Mean FSC histogram
        ax = axes[0, 1]
        ax.hist(vs_mean_fscs, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(vs_mean_fscs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(vs_mean_fscs):.3f}')
        ax.set_xlabel('vs-Mean FSC Score')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('vs-Mean FSC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Half-map FSC AUC histogram
        ax = axes[1, 0]
        ax.hist(halfmap_aucs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(halfmap_aucs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(halfmap_aucs):.3f}')
        ax.set_xlabel('Half-map FSC AUC')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Half-map FSC AUC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: vs-Mean FSC AUC histogram
        ax = axes[1, 1]
        ax.hist(vs_mean_aucs, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(vs_mean_aucs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(vs_mean_aucs):.3f}')
        ax.set_xlabel('vs-Mean FSC AUC')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('vs-Mean FSC AUC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'junk_detection_summary_{zdim_key}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Junk detection visualizations created successfully")
        
    except Exception as e:
        logger.warning(f"Junk detection visualization failed: {str(e)}")
        logger.info("Continuing without visualizations...")

def map_clusters_to_particles(junk_clusters, cluster_indices, output_folder, zdim_key, method):
    """Map cluster classifications to individual particles with error handling."""
    try:
        # Ensure inputs are proper arrays with correct types
        junk_clusters = np.asarray(junk_clusters, dtype=int)
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        
        # Find particles that belong to junk clusters
        junk_particle_indices = np.where(np.isin(cluster_indices, junk_clusters))[0]
        
        # Save the results
        output_file = os.path.join(output_folder, f'junk_indices_{zdim_key}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(junk_particle_indices, f)
        
        logger.info(f"{method}: Found {len(junk_particle_indices)} junk particles out of {len(cluster_indices)} total particles")
        return junk_particle_indices
        
    except Exception as e:
        logger.warning(f"Particle mapping failed: {str(e)}")
        # Return empty array as fallback
        return np.array([], dtype=int) 