import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURAO ---
INPUT_FILE = "mykeys.bench.csv"  # O arquivo que voc fez upload
OUTPUT_DIR = "."                 # Diretrio atual
DPI = 300                        # Alta resoluo

def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    return sns.color_palette("colorblind")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Carregar os dados do arquivo CSV
    try:
        df = pd.read_csv(INPUT_FILE)
        df['notes'] = df['notes'].fillna("")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{INPUT_FILE}' no encontrado.")
        return

    palette = setup_style()
    ensure_dir(OUTPUT_DIR)
    
    print(f"--- Generating Thesis Plots from '{INPUT_FILE}' ---")
    
    # 1. DISTRIBUTION VIEW (Violin + Box)
    plt.figure(figsize=(10, 6))
    micro_df = df[df['case'].isin(['encaps', 'decaps'])].copy()
    micro_df['case'] = micro_df['case'].map({'encaps': 'Encapsulation', 'decaps': 'Decapsulation'})
    current_palette = palette[:2]

    # Violin plot
    ax = sns.violinplot(x="case", y="ms", hue="case", data=micro_df, palette=current_palette, 
                        inner=None, alpha=0.3, legend=False)
    # Box plot (fixed legend issue)
    sns.boxplot(x="case", y="ms", hue="case", data=micro_df, palette=current_palette, 
                width=0.2, ax=ax, boxprops={'zorder': 2})

    plt.title("Latency Distribution (KEM)", fontsize=16, fontweight='bold')
    plt.ylabel("Time (ms)")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_1_distribution.png"), dpi=DPI)
    plt.close()

    # 2. TEMPORAL STABILITY
    plt.figure(figsize=(12, 5))
    sns.lineplot(x="iter", y="ms", hue="case", data=micro_df, palette=current_palette, linewidth=1.5, alpha=0.9)

    plt.title("Latency Stability per Iteration", fontsize=16, fontweight='bold')
    plt.ylabel("Latency (ms)")
    plt.xlabel("Iteration Number")
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_2_stability.png"), dpi=DPI)
    plt.close()

    # 3. CDF
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=micro_df, x="ms", hue="case", palette=current_palette, linewidth=2)

    plt.title("Cumulative Distribution Function (CDF)", fontsize=16, fontweight='bold')
    plt.ylabel("Cumulative Proportion")
    plt.xlabel("Latency (ms)")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_cdf.png"), dpi=DPI)
    plt.close()

    # 4. THROUGHPUT
    plt.figure(figsize=(8, 6))
    file_ops = df[df['case'].isin(['file_enc', 'file_dec'])].copy()

    if not file_ops.empty:
        # Clculo MB/s = (bytes / 1024^2) / (ms / 1000)
        file_ops['mb_s'] = (file_ops['bytes'] / 1048576.0) / (file_ops['ms'] / 1000.0)
        file_ops['Label'] = file_ops['case'].map({'file_enc': 'Encryption', 'file_dec': 'Decryption'})

        ax = sns.barplot(x="Label", y="mb_s", hue="Label", data=file_ops, palette="viridis")
        # Remove legend if created automatically
        if ax.get_legend():
            ax.get_legend().remove()

        plt.title("Streaming Throughput (AES-GCM)", fontsize=16, fontweight='bold')
        plt.ylabel("Speed (MB/s)")
        plt.xlabel("")

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f MB/s', padding=3, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig_4_throughput.png"), dpi=DPI)
        plt.close()

    # 5. STATS TABLE
    stats = micro_df.groupby('case')['ms'].describe(percentiles=[.5, .95, .99])
    stats = stats.reset_index()
    summary = stats[['case', 'mean', 'std', '50%', '95%', '99%']].copy()
    summary.columns = ['Operation', 'Mean (ms)', 'Std Dev', 'Median (p50)', 'p95', 'p99']
    
    # Arredondamento
    for col in summary.columns[1:]:
        summary[col] = summary[col].round(3)

    fig, ax = plt.subplots(figsize=(10, 3)) 
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary.values, colLabels=summary.columns, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2) 
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')

    plt.title("Statistical Summary", fontsize=14, fontweight='bold', y=1.1)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_5_stats_summary.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("--- Done! Images saved. ---")

if __name__ == "__main__":
    main()