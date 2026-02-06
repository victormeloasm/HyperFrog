import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "thesis_plots_simulated"
DPI = 300

def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # Returns a list of colors. We will slice this list later to avoid warnings.
    return sns.color_palette("colorblind")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- DUMMY DATA GENERATOR ---
def generate_dummy_data():
    print("--- Generating simulated data (Benchmark Simulation) ---")
    data = []
    
    # 1. Microbenchmarks (KEM) - 200 iterations
    # Simulating Encaps ~3.5ms and Decaps ~4.2ms with some noise
    np.random.seed(42) # For reproducibility
    n_iters = 200
    
    enc_times = np.random.normal(loc=3.5, scale=0.15, size=n_iters)
    dec_times = np.random.normal(loc=4.2, scale=0.25, size=n_iters)
    
    # Add some mild outliers (tail latency)
    enc_times[10] = 5.1
    dec_times[50] = 6.8
    
    for i in range(n_iters):
        data.append({'case': 'encaps', 'iter': i, 'ms': enc_times[i], 'bytes': 0, 'notes': ''})
        data.append({'case': 'decaps', 'iter': i, 'ms': dec_times[i], 'bytes': 0, 'notes': ''})
        
    # 2. File (Streaming AEAD) - 64MB
    file_size = 64 * 1024 * 1024
    # Encrypt ~160 MB/s -> ms = (bytes/1024^2 / 160) * 1000
    enc_ms = ((file_size / 1048576.0) / 160.0) * 1000.0 
    # Decrypt ~145 MB/s
    dec_ms = ((file_size / 1048576.0) / 145.0) * 1000.0
    
    data.append({'case': 'file_enc', 'iter': 0, 'ms': enc_ms, 'bytes': file_size, 'notes': ''})
    data.append({'case': 'file_dec', 'iter': 0, 'ms': dec_ms, 'bytes': file_size, 'notes': ''})
    
    # 3. KeyGen
    data.append({'case': 'keygen', 'iter': 0, 'ms': 2450.0, 'bytes': 0, 'notes': 'genus=8;mine_ms=30000'})
    
    # 4. Check
    data.append({'case': 'file_check', 'iter': 0, 'ms': 0, 'bytes': file_size, 'notes': 'OK'})
    
    return pd.DataFrame(data)

# --- PLOTTING FUNCTIONS ---

# 1. DISTRIBUTION VIEW (Violin + Box)
def plot_distribution(df, palette, output_path):
    plt.figure(figsize=(10, 6))
    
    micro_df = df[df['case'].isin(['encaps', 'decaps'])].copy()
    micro_df['case'] = micro_df['case'].map({'encaps': 'Encapsulation', 'decaps': 'Decapsulation'})
    
    # FIX: Slice palette to match the number of hues (2) to avoid warnings
    current_palette = palette[:2]

    # Added hue='case' and legend=False for compatibility
    ax = sns.violinplot(x="case", y="ms", hue="case", data=micro_df, palette=current_palette, 
                        inner=None, alpha=0.3, legend=False)
    sns.boxplot(x="case", y="ms", hue="case", data=micro_df, palette=current_palette, 
                width=0.2, ax=ax, legend=False, boxprops={'zorder': 2})
    
    plt.title("Latency Distribution (KEM)", fontsize=16, fontweight='bold')
    plt.ylabel("Time (ms)")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    print(f"[OK] Saved: {output_path}")
    plt.close()

# 2. TEMPORAL STABILITY (Iteration x Time)
def plot_stability(df, palette, output_path):
    plt.figure(figsize=(12, 5))
    
    micro_df = df[df['case'].isin(['encaps', 'decaps'])].copy()
    micro_df['case'] = micro_df['case'].map({'encaps': 'Encapsulation', 'decaps': 'Decapsulation'})
    
    # FIX: Slice palette
    current_palette = palette[:2]

    sns.lineplot(x="iter", y="ms", hue="case", data=micro_df, palette=current_palette, linewidth=1.5, alpha=0.9)
    
    plt.title("Latency Stability per Iteration", fontsize=16, fontweight='bold')
    plt.ylabel("Latency (ms)")
    plt.xlabel("Iteration Number")
    plt.legend(title=None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    print(f"[OK] Saved: {output_path}")
    plt.close()

# 3. CUMULATIVE DISTRIBUTION FUNCTION (CDF)
def plot_cdf(df, palette, output_path):
    plt.figure(figsize=(10, 6))
    
    micro_df = df[df['case'].isin(['encaps', 'decaps'])].copy()
    micro_df['case'] = micro_df['case'].map({'encaps': 'Encapsulation', 'decaps': 'Decapsulation'})
    
    # FIX: Slice palette
    current_palette = palette[:2]

    sns.ecdfplot(data=micro_df, x="ms", hue="case", palette=current_palette, linewidth=2)
    
    plt.title("Cumulative Distribution Function (CDF)", fontsize=16, fontweight='bold')
    plt.ylabel("Cumulative Proportion")
    plt.xlabel("Latency (ms)")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    print(f"[OK] Saved: {output_path}")
    plt.close()

# 4. FILE THROUGHPUT (Bars)
def plot_throughput(df, palette, output_path):
    plt.figure(figsize=(8, 6))
    
    file_ops = df[df['case'].isin(['file_enc', 'file_dec'])].copy()
    
    # Calculation MB/s
    file_ops['mb_s'] = (file_ops['bytes'] / 1048576.0) / (file_ops['ms'] / 1000.0)
    file_ops['Label'] = file_ops['case'].map({'file_enc': 'Encryption', 'file_dec': 'Decryption'})
    
    ax = sns.barplot(x="Label", y="mb_s", hue="Label", data=file_ops, palette="viridis", legend=False)
    
    plt.title("Streaming Throughput (AES-GCM)", fontsize=16, fontweight='bold')
    plt.ylabel("Speed (MB/s)")
    plt.xlabel("")
    
    # Annotate values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f MB/s', padding=3, fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    print(f"[OK] Saved: {output_path}")
    plt.close()

# 5. STATS TABLE (Rendered as Image)
def plot_stats_table(df, output_path):
    micro_df = df[df['case'].isin(['encaps', 'decaps'])].copy()
    
    # Calculate statistics
    stats = micro_df.groupby('case')['ms'].describe(percentiles=[.5, .95, .99])
    stats = stats.reset_index()
    
    # Select and rename columns
    summary = stats[['case', 'mean', 'std', '50%', '95%', '99%']].copy()
    summary.columns = ['Operation', 'Mean (ms)', 'Std Dev', 'Median (p50)', 'p95', 'p99']
    summary['Operation'] = summary['Operation'].map({'encaps': 'Encapsulation', 'decaps': 'Decapsulation'})
    
    # Round values
    for col in summary.columns[1:]:
        summary[col] = summary[col].round(3)
        
    # Plot table
    fig, ax = plt.subplots(figsize=(10, 3)) 
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary.values, colLabels=summary.columns, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2) 
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    plt.title("Statistical Summary", fontsize=14, fontweight='bold', y=1.1)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()

def main():
    # Generates data on the fly, no file reading needed
    df = generate_dummy_data()
    
    palette = setup_style()
    ensure_dir(OUTPUT_DIR)
    
    print(f"--- Generating Thesis Plots in '{OUTPUT_DIR}/' ---")
    
    plot_distribution(df, palette, os.path.join(OUTPUT_DIR, "fig_1_distribution.png"))
    plot_stability(df, palette, os.path.join(OUTPUT_DIR, "fig_2_stability.png"))
    plot_cdf(df, palette, os.path.join(OUTPUT_DIR, "fig_3_cdf.png"))
    plot_throughput(df, palette, os.path.join(OUTPUT_DIR, "fig_4_throughput.png"))
    plot_stats_table(df, os.path.join(OUTPUT_DIR, "fig_5_stats_summary.png"))
    
    print("--- Done! ---")

if __name__ == "__main__":
    main()