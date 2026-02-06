import matplotlib
matplotlib.use('Agg') # Force headless mode
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pqc_comparison():
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    print("-> Generating PQC Comparison Chart...")

    # --- DATA SOURCE (NIST Round 3 & HyperFrog v33.1 Analysis) ---
    # HyperFrog values derived from v33.1 source code:
    # PK = 32 (seed) + 2048*2 (b) = 4128 bytes (~4KB)
    # CT = 256*4096*2 (u) + 256*2 (v) + 32 = ~2.1 MB
    # KeyGen = ~5200ms (Mining dominated)
    
    algorithms = [
        'HyperFrog v33.1\n(Unstructured LWE)', 
        'ML-KEM-1024\n(Kyber - NIST L5)', 
        'FrodoKEM-1344\n(Unstruct. NIST L5)', 
        'McEliece-6960119\n(Code-based L5)'
    ]

    # Metrics (Log Scale is required due to massive differences)
    # 1. Public Key Size (Bytes)
    pk_sizes = [4128, 1568, 21520, 1047319] 
    
    # 2. Ciphertext Size (Bytes)
    ct_sizes = [2097664, 1568, 21632, 226]
    
    # 3. KeyGen Time (Milliseconds - ref: i7/Zen3 single core estimates)
    # HyperFrog is slow due to topology mining
    keygen_times = [5200, 0.08, 2.6, 310] 

    x = np.arange(len(algorithms))
    width = 0.25  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotting the grouped bars
    rects1 = ax.bar(x - width, pk_sizes, width, label='Public Key (Bytes)', color='#3498db')
    rects2 = ax.bar(x, ct_sizes, width, label='Ciphertext (Bytes)', color='#e74c3c')
    rects3 = ax.bar(x + width, keygen_times, width, label='KeyGen Time (ms)', color='#f1c40f')

    # Formatting
    ax.set_ylabel('Magnitude (Log Scale)', fontweight='bold')
    ax.set_title('PQC Landscape: HyperFrog v33.1 vs NIST Standards (Level 5)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    
    # Set Log Scale on Y-axis
    ax.set_yscale('log')
    
    # Add grid for readability
    ax.grid(True, which="major", axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Function to add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Custom formatting for readability
            if height >= 1000000:
                label = f'{height/1000000:.1f}M'
            elif height >= 1000:
                label = f'{height/1000:.1f}k'
            else:
                label = f'{height:.1f}'
            
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    
    output_path = os.path.join(cwd, "pqc_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f" SAVED: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    try:
        plot_pqc_comparison()
        print("\nSUCCESS: Comparison chart generated.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")