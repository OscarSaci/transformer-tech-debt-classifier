"""
Script per generare i grafici del carbon footprint per la tesi
Genera tutte le figure menzionate nella sezione dei risultati
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configurazione stile grafici
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Creare cartella per salvare i grafici
output_dir = Path("carbon_plots")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# DATI DEL TRAINING
# ============================================================================

training_data = {
    'Model': ['Heuristic-Only', 'Text-Based\nBaseline', 'Saliency-\nEnhanced', 'Hybrid\nEnsemble'],
    'Model_Full': ['Heuristic-Only', 'Text-Based Baseline', 'Saliency-Enhanced', 'Hybrid Ensemble'],
    'Duration_sec': [3.28, 337.67, 359.50, 586.17],
    'Duration_min': [0.05, 5.63, 5.99, 9.77],
    'CO2_kg': [0.000011, 0.001355, 0.001380, 0.002383],
    'CO2_g': [0.000011, 1.355, 1.380, 2.383],
    'Energy_Wh': [0.0088, 1.138, 1.159, 2.001],
    'CPU_util': [0.00, 21.34, 36.52, 22.39],
    'RAM_util': [68.30, 67.29, 69.90, 67.72]
}

df_training = pd.DataFrame(training_data)

# ============================================================================
# DATI DELL'INFERENCE
# ============================================================================

inference_data = {
    'Model': ['CatBoost\nDistilled', 'DistilBERT\nTeacher'],
    'Model_Full': ['CatBoost Distilled', 'DistilBERT Teacher'],
    'Duration_sec': [0.055, 2.862],
    'CO2_mg': [0.0002, 32.13],
    'CO2_per_sample_mg': [0.0012, 0.1607],
    'Throughput_samples_per_sec': [5228.80, 69.88],
    'GPU_util': [0.00, 89.50],
    'F1_Score': [0.820, 0.800]
}

df_inference = pd.DataFrame(inference_data)

# ============================================================================
# DATI LIFECYCLE
# ============================================================================

# Emissioni per 1 sample in mg
co2_per_sample = {
    'Heuristic-Only': 0.0012,  # assunto simile a CatBoost
    'Text-Based': 0.0012,
    'Saliency-Enhanced': 0.0012,
    'Hybrid Ensemble': 0.0012,
    'DistilBERT': 0.1607
}

# Training emissions in g
training_co2_g = {
    'Heuristic-Only': 0.000011,
    'Text-Based': 1.355,
    'Saliency-Enhanced': 1.380,
    'Hybrid Ensemble': 2.383,
    'DistilBERT': 1.380  # stimato
}

# Calcolo lifecycle per diverse scale
scales = np.array([1000, 10000, 100000, 1000000])
lifecycle_data = {}

for model, train_g in training_co2_g.items():
    inference_per_sample_g = co2_per_sample[model] / 1000  # da mg a g
    lifecycle_data[model] = [train_g + (n * inference_per_sample_g) for n in scales]

df_lifecycle = pd.DataFrame(lifecycle_data, index=scales)

# ============================================================================
# GRAFICO 1: Comparison of CO2 emissions across models during training phase
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

bars = ax1.bar(df_training['Model'], df_training['CO2_g'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Aggiungere valori sopra le barre
for i, (bar, val) in enumerate(zip(bars, df_training['CO2_g'])):
    height = bar.get_height()
    if val < 0.001:
        label = f'{val:.6f}'
    else:
        label = f'{val:.3f}'
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('CO$_2$ Emissions (g)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax1.set_title('Comparison of CO$_2$ Emissions Across Models During Training Phase', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(df_training['CO2_g']) * 1.15)

plt.tight_layout()
plt.savefig(output_dir / 'training_co2_emissions.png', bbox_inches='tight')
plt.savefig(output_dir / 'training_co2_emissions.pdf', bbox_inches='tight')
print("✓ Grafico 1 salvato: training_co2_emissions.png/pdf")
plt.close()

# ============================================================================
# GRAFICO 2: Comparison of training duration across model architectures
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))

bars = ax2.bar(df_training['Model'], df_training['Duration_min'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Aggiungere valori sopra le barre
for bar, val in zip(bars, df_training['Duration_min']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Training Duration (minutes)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax2.set_title('Comparison of Training Duration Across Model Architectures', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(df_training['Duration_min']) * 1.15)

plt.tight_layout()
plt.savefig(output_dir / 'training_duration.png', bbox_inches='tight')
plt.savefig(output_dir / 'training_duration.pdf', bbox_inches='tight')
print("✓ Grafico 2 salvato: training_duration.png/pdf")
plt.close()

# ============================================================================
# GRAFICO 3: Logarithmic comparison of per-sample CO2 emissions during inference
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(8, 6))
colors_inf = ['#27ae60', '#8e44ad']

bars = ax3.bar(df_inference['Model'], df_inference['CO2_per_sample_mg'], 
               color=colors_inf, alpha=0.8, edgecolor='black', linewidth=1.2)

# Aggiungere valori sopra le barre
for bar, val in zip(bars, df_inference['CO2_per_sample_mg']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height * 1.3,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_yscale('log')
ax3.set_ylabel('CO$_2$ Emissions per Sample (mg)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_title('Logarithmic Comparison of Per-Sample CO$_2$ Emissions During Inference\n(Note: Logarithmic Scale)', 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--', which='both')
ax3.set_ylim(0.0001, 1)

# Calcolare e mostrare il fattore di riduzione
reduction_factor = df_inference['CO2_per_sample_mg'].iloc[1] / df_inference['CO2_per_sample_mg'].iloc[0]
ax3.text(0.5, 0.95, f'Reduction: {reduction_factor:.0f}×', 
         transform=ax3.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
         fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'inference_co2_per_sample.png', bbox_inches='tight')
plt.savefig(output_dir / 'inference_co2_per_sample.pdf', bbox_inches='tight')
print("✓ Grafico 3 salvato: inference_co2_per_sample.png/pdf")
plt.close()

# ============================================================================
# GRAFICO 4: Logarithmic comparison of per-sample inference time
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(8, 6))

# Calcolare tempo per sample in ms
time_per_sample_ms = (df_inference['Duration_sec'] / 200) * 1000

bars = ax4.bar(df_inference['Model'], time_per_sample_ms, 
               color=colors_inf, alpha=0.8, edgecolor='black', linewidth=1.2)

# Aggiungere valori sopra le barre
for bar, val in zip(bars, time_per_sample_ms):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height * 1.3,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_yscale('log')
ax4.set_ylabel('Inference Time per Sample (ms)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
ax4.set_title('Logarithmic Comparison of Per-Sample Inference Time\n(Note: Logarithmic Scale)', 
              fontsize=13, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3, linestyle='--', which='both')
ax4.set_ylim(0.1, 100)

# Calcolare e mostrare il speedup
speedup = time_per_sample_ms.iloc[1] / time_per_sample_ms.iloc[0]
ax4.text(0.5, 0.95, f'Speedup: {speedup:.0f}×', 
         transform=ax4.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
         fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'inference_time_per_sample.png', bbox_inches='tight')
plt.savefig(output_dir / 'inference_time_per_sample.pdf', bbox_inches='tight')
print("✓ Grafico 4 salvato: inference_time_per_sample.png/pdf")
plt.close()

# ============================================================================
# GRAFICO 5: Lifecycle CO2 emissions as a function of deployment scale
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 7))

# Colori per ogni modello
colors_lifecycle = {
    'Heuristic-Only': '#2ecc71',
    'Text-Based': '#3498db',
    'Saliency-Enhanced': '#e67e22',
    'Hybrid Ensemble': '#e74c3c',
    'DistilBERT': '#9b59b6'
}

markers = ['o', 's', '^', 'D', 'p']
linestyles = ['-', '-', '-', '-', '--']

for i, (model, color) in enumerate(colors_lifecycle.items()):
    ax5.plot(scales, df_lifecycle[model], 
             marker=markers[i], markersize=8, 
             linewidth=2.5, label=model, 
             color=color, linestyle=linestyles[i],
             alpha=0.8)

ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('Number of Inferences', fontsize=12, fontweight='bold')
ax5.set_ylabel('Total Lifecycle CO$_2$ (g)', fontsize=12, fontweight='bold')
ax5.set_title('Lifecycle CO$_2$ Emissions as a Function of Deployment Scale\n(Note: Logarithmic Scale)', 
              fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, linestyle='--', which='both')
ax5.legend(loc='upper left', framealpha=0.9, edgecolor='black')

# Aggiungere annotazioni per punti chiave
# Annotare il valore a 1M per CatBoost vs DistilBERT
ax5.annotate(f'{df_lifecycle["Hybrid Ensemble"][-1]:.2f} g', 
             xy=(scales[-1], df_lifecycle["Hybrid Ensemble"][-1]),
             xytext=(scales[-1]*0.3, df_lifecycle["Hybrid Ensemble"][-1]*3),
             fontsize=8, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=1.5))

ax5.annotate(f'{df_lifecycle["DistilBERT"][-1]:.0f} g\n(160.7 kg!)', 
             xy=(scales[-1], df_lifecycle["DistilBERT"][-1]),
             xytext=(scales[-1]*0.2, df_lifecycle["DistilBERT"][-1]*0.3),
             fontsize=8, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

ax5.set_xlim(500, 2000000)
ax5.set_ylim(0.005, 300000)

plt.tight_layout()
plt.savefig(output_dir / 'lifecycle_emissions.png', bbox_inches='tight')
plt.savefig(output_dir / 'lifecycle_emissions.pdf', bbox_inches='tight')
print("✓ Grafico 5 salvato: lifecycle_emissions.png/pdf")
plt.close()

# ============================================================================
# GRAFICO BONUS: Combined overview con subplots
# ============================================================================

fig6, axes = plt.subplots(2, 3, figsize=(16, 10))
fig6.suptitle('Carbon Footprint Analysis - Complete Overview', fontsize=16, fontweight='bold')

# Subplot 1: Training CO2
ax = axes[0, 0]
bars = ax.bar(df_training['Model'], df_training['CO2_g'], color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, df_training['CO2_g']):
    height = bar.get_height()
    label = f'{val:.4f}' if val < 0.001 else f'{val:.3f}'
    ax.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom', fontsize=7)
ax.set_ylabel('CO$_2$ (g)', fontsize=10)
ax.set_title('Training CO$_2$ Emissions', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8, rotation=0)
ax.grid(axis='y', alpha=0.3)

# Subplot 2: Training Duration
ax = axes[0, 1]
bars = ax.bar(df_training['Model'], df_training['Duration_min'], color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, df_training['Duration_min']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Duration (min)', fontsize=10)
ax.set_title('Training Duration', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8, rotation=0)
ax.grid(axis='y', alpha=0.3)

# Subplot 3: Training Energy
ax = axes[0, 2]
bars = ax.bar(df_training['Model'], df_training['Energy_Wh'], color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, df_training['Energy_Wh']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Energy (Wh)', fontsize=10)
ax.set_title('Training Energy Consumption', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8, rotation=0)
ax.grid(axis='y', alpha=0.3)

# Subplot 4: Inference CO2 (log scale)
ax = axes[1, 0]
bars = ax.bar(df_inference['Model'], df_inference['CO2_per_sample_mg'], color=colors_inf, alpha=0.8, edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel('CO$_2$/sample (mg, log)', fontsize=10)
ax.set_title('Inference CO$_2$ per Sample', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', alpha=0.3, which='both')

# Subplot 5: Inference Time (log scale)
ax = axes[1, 1]
bars = ax.bar(df_inference['Model'], time_per_sample_ms, color=colors_inf, alpha=0.8, edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel('Time/sample (ms, log)', fontsize=10)
ax.set_title('Inference Time per Sample', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', alpha=0.3, which='both')

# Subplot 6: Lifecycle mini version
ax = axes[1, 2]
for model, color, ls in zip(['Heuristic-Only', 'Hybrid Ensemble', 'DistilBERT'], 
                             ['#2ecc71', '#e74c3c', '#9b59b6'],
                             ['-', '-', '--']):
    ax.plot(scales, df_lifecycle[model], marker='o', markersize=5, 
            linewidth=2, label=model, color=color, linestyle=ls, alpha=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Inferences', fontsize=10)
ax.set_ylabel('Total CO$_2$ (g, log)', fontsize=10)
ax.set_title('Lifecycle Emissions', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / 'carbon_overview_complete.png', bbox_inches='tight')
plt.savefig(output_dir / 'carbon_overview_complete.pdf', bbox_inches='tight')
print("✓ Grafico BONUS salvato: carbon_overview_complete.png/pdf")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - FILE GENERATI")
print("="*80)
print(f"\nCartella di output: {output_dir.absolute()}\n")
print("File generati:")
print("  1. training_co2_emissions.png/pdf")
print("  2. training_duration.png/pdf")
print("  3. inference_co2_per_sample.png/pdf")
print("  4. inference_time_per_sample.png/pdf")
print("  5. lifecycle_emissions.png/pdf")
print("  6. carbon_overview_complete.png/pdf (BONUS)")
print("\nTotale: 12 file (6 PNG + 6 PDF)")
print("="*80)

# ============================================================================
# STATISTICHE CHIAVE
# ============================================================================

print("\n" + "="*80)
print("STATISTICHE CHIAVE")
print("="*80)

print("\nTRAINING:")
print(f"  • Modello più veloce: {df_training.loc[df_training['Duration_min'].idxmin(), 'Model_Full']} ({df_training['Duration_min'].min():.2f} min)")
print(f"  • Modello più lento: {df_training.loc[df_training['Duration_min'].idxmax(), 'Model_Full']} ({df_training['Duration_min'].max():.2f} min)")
print(f"  • Minime emissioni: {df_training.loc[df_training['CO2_g'].idxmin(), 'Model_Full']} ({df_training['CO2_g'].min():.6f} g)")
print(f"  • Massime emissioni: {df_training.loc[df_training['CO2_g'].idxmax(), 'Model_Full']} ({df_training['CO2_g'].max():.3f} g)")

speedup_training = df_training['Duration_min'].max() / df_training['Duration_min'].min()
reduction_training = df_training['CO2_g'].max() / df_training['CO2_g'].min()
print(f"  • Speedup max: {speedup_training:.1f}×")
print(f"  • Riduzione emissioni max: {reduction_training:.0f}×")

print("\nINFERENCE:")
speedup_inf = df_inference['Duration_sec'].iloc[1] / df_inference['Duration_sec'].iloc[0]
reduction_inf = df_inference['CO2_per_sample_mg'].iloc[1] / df_inference['CO2_per_sample_mg'].iloc[0]
throughput_gain = df_inference['Throughput_samples_per_sec'].iloc[0] / df_inference['Throughput_samples_per_sec'].iloc[1]

print(f"  • Speedup CatBoost vs DistilBERT: {speedup_inf:.1f}×")
print(f"  • Riduzione CO2: {reduction_inf:.0f}×")
print(f"  • Throughput gain: {throughput_gain:.1f}×")

print("\nLIFECYCLE (1M inferences):")
for model in ['Heuristic-Only', 'Hybrid Ensemble', 'DistilBERT']:
    emissions_1m = df_lifecycle.loc[1000000, model]
    unit = 'g' if emissions_1m < 1000 else 'kg'
    value = emissions_1m if emissions_1m < 1000 else emissions_1m / 1000
    print(f"  • {model}: {value:.2f} {unit}")

ratio_hybrid_vs_bert = df_lifecycle.loc[1000000, 'DistilBERT'] / df_lifecycle.loc[1000000, 'Hybrid Ensemble']
print(f"  • DistilBERT è {ratio_hybrid_vs_bert:.0f}× più inquinante del Hybrid Ensemble!")

print("="*80 + "\n")

print("✅ Tutti i grafici sono stati generati con successo!")
print("   Puoi usare i file PNG per presentazioni o i PDF per la tesi LaTeX.\n")
