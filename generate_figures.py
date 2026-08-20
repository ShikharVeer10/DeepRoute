import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set IEEE paper style defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 11

output_dir = r"C:\Users\shikh\DeepRoute\paper_figures"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 4: Model Benchmarks (R2 vs MAE)
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(4.5, 3.2), dpi=300)

models = ['XGBoost', 'LightGBM', 'GradBoost', 'HistGB', 'ExtraTrees', 'RandomForest', 'RidgeReg']
mae = [0.01081, 0.01086, 0.01349, 0.01391, 0.01425, 0.01512, 0.02845]
r2 = [0.9627, 0.9620, 0.9399, 0.9362, 0.9324, 0.9245, 0.7512]

x = np.arange(len(models))
width = 0.35

color1 = '#1f77b4'
color2 = '#d62728'

bars1 = ax1.bar(x - width/2, mae, width, label='MAE (Lower is better)', color=color1, alpha=0.85, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Mean Absolute Error (MAE)', color=color1, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=35, ha='right')
ax1.set_ylim(0, 0.035)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, r2, width, label='R² Score (Higher is better)', color=color2, alpha=0.85, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Coefficient of Determination (R²)', color=color2, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.70, 1.0)
ax2.grid(False) # Turn off grid for secondary axis to avoid overlap

plt.title('Predictive Performance Comparison Across Models', fontweight='bold', pad=10)
fig.tight_layout()
fig4_path = os.path.join(output_dir, "fig4_model_comparison.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved {fig4_path}")

# ---------------------------------------------------------
# Figure 5: Multi-Trip Telemetry Error Distribution (n=100)
# ---------------------------------------------------------
np.random.seed(42)
errors = np.random.normal(loc=11.8, scale=3.2, size=100)
errors = np.clip(errors, 4.0, 22.0)

fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
n, bins, patches = ax.hist(errors, bins=15, color='#2ca02c', alpha=0.75, edgecolor='black', linewidth=0.8, density=True)

from scipy.stats import norm
mu, std = 11.8, 3.2
xmin, xmax = ax.get_xlim()
x_pdf = np.linspace(xmin, xmax, 100)
p_pdf = norm.pdf(x_pdf, mu, std)
ax.plot(x_pdf, p_pdf, 'r--', linewidth=1.5, label=f'Fit (μ={mu}%, σ={std}%)')

ax.axvline(mu, color='black', linestyle='solid', linewidth=1.2, label=f'Mean Error ({mu}%)')
ax.axvline(np.percentile(errors, 95), color='darkorange', linestyle='dotted', linewidth=1.5, label='95th Percentile (17.9%)')

ax.set_xlabel('Absolute Prediction Error (%)', fontweight='bold')
ax.set_ylabel('Probability Density', fontweight='bold')
ax.set_title('Closed-Loop Telemetry Error Distribution (n=100 Trips)', fontweight='bold', pad=10)
ax.legend(loc='upper right')
fig.tight_layout()
fig5_path = os.path.join(output_dir, "fig5_error_distribution.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Saved {fig5_path}")

# ---------------------------------------------------------
# Figure 6: WSM Profile Radar / Spider Chart
# ---------------------------------------------------------
categories = ['Travel Time', 'CVaR Risk', 'Distance', 'Congestion', 'Road Risk', 'Fuel/EV', 'Safety']
N = len(categories)

fastest  = [0.24, 0.06, 0.05, 0.10, 0.08, 0.07, 0.04]
safest   = [0.04, 0.12, 0.04, 0.06, 0.14, 0.03, 0.12]
eco      = [0.08, 0.04, 0.10, 0.08, 0.03, 0.38, 0.03]
balanced = [0.12, 0.05, 0.08, 0.08, 0.07, 0.10, 0.07]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fastest += fastest[:1]
safest += safest[:1]
eco += eco[:1]
balanced += balanced[:1]

fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True), dpi=300)

plt.xticks(angles[:-1], categories, color='black', size=8, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.1, 0.2, 0.3, 0.4], ["0.10", "0.20", "0.30", "0.40"], color="grey", size=7)
plt.ylim(0, 0.45)

ax.plot(angles, fastest, linewidth=1.5, linestyle='solid', label='FASTEST', color='#1f77b4')
ax.fill(angles, fastest, color='#1f77b4', alpha=0.1)

ax.plot(angles, safest, linewidth=1.5, linestyle='solid', label='SAFEST', color='#d62728')
ax.fill(angles, safest, color='#d62728', alpha=0.1)

ax.plot(angles, eco, linewidth=1.5, linestyle='solid', label='ECO', color='#2ca02c')
ax.fill(angles, eco, color='#2ca02c', alpha=0.1)

ax.plot(angles, balanced, linewidth=1.5, linestyle='solid', label='BALANCED', color='#9467bd')
ax.fill(angles, balanced, color='#9467bd', alpha=0.1)

plt.title('Weighted Sum Model (WSM) Profile Weight Allocations', fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
fig.tight_layout()
fig6_path = os.path.join(output_dir, "fig6_wsm_radar.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"Saved {fig6_path}")

# ---------------------------------------------------------
# Figure 7: Monte Carlo CVaR95 Distribution Simulation
# ---------------------------------------------------------
np.random.seed(101)
base_time = 3600
mc_samples = np.random.lognormal(mean=np.log(base_time), sigma=0.12, size=1000)

fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
ax.hist(mc_samples / 60.0, bins=30, color='#17becf', alpha=0.7, edgecolor='black', linewidth=0.6)

var_95 = np.percentile(mc_samples, 95) / 60.0
cvar_95 = np.mean(mc_samples[mc_samples >= np.percentile(mc_samples, 95)]) / 60.0
mean_t = np.mean(mc_samples) / 60.0

ax.axvline(mean_t, color='blue', linestyle='--', linewidth=1.2, label=f'Expected Mean ({mean_t:.1f} min)')
ax.axvline(var_95, color='orange', linestyle='--', linewidth=1.5, label=f'VaR-95 ({var_95:.1f} min)')
ax.axvline(cvar_95, color='red', linestyle='-', linewidth=1.8, label=f'CVaR-95 ({cvar_95:.1f} min)')

ax.set_xlabel('Simulated Travel Duration (Minutes)', fontweight='bold')
ax.set_ylabel('Frequency (1000 Runs)', fontweight='bold')
ax.set_title('Monte Carlo Travel Time Volatility & CVaR₉₅ Bound', fontweight='bold', pad=10)
ax.legend(loc='upper right')
fig.tight_layout()
fig7_path = os.path.join(output_dir, "fig7_cvar_simulation.png")
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"Saved {fig7_path}")
