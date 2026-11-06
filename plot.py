import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm

#Training and Validation Plots PLOT1
loss_data = np.load("loss_history.npz")
train_loss_history = loss_data["train_loss_history"]
val_loss_history = loss_data["val_loss_history"]
def plot_training_loss(train_loss_history, val_loss_history):
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_history,label="Training Loss")
    plt.plot(val_loss_history,label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
plot_training_loss(train_loss_history, val_loss_history)

#Residual for PbPb-Prediction residuals + 2D residual plot PLOT2
plot_data = np.load("dnn_plot_data.npz")
dnn_pred= plot_data["dnn_pred"]
dnn_truth = plot_data["dnn_truth"]
pp_truth = plot_data["pp_truth_test"]
dnn_residuals = dnn_truth-dnn_pred
fig, ax =plt.subplots(figsize=(8,7))
counts, xedges, yedges, im = ax.hist2d(
    pp_truth,
    dnn_residuals,
    bins=50,
    range=[[0,500],[-150,250]],
    cmap="plasma",
    norm=LogNorm(vmin=1)
)
ax.axhline(0, color="red",linestyle="--",alpha=0.7)
ax.set_xlabel(r"$p_T^{\mathrm{ppTruth}}$ [GeV]")
ax.set_ylabel(
    r"$p_T^{\mathrm{PbPbtruth}} - p_T^{\mathrm{DNN}}$ [GeV]"
)
ax.set_title("DNN residuals")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()
plt.close(fig)

#PLOT3
pt_min, pt_max=50,110
fit_min,fit_max=-5,5
mask =(
    (pp_truth >=pt_min) &
    (pp_truth <=pt_max)
)
residual_selected = dnn_residuals[mask]
residual_selected = residual_selected[
    np.isfinite(residual_selected)
]
print("Number of selected DNN jets:" , len(residual_selected))
if len(residual_selected) < 2:
    raise ValueError("Too few selected residuals to fit a Gaussian.")

print("Mean residual:", np.mean(residual_selected), "GeV")
print(
    "Residual standard deviation:",
    np.std(residual_selected),
    "GeV"
)

# Fit Gaussian to the central residual region
peak = residual_selected[
    (residual_selected >= fit_min) &
    (residual_selected <= fit_max)
]

if len(peak) < 2:
    raise ValueError(
        "Too few residuals between -5 and 5 GeV for the Gaussian fit."
    )

mu_dnn, sigma_dnn = norm.fit(peak)

print(
    f"DNN Gaussian: mean = {mu_dnn:.2f} GeV, "
    f"sigma = {sigma_dnn:.2f} GeV"
)

# Plot settings
x_min, x_max = -30, 30
y_min, y_max = 0, 0.6

bins = np.linspace(x_min, x_max, 61)
x = np.linspace(x_min, x_max, 1000)

fig, ax = plt.subplots(figsize=(8, 6))

# Normalized probability-density histogram
ax.hist(
    residual_selected,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    label="DNN"
)

# Gaussian fit to the central residuals
ax.plot(
    x,
    norm.pdf(x, mu_dnn, sigma_dnn),
    "--",
    linewidth=1.8,
    label=(
        fr"DNN Gaussian: $\mu={mu_dnn:.1f}$, "
        fr"$\sigma={sigma_dnn:.1f}$ GeV"
    )
)

ax.axvline(0, color="black", linestyle=":", alpha=0.8)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(r"Residual $\delta p_T$ [GeV]")
ax.set_ylabel(r"Probability density [GeV$^{-1}$]")
ax.set_title(
    fr"DNN residuals for "
    fr"${pt_min} \leq p_T^{{\mathrm{{ppTruth}}}} "
    fr"\leq {pt_max}$ GeV"
)

ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
plt.close(fig)


