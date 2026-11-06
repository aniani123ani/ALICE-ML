
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

file = uproot.open("bdt_dataset_randomcones.root")
tree = file["t"]

data = tree.arrays(
    ["pt_jet", "pt_leading", "avg_sums",
     "ppTruth", "PbPbtruth"],
    library="np"
)

for name, values in data.items():
    print(name, values.shape)


X = np.column_stack([
    data["pt_jet"],
    data["pt_leading"],
    data["avg_sums"]
])

y = data["PbPbtruth"]

features = ["pt_jet", "pt_leading", "avg_sums"]
target = "PbPbtruth"

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Features:", features)
print("Target:", target)


X_train, X_test, y_train, y_test, ppTruth_train, ppTruth_test = (
    train_test_split(
        X,
        y,
        data["ppTruth"],
        test_size=0.2,
        random_state=42
    )
)

#XGBoost

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=1.0,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

print("BDT training completed.")

#predictions

y_pred = model.predict(X_test)

print("Predictions completed.")
print("Number of predictions:", len(y_pred))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Test RMSE:", rmse, "GeV")


# 6. Baseline residual
baseline_residual = y_test - ppTruth_test

fig, ax = plt.subplots(figsize=(8, 7))

counts, xedges, yedges, im = ax.hist2d(
    ppTruth_test,
    baseline_residual,
    bins=50,
    range=[[0, 500], [-150, 250]],
    cmap="plasma",
    norm=colors.LogNorm(vmin=1)
)

ax.axhline(0, color="red", linestyle="--", alpha=0.7)

ax.set_xlabel(r"$p_T^{\mathrm{PPTruth}}$ [GeV]")
ax.set_ylabel(
    r"$p_T^{\mathrm{PbPbtruth}}-p_T^{\mathrm{PPTruth}}$ [GeV]"
)

ax.set_title("Baseline residual")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()


#residuals for the BDT model

ml_residual = y_test - y_pred

fig, ax = plt.subplots(figsize=(8, 7))

counts, xedges, yedges, im = ax.hist2d(
    ppTruth_test,
    ml_residual,
    bins=50,
    range=[[0, 500], [-150, 250]],
    cmap="plasma",
    norm=colors.LogNorm(vmin=1)
)

ax.axhline(0, color="red", linestyle="--", alpha=0.7)

ax.set_xlabel(r"$p_T^{\mathrm{PPTruth}}$ [GeV]")
ax.set_ylabel(
    r"$p_T^{\mathrm{PbPbtruth}}-p_T^{\mathrm{BDT}}$ [GeV]"
)

ax.set_title("BDT residuals with random-cone feature")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()




PbPbpredict_cone = data["PbPbtruth"] - data["avg_sums"]

rmse_cone = np.sqrt(
    mean_squared_error(
        data["ppTruth"],
        PbPbpredict_cone
    )
)

print("Random-cone subtraction RMSE:", rmse_cone, "GeV")

# Prediction minus reference, consistently signed.
cone_residual = PbPbpredict_cone - data["ppTruth"]


fig, ax = plt.subplots(figsize=(8, 7))

counts, xedges, yedges, im = ax.hist2d(
    data["ppTruth"],
    cone_residual,
    bins=50,
    range=[[0, 500], [-150, 250]],
    cmap="plasma",
    norm=colors.LogNorm(vmin=1)
)

ax.axhline(0, color="red", linestyle="--", alpha=0.7)

ax.set_xlabel(r"$p_T^{\mathrm{PPTruth}}$ [GeV]")
ax.set_ylabel(
    r"$p_T^{\mathrm{cone\ prediction}}-p_T^{\mathrm{PPTruth}}$ [GeV]"
)

ax.set_title("Two-random-cone subtraction residuals")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. Compare residuals in a selected PPTruth range
# --------------------------------------------------

pt_min, pt_max = 50, 110

mask_bdt = (
    (ppTruth_test >= pt_min) &
    (ppTruth_test <= pt_max)
)

bdt_residual_selected = ml_residual[mask_bdt]

mask_cone = (
    (data["ppTruth"] >= pt_min) &
    (data["ppTruth"] <= pt_max)
)

cone_residual_selected = cone_residual[mask_cone]


# Remove non-finite values
bdt_residual_selected = bdt_residual_selected[
    np.isfinite(bdt_residual_selected)
]

cone_residual_selected = cone_residual_selected[
    np.isfinite(cone_residual_selected)
]


# Fit Gaussian peaks
fit_min = -5
fit_max = 5

bdt_peak = bdt_residual_selected[
    (bdt_residual_selected >= fit_min) &
    (bdt_residual_selected <= fit_max)
]

cone_peak = cone_residual_selected[
    (cone_residual_selected >= fit_min) &
    (cone_residual_selected <= fit_max)
]

mu_bdt, sigma_bdt = norm.fit(bdt_peak)
mu_cone, sigma_cone = norm.fit(cone_peak)

print(
    f"Random cones: mean = {mu_cone:.2f} GeV, "
    f"width = {sigma_cone:.2f} GeV"
)

print(
    f"BDT: mean = {mu_bdt:.2f} GeV, "
    f"width = {sigma_bdt:.2f} GeV"
)


# --------------------------------------------------
# 10. Plot normalized residual distributions
# --------------------------------------------------

x = np.linspace(-30, 30, 1000)
bins = np.linspace(-30, 30, 61)

plt.figure(figsize=(8, 6))

plt.hist(
    cone_residual_selected,
    bins=bins,
    density=True,
    histtype="step",
    color="blue",
    linewidth=1.8,
    label="Two random cones"
)

plt.hist(
    bdt_residual_selected,
    bins=bins,
    density=True,
    histtype="step",
    color="red",
    linewidth=1.8,
    label="BDT"
)

plt.plot(
    x,
    norm.pdf(x, mu_cone, sigma_cone),
    color="blue",
    linestyle="--",
    label=(
        f"Random cones: "
        f"$\\mu={mu_cone:.1f}$, "
        f"$\\sigma={sigma_cone:.1f}$ GeV"
    )
)

plt.plot(
    x,
    norm.pdf(x, mu_bdt, sigma_bdt),
    color="red",
    linestyle="--",
    label=(
        f"BDT: "
        f"$\\mu={mu_bdt:.1f}$, "
        f"$\\sigma={sigma_bdt:.1f}$ GeV"
    )
)

plt.ylim(0, 1.0)

plt.axvline(
    0,
    color="black",
    linestyle=":",
    alpha=0.8
)

plt.xlabel(r"Residual $\delta p_T$ [GeV]")
plt.ylabel(r"Probability density [GeV$^{-1}$]")

plt.title(
    rf"Residuals for "
    rf"${pt_min} \leq p_T^{{\mathrm{{PPTruth}}}}"
    rf" \leq {pt_max}$ GeV"
)

plt.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.show()
