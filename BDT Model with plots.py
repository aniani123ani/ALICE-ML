import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
file = uproot.open("ml_dataset.root")
tree = file["t"]
data = tree.arrays(
    ["pt_jet", "pt_leading", "pt_soft_sum", "ppTruth", "PbPbtruth"],
    library="np"
)

for name, values in data.items():
    print(name, values.shape)

# inputs for bdt
X = np.column_stack([
    data["pt_jet"],
    data["pt_leading"],
    data["pt_soft_sum"]
])
#output
y = data["PbPbtruth"]
print("X shape:", X.shape)
print("y shape:", y.shape)

#input and output types
features = ["pt_jet", "pt_leading", "pt_soft_sum"]
target = "PbPbtruth"

# which variables will be used
print("Features:", features)
print("Target:", target)
from sklearn.model_selection import train_test_split

# Spliting the dataset into training and test samples
X_train, X_test, y_train, y_test, ppTruth_train, ppTruth_test = train_test_split(
    X,
    y,
    data["ppTruth"],
    test_size=0.2,
    random_state=42
)
import xgboost
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train)
print("BDT training completed.")

#Predicting the PbPb jet pT for the test set
y_pred = model.predict(X_test)
print("predictions completed.")
print("Number of predictions:", len(y_pred))

from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse, "GeV")

#baseline residual before applying ML
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
ax.set_xlabel(r"PPTruth [GeV]")
ax.set_ylabel(r"$p_T^{\mathrm{PbPbtruth}} - PPTruth$ [GeV]")
ax.set_title("Baseline residual")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()


############################
#the residual after applying the BDT
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
ax.set_xlabel(r"PPTruth [GeV]")
ax.set_ylabel(r"$p_T^{\mathrm{PbPbtruth}} - p_T^{\mathrm{BDT}}$ [GeV]")
ax.set_title("BDT residuals")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()


baseline_rmse = np.sqrt(mean_squared_error(y_test, ppTruth_test))


PbPbpredict_cone=data["PbPbtruth"]-data["pt_soft_sum"]
from sklearn.metrics import mean_squared_error
rmse_cone = np.sqrt(
    mean_squared_error(
        data["ppTruth"],
        PbPbpredict_cone
    )
)

print("Cone-subtraction RMSE:", rmse_cone, "GeV")
cone_residual = data["ppTruth"] - PbPbpredict_cone


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
ax.set_xlabel(r"PPTruth [GeV]")
ax.set_ylabel(r"$p_T^{\mathrm{PbPbtruth}} - p_T^{\mathrm{Cone}}$ [GeV]")
ax.set_title("Cone-subtraction residuals")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()




from scipy.stats import norm


pt_min, pt_max = 50, 110

mask_bdt = (ppTruth_test >= pt_min) & (ppTruth_test <= pt_max)
bdt_residual_selected = ml_residual[mask_bdt]

mask_cone = (data["ppTruth"] >= pt_min) & (data["ppTruth"] <= pt_max)
cone_residual_selected = cone_residual[mask_cone]


bdt_residual_selected = bdt_residual_selected[
    np.isfinite(bdt_residual_selected)
]
cone_residual_selected = cone_residual_selected[
    np.isfinite(cone_residual_selected)
]

# Fit Gaussians to the peaks only
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


print(f"Cone: mean = {mu_cone:.2f} GeV, width = {sigma_cone:.2f} GeV")
print(f"BDT:  mean = {mu_bdt:.2f} GeV, width = {sigma_bdt:.2f} GeV")


# Common residual axis
x = np.linspace(-15, 15, 1000)
bins = np.linspace(-15, 15, 61)

plt.figure(figsize=(8, 6))

# Normalized residual histograms
plt.hist(
    cone_residual_selected, bins=bins, density=True,
    histtype="step", color="blue", linewidth=1.8,
    label="Cone subtraction"
)
plt.hist(
    bdt_residual_selected, bins=bins, density=True,
    histtype="step", color="red", linewidth=1.8,
    label="BDT"
)

# Gaussian fit curves
plt.plot(
    x, norm.pdf(x, mu_cone, sigma_cone),
    color="blue", linestyle="--",
    label=fr"Cone Gaussian: $\mu={mu_cone:.1f}$, $\sigma={sigma_cone:.1f}$ GeV"
)
plt.plot(
    x, norm.pdf(x, mu_bdt, sigma_bdt),
    color="red", linestyle="--",
    label=fr"BDT Gaussian: $\mu={mu_bdt:.1f}$, $\sigma={sigma_bdt:.1f}$ GeV"
)
plt.ylim(0, 1.0)
plt.axvline(0, color="black", linestyle=":", alpha=0.8)

plt.xlabel(r"Residual $\delta p_T$ [GeV]")
plt.ylabel(r"Probability density [GeV$^{-1}$]")
plt.title(fr"Residuals for ${pt_min} \leq p_T^{{\mathrm{{PPTruth}}}} \leq {pt_max}$ GeV")
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()

print(f"Cone: mean = {mu_cone:.2f} GeV, width = {sigma_cone:.2f} GeV")
print(f"BDT:  mean = {mu_bdt:.2f} GeV, width = {sigma_bdt:.2f} GeV")

