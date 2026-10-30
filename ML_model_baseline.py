import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
file = uproot.open("/home/m26_an6685te/Downloads/pythia8317/examples/bdt_dataset.root")
tree = file["t"]
data = tree.arrays(["pt_jet", "pt_leading", "pt_soft_sum", "ppTruth","PbPbtruth"],
    library="np")
print(data)


X = np.column_stack([
    data["pt_jet"],
    data["pt_leading"],
    data["pt_soft_sum"]
])

y = data["PbPbtruth"]

X_train, X_test, y_train, y_test , ppTruth_train, ppTruth_test = train_test_split(
    X,
    y,
    data["ppTruth"],
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
#standardizing the data in this step because each input node has different scales which can cause problem in the ML training
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("X_train mean:", X_train.mean(axis=0))
print("X_train std:", X_train.std(axis=0))


#Next i have to translate this data into PyTorch tensors
import torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)
#i am not quite sure if i did it correctly but!!!
print("X_tarain:",X_train.shape, X_train.dtype)
print("y_train:",y_train.shape, y_train.dtype)
#when I run this I get [1471,3] for X_train mean we train 1471 training jets and each jet has 3 input featuires as I have adde (pt_jet, pt_leading,pt_soft_sum)
#you can make a new file and define different models there and then you can use them. 
#NOw I will define the MLP model using PyTorch. I will create a simple feedforward neural network with one hidden layer.
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3,8)
        self.fc2 = nn.Linear(8,1)

    def forward (self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #i used ReLU activation function. I also wanted to use Signoid. I am not quite sure if it makes sense or not!
model = SimpleMLP()
print(model)
#save this also in a different file.(or put it in the top)
criterion =nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#let's test the training, add the validatitionn inside this loop (put the model.eval inside the training loop, also dont name .no_grad, dont include eval because it will  freeze it.)
num_epochs = 1000
loss_history =[]
for epoch in range(num_epochs):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    #i wanna plot the loss vs. epoch so will add the foll9owing line
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch{epoch + 1}, Loss: {loss.item():.4}")

        #plotting loss vs. epoch
import matplotlib.pyplot as plt
plt.plot(range(1, num_epochs +1 ), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

#testing the reaming 20% data
model.eval()
with torch.no_grad():
    predictions = model(X_test)
print("Predictions shape:", predictions.shape)
print("Targets shape:", y_test.shape)
test_loss = criterion(predictions, y_test)
print("Test MSE:", test_loss.item())
#converting RMSE into GeV
test_rmse = torch.sqrt(test_loss)
print("Test RMSE:", test_rmse.item(), "GeV")


##########################################seperate  the training part

#checking the residuals to see if they are normally distributed
residuals = y_test - predictions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


dnn_residuals = residuals.detach().cpu().numpy().flatten()

# Keep all finite residuals
dnn_residuals = dnn_residuals[np.isfinite(dnn_residuals)]

print("Number of residuals:", len(dnn_residuals))
print("Minimum residual:", dnn_residuals.min(), "GeV")
print("Maximum residual:", dnn_residuals.max(), "GeV")
print("Mean residual:", dnn_residuals.mean(), "GeV")
print("Standard deviation:", dnn_residuals.std(), "GeV")


mu_dnn, sigma_dnn = norm.fit(dnn_residuals)

x = np.linspace(dnn_residuals.min(), dnn_residuals.max(), 1000)

plt.figure(figsize=(8, 6))
plt.hist(
    dnn_residuals,
    bins=60,
    density=True,
    histtype="step",
    linewidth=1.8,
    label="DNN residuals"
)

plt.plot(
    x,
    norm.pdf(x, mu_dnn, sigma_dnn),
    "--",
    label=f"Gaussian fit: mean={mu_dnn:.2f} GeV, sigma={sigma_dnn:.2f} GeV"
)

plt.axvline(0, color="black", linestyle=":")
plt.xlabel(r"$p_T^{\mathrm{PbPbtruth}}-p_T^{\mathrm{DNN}}$ [GeV]")
plt.ylabel("Normalized density")
plt.title("DNN Residual Distribution")
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


dnn_pred = predictions.detach().cpu().numpy().flatten()
dnn_truth = y_test.detach().cpu().numpy().flatten()

# Residual: true PbPb jet pT - DNN-predicted PbPb jet pT
dnn_residual = dnn_truth - dnn_pred

# Convert ppTruth_test to a NumPy array if necessary
pp_truth = np.asarray(ppTruth_test).flatten()


fig, ax = plt.subplots(figsize=(8, 7))

counts, xedges, yedges, im = ax.hist2d(
    pp_truth,
    dnn_residual,
    bins=50,
    range=[[0, 500], [-150, 250]],
    cmap="plasma",
    norm=colors.LogNorm(vmin=1)
)

ax.axhline(0, color="red", linestyle="--", alpha=0.7)

ax.set_xlabel(r"$p_T^{\mathrm{ppTruth}}$ [GeV]")
ax.set_ylabel(
    r"$p_T^{\mathrm{PbPbtruth}} - p_T^{\mathrm{DNN}}$ [GeV]"
)
ax.set_title("DNN residuals")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of jets per bin")

plt.tight_layout()
plt.show()


from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Selecting the same ppTruth interval as in your BDT figure
pt_min, pt_max = 50, 110

mask_dnn = (
    (pp_truth >= pt_min) &
    (pp_truth <= pt_max)
)

dnn_residual_selected = dnn_residual[mask_dnn]

# Remove non-finite values
dnn_residual_selected = dnn_residual_selected[
    np.isfinite(dnn_residual_selected)
]

print("Number of selected DNN jets:", len(dnn_residual_selected))
print("Mean residual:", np.mean(dnn_residual_selected), "GeV")
print("Residual standard deviation:", np.std(dnn_residual_selected), "GeV")

# Fit Gaussian to the central residual region
fit_min = -5
fit_max = 5

dnn_peak = dnn_residual_selected[
    (dnn_residual_selected >= fit_min) &
    (dnn_residual_selected <= fit_max)
]

mu_dnn, sigma_dnn = norm.fit(dnn_peak)

print(f"DNN Gaussian: mean = {mu_dnn:.2f} GeV, "
      f"sigma = {sigma_dnn:.2f} GeV")

# Plot settings
x_min, x_max = -30, 30
y_min, y_max = 0, 0.6
bins = np.linspace(x_min, x_max, 61)
x = np.linspace(x_min, x_max, 1000)

plt.figure(figsize=(8, 6))


plt.hist(
    dnn_residual_selected,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    label="DNN"
)

# Gaussian fits
plt.plot(
    x,
    norm.pdf(x, mu_dnn, sigma_dnn),
    "--",
    linewidth=1.8,
    label=fr"DNN Gaussian: $\mu={mu_dnn:.1f}$, "
          fr"$\sigma={sigma_dnn:.1f}$ GeV"
)

plt.axvline(0, color="black", linestyle=":", alpha=0.8)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel(r"Residual $\delta p_T$ [GeV]")
plt.ylabel(r"Probability density [GeV$^{-1}$]")
plt.title(
    fr"DNN residuals for "
    fr"${pt_min} \leq p_T^{{\mathrm{{ppTruth}}}} "
    fr"\leq {pt_max}$ GeV"
)

plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Settings
pt_min, pt_max = 50, 110
fit_min, fit_max = -5, 5

#DNNresiduals 
pp_truth_test = np.asarray(ppTruth_test).flatten()
dnn_pred = predictions.detach().cpu().numpy().flatten()
dnn_truth = y_test.detach().cpu().numpy().flatten()

dnn_residual = dnn_truth - dnn_pred

mask_dnn = (
    (pp_truth_test >= pt_min) &
    (pp_truth_test <= pt_max)
)

dnn_selected = dnn_residual[mask_dnn]
dnn_selected = dnn_selected[np.isfinite(dnn_selected)]

#Cone subtraction
# Cone prediction = ppTruth - soft contribution
pt_cone = data["ppTruth"] - data["pt_soft_sum"]

# Residual = true PbPb pT - cone prediction
cone_residual = data["PbPbtruth"] - pt_cone

mask_cone = (
    (data["ppTruth"] >= pt_min) &
    (data["ppTruth"] <= pt_max)
)

cone_selected = cone_residual[mask_cone]
cone_selected = cone_selected[np.isfinite(cone_selected)]

#Gaussian fits
def fit_central_gaussian(residuals):
    peak = residuals[
        (residuals >= fit_min) &
        (residuals <= fit_max)
    ]
    if len(peak) < 2:
        raise ValueError(
            "Too few residuals in the Gaussian fit range."
        )
    return norm.fit(peak)

mu_dnn, sigma_dnn = fit_central_gaussian(dnn_selected)
mu_cone, sigma_cone = fit_central_gaussian(cone_selected)

print(f"DNN:  mean = {mu_dnn:.2f} GeV, "
      f"sigma = {sigma_dnn:.2f} GeV")
print(f"Cone: mean = {mu_cone:.2f} GeV, "
      f"sigma = {sigma_cone:.2f} GeV")

print("Selected DNN jets:", len(dnn_selected))
print("Selected cone jets:", len(cone_selected))



