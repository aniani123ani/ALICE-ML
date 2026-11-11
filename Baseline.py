import uproot
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
file = uproot.open("/home/m26_an6685te/Downloads/pythia8317/examples/ml_dataset.root")
tree = file["t"]
data= tree.arrays(["pt_jet", "pt_leading", "pt_soft_sum", "ppTruth","PbPbtruth"],
    library="np")
from Models import SimpleMLP
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#Defining the input and otput variables for the MLP model. I will use pt_jet, pt_leading, and avg_sums as input features and PbPbtruth as the target variable.
X=np.column_stack([
    data["pt_jet"],
    data["pt_leading"],
    data["pt_soft_sum"]
])
y = data["PbPbtruth"]
pp_truth=data["ppTruth"]

X_train, X_temp, y_train,y_temp,pp_train,pp_temp= train_test_split(
    X,
    y,
    pp_truth,
    test_size=0.20,
    random_state=42
)
X_val, X_test, y_val, y_test,pp_val,pp_truth_test =train_test_split(    
    X_temp,
    y_temp,
    pp_temp,
    test_size=0.25,
    random_state=42
)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_val= StandardScaler().fit_transform(X_val)


X_train =torch.tensor(X_train, dtype=torch.float32)
X_val=torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
y_val = torch.tensor(y_val,dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)


###Training
train_loss_history=[]
val_loss_history = []
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    train_predictions = model(X_train)
    train_loss = criterion(train_predictions, y_train)
    train_loss.backward()
    optimizer.step()
    train_loss_history.append(train_loss.item())
    
    #validation
    import torch
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)
        val_loss_history.append(val_loss.item())
    if (epoch+1)%1000==0:
        print(f"Epoch{epoch+1},"
              f"Training Loss: {train_loss.item():.4}," 
              f"Validation Loss: {val_loss.item():.4}")
np.savez(
    "loss_history.npz",
    train_loss_history=train_loss_history,
    val_loss_history=val_loss_history
)

#testing on the remaining 5% of the data at the end
model.eval()
with torch.no_grad():
    predictions = model(X_test)
test_loss = criterion(predictions, y_test)
test_rmse = torch.sqrt(test_loss)
print("Test MSE:",test_loss.item())
print("Test RMSE",test_rmse.item())

dnn_pred = predictions.detach().cpu().numpy().flatten()
dnn_truth=y_test.detach().cpu().numpy().flatten()
np.savez(
    "dnn_plot_data.npz",
    dnn_pred=dnn_pred,
    dnn_truth=dnn_truth,
    pp_truth_test=pp_truth_test
)

######### MODEL "2" VAE
from Models import VAE, AE, loss_function
vae_model = VAE()
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.01)
train_loss_history=[]
val_loss_history = []
train_reconstruction_loss_history=[]
train_prediction_history=[]
train_KLD_history=[]

#Training loop for VAE
for epoch in range(1000):
    vae_model.train()
    vae_optimizer.zero_grad()
    encoded, decoded, mu, log_var, prediction = vae_model(X_train)
    train_loss = loss_function(decoded, X_train, mu, log_var, prediction, y_train)
    train_loss.backward()
    vae_optimizer.step()
    train_loss_history.append(train_loss.item())
    
    #validation
    vae_model.eval()
    with torch.no_grad():
        encoded_val, decoded_val, mu_val, log_var_val, prediction_val = vae_model(X_val)
        val_loss = loss_function(decoded_val, X_val, mu_val, log_var_val, prediction_val, y_val)
        val_loss_history.append(val_loss.item())
    
    if (epoch+1)%1000==0:
        print(f"Epoch {epoch+1},"
              f"Training Loss: {train_loss.item():.4},"
              f"Validation Loss: {val_loss.item():.4}")

vae_model.eval()
with torch.no_grad():
    encoded_test, decoded_test, mu_test, log_var_test, prediction_test = vae_model(X_test)
    test_loss = loss_function(decoded_test, X_test, mu_test, log_var_test, prediction_test, y_test)
    
    print("VAE Test Loss:", test_loss.item())

    test_prediction_loss=nn.MSELoss()(predictions, y_test)
    test_rmse = torch.sqrt(test_prediction_loss)
print("VAE Test Prediction Loss:", test_prediction_loss.item())
print("VAE Test Prediction RMSE:", test_rmse.item())
vae_pred = predictions.cpu().numpy().flatten()
vae_truth = y_test.cpu().numpy().flatten()
np.savez(
    "vae_plot_data.npz",
    vae_pred=vae_pred,
    vae_truth=vae_truth,
    pp_truth_test=pp_truth_test
)

###### MODEL 3

###### MODEL 3 Convolutional Neural Network (CNN)
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

X_train_cnn = X_train.unsqueeze(1)
X_val_cnn = X_val.unsqueeze(1)
X_test_cnn = X_test.unsqueeze(1)
print("X_train_cnn shape:", X_train_cnn.shape)
print("X_val_cnn shape:", X_val_cnn.shape)
print("X_test_cnn shape:", X_test_cnn.shape)


batch_size = 60

train_dataset = TensorDataset(X_train_cnn, y_train)
val_dataset = TensorDataset(X_val_cnn, y_val)
test_dataset = TensorDataset(X_test_cnn, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Train loader length:", len(train_loader))
print("Validation loader length:", len(val_loader))
print("Test loader length:", len(test_loader))

data, target = next(iter(train_loader))
print("Data shape:", data.shape)
print("Target shape:", target.shape)

from Models import CNN
model=CNN(in_channels=1, num_outputs=1)
sample_data, sample_target = next(iter(train_loader))
sample_output = model(sample_data)
print("output shape:", sample_output.shape)
print("target shape:", sample_target.shape)
print("input output:", sample_data.shape)

###Training the CNN model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss += train_loss / len(train_loader)

    #Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            predictions = model(data)
            loss = criterion(predictions, target)
            val_loss += loss.item()
    val_loss += val_loss / len(val_loader)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
np.savez(
    "cnn_loss_history.npz",
    train_loss_history=train_loss_history,
    val_loss_history=val_loss_history
)



### Evaluating the Model
model.eval()
predictions_list = []
target_list = []
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        predictions_list.extend(outputs.cpu().numpy().flatten())
        target_list.extend(target.cpu().numpy().flatten())

predictions = np.array(predictions_list)
targets_list = np.array(target_list)
test_mse = np.mean((predictions - targets_list) ** 2)
test_rmse = np.sqrt(test_mse)
print("CNN Test MSE:", test_mse)
print("CNN Test RMSE:", test_rmse)        


baseline_prediction=torch.mean(y_train).item()
baseline_predictions = np.full_like(targets_list, baseline_prediction)
baseline_mse = np.mean((targets_list - baseline_predictions) ** 2)
baseline_rmse = np.sqrt(baseline_mse)
print("Baseline Test MSE:", baseline_mse)

cnn_truth = targets_list
cnn_pred = predictions_list
np.savez(
    "cnn_plot_data.npz",
    cnn_pred=cnn_pred,
    cnn_truth=cnn_truth,
    pp_truth_test=pp_truth_test
)





