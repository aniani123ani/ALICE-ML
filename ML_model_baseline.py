#importing uproot in order to read the root file and convert it into numpy array
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
file = uproot.open("/home/m26_an6685te/Downloads/pythia8317/examples/pp_ml_N20.root")
tree = file["t"]
data = tree.arrays(library="np")
print(data)
#i want to undertand the scale of my data here
for name, values in data.items():
    print(name)
    print("  min:", values.min())
    print("  max:", values.max())
    print("  mean:", values.mean())
    print("  std:", values.std())
#adding the first inpput. I chose pt_jet(truth), pt_leading,pt_soft_sum and let's see pt_smeared as an output-
#So, basically I am defining 3 nodes in this very simple MLP model.
X = np.column_stack([
    data["pt_jet"],
    data["pt_leading"],
    data["pt_soft_sum"]
])

y = data["pt_smeared"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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
#based on my study for regression model we use Means Squred Error
criterion =nn.MSELoss()
# And Optimizer/learining rate is lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#let's test the training
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
#import matplotlib.pyplot as plt
#plt.plot(range(1, num_epochs +1 ), loss_history)
#plt.xlabel("Epoch")
#plt.ylabel("Loss")
#plt.title("Training Loss")
#plt.show()

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


#after this run I see  that the training and test losses are quite close to eachotehr. we have to better train the model
#they should be reasonably close to each other. not too close or exact to avoid overfitting.



# Now, i want to see the visualization of what the model actually predicted.
import matplotlib.pyplot as plt
#plt.scatter(y_test.numpy(), predictions.numpy())
#plt.xlabel("Actual pt_pt_smeared [GEV]")
#plt.ylabel("Predicted pt_smeared [GeV]")
#plt.title("Actual vs. Predicted pt_smeared")
#plt.show()


#checking the residuals to see if they are normally distributed
residuals = predictions - y_test
print("Mean residual:", residuals.mean().item(), "GeV")
print("Std residual:", residuals.std().item(), "GeV")
plt.hist(residuals.numpy().flatten(), bins=30)
plt.xlabel("Predictions error [GeV]")
plt.ylabel("Number of jets")
plt.title("MLP Residuals")
plt.show()
# the results her show that the Model is under predicting the pt_smeared real data and I dont know how to fix it. the difference is like minus 50. the Mead residual is minus 47 and the Std residual 8.849