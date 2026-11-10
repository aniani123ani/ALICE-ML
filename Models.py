######First MODEL
import torch
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
    ############SECOND MODEL VAEs

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_hidden = 8
        self.encoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
class VAE(AE):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        self.prediction=nn.Linear(self.num_hidden, 1)
    def reparameterize(self, mu, log_var):   
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        encoded = self.encoder(x)  # Log variance
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        prediction=self.prediction(z)
        return encoded,decoded, mu, log_var, prediction
    
def loss_function(recon_x, x, mu, log_var, prediction, y):

    # Reconstruction loss
    reconstruction_loss = nn.MSELoss(reduction="sum")(recon_x, x)
    prediction_loss = nn.MSELoss(reduction="sum")(prediction, y)

    # KL divergence
    KLD = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )
    beta = 1.0 
    lambda_pred = 1.0  
    total_loss = (
        lambda_pred * prediction_loss 
        + beta * KLD 
        + reconstruction_loss
    )
    return total_loss

    ###########Third Model CNNs


    ###########Third Model CNNs
