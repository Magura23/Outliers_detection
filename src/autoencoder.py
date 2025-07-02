import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from tqdm import trange
from joblib import Parallel, delayed


class EncoderExpressed(nn.Module):
    def __init__(self, input_dim, encoder_matrix,  latent_dim=16):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias = False)
        
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor(encoder_matrix, dtype=torch.float32))
            
    def forward(self, x):
        return self.encoder(x)
        
        
class DecoderExpressed(nn.Module):
    def __init__(self,  decoder_matrix, bias,  output_dim,  latent_dim = 16):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.decoder = nn.Linear(latent_dim, output_dim, bias = True )
        
        with torch.no_grad():
            self.decoder.weight.copy_(torch.tensor(decoder_matrix, dtype=torch.float32))
            if bias is not None:
                self.decoder.bias.copy_(torch.tensor(bias.squeeze(), dtype=torch.float32))
        
        
    def forward(self, latent):
        return self.decoder(latent)
        
        
class AutoencoderExpressed(nn.Module):
    def __init__(self, encoder, decoder, theta, size_factors):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.theta = torch.tensor(theta, dtype=torch.float32) # (n_genes, )

        self.size_factors = torch.tensor(size_factors, dtype=torch.float32) #(n_samples, )

    def forward(self, x):
            latent = self.encoder(x)
            decoded = self.decoder(latent)
            return decoded
        
    def get_mu(self, x):
            y = torch.clamp(x, min= -700, max = 700)
            return torch.exp(y)*self.size_factors.unsqueeze(1)
        
    def get_theta(self):
            return self.theta
        
    def get_latent(self, x):
            return self.encoder(x)
        
        
    @staticmethod
    def nb_nll_complete(k, mu, theta):
        eps = 1e-8
    
        log_prob = (
            torch.lgamma(k + theta)
            - torch.lgamma(theta)
            - torch.lgamma(k + 1)
            + theta * torch.log(theta + eps)
            + k * torch.log(mu + eps)
            - (k + theta) * torch.log(mu + theta + eps)
        )
        return -log_prob.mean()
            
            
        
        
    def update_encoder(self, x, k, theta):
    
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = True

    
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        encoder_optimizer.zero_grad()

        h = self.encoder(x)
        y = self.decoder(h) 
        mu = self.get_mu(y)
        loss = self.nb_nll_complete(k, mu, theta)
        loss.backward()
        encoder_optimizer.step()
            
            
    @staticmethod
    def fit_theta_j(k_j, mu_j, theta_init, min_theta, max_theta):
        k_j = torch.tensor(k_j, dtype=torch.float32)
        mu_j = torch.tensor(mu_j, dtype=torch.float32)
        def nll(th):
            th = torch.tensor(th, dtype=torch.float32)
            return AutoencoderExpressed.nb_nll_complete(k_j, mu_j, th).item()
        result = minimize(
            nll,
            x0=np.clip(theta_init, min_theta, max_theta),
            bounds=[(min_theta, max_theta)],
            method='L-BFGS-B'
        )
        return result.x[0]
   
    def update_theta(self, x, k, min_theta=0.01, max_theta=1000, n_jobs = 8):
       with torch.no_grad():
           h = self.encoder(x)
           y = self.decoder(h)
           mu = self.get_mu(y).cpu().numpy()
           k_np = k.cpu().numpy()
           theta_init = self.theta.cpu().numpy()
           n_genes = k_np.shape[1]
           
           theta_new = Parallel(n_jobs=n_jobs)(
               delayed(self.fit_theta_j)(k_np[:, j], mu[:, j], theta_init[j], min_theta, max_theta)
               for j in range(n_genes)
           )
           self.theta.copy_(torch.tensor(theta_new, dtype=torch.float32))
    
                
                
                
    def update_decoder(self, x, k, theta):
        for param in self.decoder.parameters():
                param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = False
                
        
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        decoder_optimizer.zero_grad()

        h = self.encoder(x).detach()  # freeze encoder
        y = self.decoder(h)
        mu = self.get_mu(y)
        loss = self.nb_nll_complete(k, mu, theta)
        loss.backward()
        decoder_optimizer.step()

        
    
            
    def fit(self, x, k, n_epochs=100):
            threshold = 1e-5
            losses =[]
            prev_nll = None
            for epoch in trange(n_epochs, desc="Training"):
                self.update_encoder(x, k, self.theta)
                self.update_decoder(x, k, self.theta)
                self.update_theta(x, k)
            
                mu = self.get_mu(self.decoder(self.encoder(x)))
                nll = self.nb_nll_complete(k, mu, self.theta)
                losses.append(nll)
                print(f'Epoch {epoch}, NLL: {nll:.6f}')
                
                if prev_nll is not None and abs(nll.item()-prev_nll)<threshold:
                    print(f"Converged at epoch {epoch} (ΔNLL < {threshold})")
                    break
                
                prev_nll = nll.item()
                
            return losses
                
                


                    
                    
            
            
        
            
            
    