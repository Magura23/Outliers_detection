import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from tqdm import trange
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
        
        
    def nb_nll(self, k, mu, theta):
            """
            Negative log-likelihood for negative binomial distribution.
            Parameters
            ----------
            k : torch.Tensor
                The number of successes (observed counts).
            mu : torch.Tensor
                The expected mean (fitted values).
            theta : torch.Tensor
                The dispersion parameter.
            """
            eps = 1e-8
            t1 = k * torch.log(mu+eps)
            t2 = (k*theta)*torch.log(mu+theta+eps)
            
            nll = -(t1-t2)
            
            return nll.mean()
        
        
        
    def nb_nll_theta(self, k, mu, theta):
            eps = 1e-8
            t1 = theta*torch.log(theta+eps)
            t2 = (k+theta)*torch.log(mu+theta+eps)
            t3 = torch.lgamma(k+theta)
            t4 = torch.lgamma(theta+1)
            
            nll = -t1+t2+-t3+t4
            return nll.mean()
            
            
        
        
    def update_encoder(self, x, k, theta):
        
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
                
            encoder_optimizer = torch.optim.LBFGS(list(self.encoder.parameters()))
            
            def closure():
                encoder_optimizer.zero_grad()
                h = self.encoder(x)
                y = self.decoder(h)
                mu = self.get_mu(y)
                loss = self.nb_nll(k, mu, theta)
                loss.backward()
                return loss
            
            encoder_optimizer.step(closure)
            
            
    def update_decoder(self, x, k, theta):
        
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True
                
            decoder_optimizer = torch.optim.LBFGS(list(self.decoder.parameters()))
            
            def closure():
                decoder_optimizer.zero_grad()
                h = self.encoder(x).detach()  # freeze encoder
                y = self.decoder(h)
                mu = self.get_mu(y)
                loss = self.nb_nll(k, mu, theta)
                loss.backward()
                return loss
            
            decoder_optimizer.step(closure)
            
    def update_theta(self, x, k, min_theta=0.01, max_theta=1000):
            with torch.no_grad():
                h = self.encoder(x)
                y = self.decoder(h)
                mu = self.get_mu(y)
                n_genes = k.shape[1]
                theta_new = self.theta.clone()
                for j in range(n_genes):
                    k_j = k[:, j]
                    mu_j = mu[:, j]
                    
                    result = minimize(
                        lambda th: self.nb_nll_theta(k_j, mu_j, torch.tensor(th, dtype=torch.float32)),
                        x0 = np.clip(self.theta[j].item(), min_theta, max_theta),
                        bounds=[(min_theta, max_theta)],
                        method='L-BFGS-B'
                    )
                    
                    theta_new[j] = result.x[0]
                
                self.theta.copy_(theta_new)
                
    def fit(self, x, k, n_epochs=100):
            losses =[]
            for epoch in trange(n_epochs, desc="Training"):
                self.update_encoder(x, k, self.theta)
                self.update_decoder(x, k, self.theta)
                self.update_theta(x, k)
            
                mu = self.get_mu(self.decoder(self.encoder(x)))
                nll = self.nb_nll(k, mu, self.theta)
                losses.append(nll)
                print(f'Epoch {epoch}, NLL: {nll:.4f}')
                
            return losses
                
                


                    
                    
            
            
        
            
            
    