import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from tqdm import trange
from joblib import Parallel, delayed
import torch.nn.init as init
import numpy as np

class ZINBHeads(nn.Module):
    def __init__(self, q, p0, 
                 init_mu_bias=None, 
                 init_pi_bias=2.0, 
                 theta_init=None):
        """
        q      = latent dimension
        p0     = number of non-expressed genes
        init_mu_bias: (p0,) array for initializing b_mu (e.g. log-trimmed mean)
        init_pi_bias: scalar or (p0,) array for initializing b_pi (e.g. 2.0 => pi~0.88)
        theta_init:   (p0,) initial dispersions
        """
        super().__init__()
        # Mean head
        self.W_mu = nn.Parameter(torch.randn(q, p0) * 1e-2)
        self.b_mu = nn.Parameter(
            torch.as_tensor(init_mu_bias
                            if init_mu_bias is not None 
                            else torch.zeros(p0),
                            dtype=torch.float32).clone().detach()
        )
        # Zero‑inflation head
        self.W_pi = nn.Parameter(torch.randn(q, p0) * 1e-2)
        # self.b_pi = nn.Parameter(
        #     torch.ones(p0, dtype=torch.float32) * init_pi_bias
        # )
        self.b_pi = nn.Parameter(torch.full((1, p0), -2.0))
    
        if theta_init is not None:
            self.log_theta = nn.Parameter(torch.log(torch.tensor(theta_init, dtype=torch.float32)))
        else:
            self.log_theta = nn.Parameter(torch.zeros(p0))

        # Xavier initialization for W_mu and W_pi
        init.xavier_uniform_(self.W_mu)
        init.xavier_uniform_(self.W_pi)

    def forward(self, Z, size_factors):
        """
        Z:   (batch_size, q) latent vectors from Phase 1
        size_factors:   (batch_size,) size factors
        returns:
          mu:   (batch_size, p0) NB means
          pi:   (batch_size, p0) zero-inflation probs
          theta:       (p0,)     dispersions
        """
        # 1) predict mean logits
        Y_mu = Z @ self.W_mu + self.b_mu        # (B, p0)
        Y_mu = torch.clamp(Y_mu, min=-700)      # stability
        mu = size_factors.unsqueeze(1) * torch.exp(Y_mu)   # scale by size factor

        # 2) predict zero-inflation logits
        Y_pi = Z @ self.W_pi + self.b_pi        # (B, p0)
        pi = torch.sigmoid(Y_pi)                # in (0,1)

        # 3) dispersions
        theta = torch.exp(self.log_theta)       # (p0,)

        return mu, pi, theta
    
    

    def zinb_nll(self, k, mu, pi, theta, eps=1e-8):
        """
        Full ZINB negative log-likelihood.
        k:     raw counts (B, p0)
        mu:    predicted means  (B, p0)
        pi:    dropout probs    (B, p0)
        theta: dispersions      (p0,)
        """
        # broadcast theta
        theta_b = theta.unsqueeze(0)  # (1, p0) -> (B, p0)
        theta_b = theta_b.expand_as(k)  # Ensure shape matches k for masking
        # zero mask
        mask0 = (k == 0)
        # NB pmf at zero
        nb0 = (theta_b / (mu + theta_b + eps)).pow(theta_b)

        # loss for zeros
        loss0 = -torch.log(pi + (1 - pi) * nb0 + eps)[mask0].sum()

        # positives
        mask1 = ~mask0
        # -log(1-pi)
        loss1 = -torch.log(1 - pi + eps)[mask1].sum()
        # -log NB(k;mu,theta)
        k1     = k[mask1]
        mu1    = mu[mask1]
        th1    = theta_b[mask1]
        # NB log‐pmf
        log_nb = (
              torch.lgamma(k1 + th1)
            - torch.lgamma(th1)
            - torch.lgamma(k1 + 1)
            + k1    * torch.log(mu1    / (mu1    + th1) + eps)
            + th1   * torch.log(th1   / (mu1    + th1) + eps)
        )
        loss2 = -log_nb.sum()

        return loss0 + loss1 + loss2
    

    
    def fit(self, z, k, size_factors, n_epochs = 100, lr_mu=1e-2, lr_pi=1e-1, lr_theta=1e-2):
        threshold = 1e-5
        losses =[]
        prev_zinb = None
        
        # Separate parameter groups
        mu_params = [self.W_mu, self.b_mu]
        pi_params = [self.W_pi, self.b_pi]
        theta_params = [self.log_theta]

        optimizer_mu = torch.optim.Adam(mu_params, lr=lr_mu)
        optimizer_pi = torch.optim.Adam(pi_params, lr=lr_pi)
        optimizer_theta = torch.optim.Adam(theta_params, lr=lr_theta)
        # optimizer  = torch.optim.Adam(self.parameters(), lr=1e-1)        
        for epoch in trange(n_epochs, desc="Training"):
            # optimizer.zero_grad()
            optimizer_mu.zero_grad()
            optimizer_pi.zero_grad()
            optimizer_theta.zero_grad()

            mu, pi, theta = self.forward(z, size_factors)
            loss = self.zinb_nll(k, mu, pi, theta)
            loss.backward()

            optimizer_mu.step()
            optimizer_pi.step()
            optimizer_theta.step()
            # Clamp log_theta to keep theta in [0.01, 1000]
            with torch.no_grad():
                self.log_theta.clamp_(min=np.log(0.01), max=np.log(1000.0))
                
            losses.append(loss.detach().cpu())
            #print(f'Epoch {epoch}, NLL: {loss:.6f}')
            
            if prev_zinb is not None and abs(loss.item()-prev_zinb)<threshold:
                print(f"Converged at epoch {epoch} (ΔNLL < {threshold})")
                break
            
            prev_zinb = loss.item()
        
        return losses

