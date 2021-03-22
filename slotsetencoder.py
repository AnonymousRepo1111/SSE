import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotSetEncoder(nn.Module):
    def __init__(self, K, h, d, d_hat, eps=1e-8, _slots='Random', rtol=1e-4):
        super(SlotSetEncoder, self).__init__()
        self.K = K                                              #Number of Slots
        self.h = h                                              #Slot Size
        self.d = d                                              #Input Dimension
        self.d_hat = d_hat                                      #Linear Projection Dimension
        self.eps = eps                                          #Additive epsilon for stability
        self._slots = _slots                                    #Use Random or Deterministic Slots
        self.sqrt_d_hat = 1.0 / math.sqrt(d_hat)                #Normalization Term
        self.rtol = rtol                                        #Relative tolerance for tests

        if _slots == 'Random':
            self.sigma = nn.Parameter(torch.rand(1, 1, h))
            self.mu    = nn.Parameter(torch.rand(1, 1 ,h))
        elif _slots == 'Deterministic':
            self.S = nn.Parameter(torch.rand(1, K, h))
        else:
            raise ValueError('{} not implemented for slots'.format(_slots))
        
        self.k = nn.Linear(in_features=d, out_features=d_hat, bias=False)
        self.q = nn.Linear(in_features=h, out_features=d_hat, bias=False)
        self.v = nn.Linear(in_features=d, out_features=d_hat, bias=False)

        self.norm_slots = nn.LayerNorm(normalized_shape=h)
           
    def forward(self, X, S=None):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots Based on N
        if S is None:   #S \in R^{N x K xh}
            if self._slots == 'Random':
                S = torch.normal(self.mu.repeat(N, self.K, 1).to(device), torch.exp(self.sigma.repeat(N, self.K, 1).to(device)))
            else:
                #Deterministic slots
                S = self.S.repeat(N, 1, 1)
            
        S = self.norm_slots(S)

        #Linear Projections
        k = self.k(X)   #k \in R^{N x n x d_hat}
        v = self.v(X)   #v \in R^{N x n x d_hat}
        q = self.q(S)   #q \in R^{N x K x d_hat}
        
        #Compute M
        M = self.sqrt_d_hat * torch.bmm(k, q.transpose(1, 2)) #M \in R^{N x n x K}
        
        #Compute sigmoid attention
        attn = torch.sigmoid(M) + self.eps         #attn \in R^{N x n x K}
        
        #Compute attention weights
        W = attn / attn.sum(dim=2, keepdims=True)   #W \in R^{N x n x K}
        
        #Compute S_hat
        S_hat = torch.bmm(W.transpose(1, 2), v)     #S_hat \in R^{N x K x d_hat}
        return S_hat 

    def check_minibatch_consistency_random_slots(self, X, split_size):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat_X = self.forward(X=X, S=S)
        
        #Split X each with split_size elements.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)

        #Encode splits
        S_hat_split = torch.zeros(N, self.K, self.d_hat).to(device)
        for split_i in X:
            S_hat_split_i = self.forward(X=split_i, S=S)
            S_hat_split = S_hat_split + S_hat_split_i 
        
        consistency = torch.all(torch.isclose(S_hat_X, S_hat_split, rtol=self.rtol))
        print('Random  Slot Encoder is MiniBatch Consistent                     : ', consistency)
    
    def check_minibatch_consistency_deterministic_slots(self, X, split_size):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)

        #Encode full set
        S_hat_X = self.forward(X=X, S=S)
        
        #Split X each with split_size elements.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)

        #Encode splits
        S_hat_split = torch.zeros(N, self.K, self.d_hat).to(device)
        for split_i in X:
            S_hat_split_i = self.forward(X=split_i, S=S)
            S_hat_split = S_hat_split + S_hat_split_i 
        consistency = torch.all(torch.isclose(S_hat_X, S_hat_split, rtol=self.rtol))     
        print('Deterministic Slot Encoder is MiniBatch Consistent               : ', consistency)
    
    def check_input_invariance_random_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on X
        permutations = torch.randperm(n)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm, rtol=self.rtol))
        print('Random  Slot Encoder is Permutation Invariant w.r.t Input        : ', consistency)
    
    def check_input_invariance_deterministic_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on X
        permutations = torch.randperm(n)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm, rtol=self.rtol))
        print('Deterministic Slot Encoder is Permutation Invariant w.r.t Input  : ', consistency)
 
    def check_slot_equivariance_random_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on S
        permutations = torch.randperm(self.K)
        S = S[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)

        #Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm, rtol=self.rtol))
        print('Random  Slot Encoder is Permutation Equivariant w.r.t Slots      : ', consistency)
    
    def check_slot_equivariance_deterministic_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on S
        permutations = torch.randperm(self.K)
        S = S[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)

        #Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm, rtol=self.rtol))
        print('Deterministic Slot Encoder is Permutation Equivariant w.r.t Slots: ', consistency)
   

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #NOTE: To remove seeds, consider changing the value of rtol below.
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    
    N = 32                                  #Batch Size
    n = 5000                                #Set Size
    d = 3                                   #Element Dimension
    K = 16                                  #Number of Slots
    h =  32                                 #Slot Size
    d_hat = 32                              #Linear Projection Dimension
    split_size = 50                         #Number of elements in each split
    X = torch.rand(N, n, d).to(device)      #Input Set
    rtol = 1e-4                             #Precision of comparison. NOTE:If manual seeds are turned off, consider reducing rtol to 1e-3 or 1e-2.

    random_slot_encoder = SlotSetEncoder(K=K, h=h, d=d, d_hat=d_hat, _slots='Random').to(device)
    random_slot_encoder(X)
    random_slot_encoder.check_minibatch_consistency_random_slots(X, split_size=split_size)
    random_slot_encoder.check_input_invariance_random_slots(X)
    random_slot_encoder.check_slot_equivariance_random_slots(X)
    
    print()
    deterministic_slot_encoder = SlotSetEncoder(K=K, h=h, d=d, d_hat=d_hat, _slots='Deterministic').to(device)
    deterministic_slot_encoder(X)
    deterministic_slot_encoder.check_minibatch_consistency_deterministic_slots(X, split_size=split_size)
    deterministic_slot_encoder.check_input_invariance_deterministic_slots(X)
    deterministic_slot_encoder.check_slot_equivariance_deterministic_slots(X)
