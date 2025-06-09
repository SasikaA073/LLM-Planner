import torch
import torch.nn as nn
import math
import torch.nn.functional as F





# # sasika edited this best v4; 
# class DCTAdapter(nn.Module):
#     def __init__(self, in_features=4096, num_components=24, tau=1.0):
#         super().__init__()
#         self.tau = tau
   
#         self.adapter_down = nn.Linear(in_features, 18, bias=False) # 18 best
#         self.adapter_up = nn.Linear(18, in_features, bias=False) # 18 best


#     def dct1(self, x):  # same as before
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
#         return x @ dct_mat.T

#     def idct1(self, x):
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
#         return x @ dct_mat

#     def create_dct_matrix(self, N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct

#     def forward(self, hidden_states):
#         print("From inside the DCT Adapter")
#         print("This is the hidden states shape:", hidden_states.shape)
#         dct = self.dct1(hidden_states)  # [B, T, C]

#         z = dct.reshape(-1, dct.shape[-1])  # [B*T, C]
#         z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
#         out = z_pert.view_as(dct)
#         idct = self.idct1(out)
#         return hidden_states + idct # 32.00
    


#         # return idct # 20.00
# #         # return hidden_states + (1/10)*idct  # residual connection
#         # return 0.5*hidden_states + 0.5*idct # 30.00


# # max pool -> low dimension 
# # dconv layer -> up sample
# # reduce r in lora so that the params equal to our number of params



#best v4
class DCTAdapter(nn.Module):
    def __init__(self, in_features=768, num_components=24, tau=1.0):
        super().__init__()
        self.tau = tau
        self.adapter_gate_logits = nn.Parameter(torch.randn(in_features))  # One per DCT dim
        self.adapter_down = nn.Linear(in_features, 18, bias=False) # 18 best
        self.adapter_up = nn.Linear(18, in_features, bias=False) # 18 best

    def dct1(self, x):  # same as before
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat.T

    def idct1(self, x):
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat

    def create_dct_matrix(self, N, device=None, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / N)
        return dct

    def forward(self, hidden_states):
        dct = self.dct1(hidden_states)  # [B, T, C]

        z = dct.reshape(-1, dct.shape[-1])  # [B*T, C]
        z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
        out = z_pert.view_as(dct)
        idct = self.idct1(out)
        return hidden_states + idct  # residual connection