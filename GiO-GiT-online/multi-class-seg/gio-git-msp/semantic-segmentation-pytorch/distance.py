# -----------------------------------------
# Project: 'GiO-GiT' 
# Written by Jie Hong (jie.hong@anu.edu.au)
# -----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class Logits(nn.Module):                    
      def __init__(self):
          super(Logits, self).__init__()

      def forward(self, out_s, out_t):
          loss = F.mse_loss(out_s, out_t)
                       
          return loss


class SoftTarget(nn.Module):       
      def __init__(self, T):
          super(SoftTarget, self).__init__()
          self.T = T
           
      def forward(self, out_s, out_t):
          loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
          F.softmax(out_t/self.T, dim=1), reduction='batchmean') * self.T * self.T
               
          return loss
