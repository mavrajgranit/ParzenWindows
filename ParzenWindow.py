import torch, math
import torch.nn as nn

class ParzenWindow(nn.Module):

    def __init__(self, in_features, out_features):
        super(ParzenWindow, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mikro = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        #invsqr = 1 / math.sqrt(self.in_features)
        w_range = 1.0#invsqr
        self.mikro.data.uniform_(-w_range, w_range)
        self.sigma.data.uniform_(w_range)#normal_()#uniform_(mu_range)

    def forward(self, input):
        sum = torch.sum((self.mikro-input) ** 2,1)
        return torch.exp(-sum/ (2*self.sigma ** 2))