import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LNN (nn.Module):
    def __init__(self, ncp_input_size, hidden_size, num_classes, sequence_length):
        super(LNN, self).__init__()

        self.hidden_size = hidden_size
        self.ncp_input_size = ncp_input_size
        self.sequence_length = sequence_length

        ### CNN HEAD
        self.conv1 =  nn.Conv2d(1,16,3)  # in channels, output channels, kernel size
        self.conv2 =  nn.Conv2d(16,32,3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 =  nn.Conv2d(32,64,5, padding=2, stride=2)
        self.conv4 =  nn.Conv2d(64,128,5, padding=2, stride = 2)
        self.bn4 = nn.BatchNorm2d(128)

        ### DESIGNED NCP architecture
        wiring = AutoNCP(hidden_size, num_classes)    # 234,034 parameters

        self.rnn = CfC(ncp_input_size, wiring)

        make_wiring_diagram(wiring, "kamada")
    
    def forward(self, x):
        # Forward pass through CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))  # Max-pooling with 2x2 window
    
        # Ensure x is on the same device as the model
        x = x.to(device)
    
        # Reshape output for RNN (LNN) processing
        x = x.view(-1, self.sequence_length, self.ncp_input_size)
        
        # Initial hidden state (zeros) on the same device
        h0 = torch.zeros(x.size(0), self.hidden_size).to(device)
    
        # Ensure h0 is on the same device as x before concatenation
        h0 = h0.to(device)
    
        # Forward pass through RNN (LNN) layer
        out, _ = self.rnn(x, h0)
        
        # Ensure out is on the same device as the model
        out = out.to(device)
        
        out = out[:, -1, :]  # Extract output from the last time step for classification
        return out


def make_wiring_diagram(wiring, layout):
    sns.set_style("white")
    plt.figure(figsize=(6, 6))
    legend_handles = wiring.draw_graph(layout=layout,neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()
