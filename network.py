from torch import nn

class SurrogateNetwork(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)  # add this
        self.layer4 = nn.Linear(128, 1)    # rename this
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x
