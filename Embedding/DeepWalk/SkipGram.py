import torch

class SimpleSkipGram(torch.nn.Module):
    def __init__(self) -> None:
        super(SimpleSkipGram, self).__init__()
        