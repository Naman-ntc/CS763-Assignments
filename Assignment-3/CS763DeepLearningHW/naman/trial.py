import torch

model = torch.load("model2.dat")

from generateTestResults import saveTestResults

saveTestResults(model)