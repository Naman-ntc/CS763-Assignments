import torch
from Model import *
from readData import *
import numpy as np


model = Model(-1,128,268,200,1)
model.forward(data[0])