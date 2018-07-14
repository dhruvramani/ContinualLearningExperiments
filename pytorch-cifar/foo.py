import torch
from torch.autograd import Variable
import torch.nn.functional as F
from models.resnetlin import *

'''class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(20, 10)
        self.lin2 = torch.nn.Linear(10, 5)

    def forward(self, X):
        out = self.lin1(X)
        out = self.lin2(out)

        return out

model = Model()
x1 = Variable(torch.randn(50, 20), requires_grad=False)
#x2 = Variable(torch.randn(10, 20), requires_grad=False)

y1 = model(x1)
#y2 = model(x2)

print(list(model.lin1.parameters()))
#print(y2.data.numpy())'''

model = ResNet18()
old_param = list(model.parameters())
model = torch.nn.DataParallel(model)

print(old_param == list(model.parameters()))
