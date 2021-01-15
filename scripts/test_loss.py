import torch
import torch.nn as nn
import torch.nn.functional as F
a = torch.tensor([1., 2, 3, 4])
b = torch.tensor([4., 5, 6, 7])
loss_fn1 = nn.L1Loss(reduction='mean')
loss = loss_fn1(a, b)
print(loss)

loss_fn2 = nn.L1Loss(reduction='sum')
loss = loss_fn2(a, b)
print(loss)

loss_fn3 = nn.L1Loss(reduction='none')
loss = loss_fn3(a, b)
print(loss)