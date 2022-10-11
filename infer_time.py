
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity


torch.__version__
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10000 # all the test set

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

# measure time
inputs,labels = next(iter(test_loader))

class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x

    def num_flat_features(self, x):
            size = x.size()[1:]
            return np.prod(size)


model = LeNet().to(DEVICE)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs.cuda())

# print(type(prof.key_averages()))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

#prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                         model_inference         0.03%     422.000us        99.84%        1.562s        1.562s       0.000us         0.00%       7.487ms       7.487ms             1  
#                                                aten::to         0.03%     452.000us         0.26%       3.990ms       3.990ms       0.000us         0.00%       3.221ms       3.221ms             1  
#                                          aten::_to_copy         0.00%      11.000us         0.23%       3.538ms       3.538ms       0.000us         0.00%       3.221ms       3.221ms             1  
#                                             aten::copy_         0.00%      16.000us         0.21%       3.355ms       3.355ms       3.221ms        43.02%       3.221ms       3.221ms             1  
#                        Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       3.221ms        43.02%       3.221ms       3.221ms             1  
#                                            aten::conv2d         0.00%      11.000us        99.50%        1.556s     778.249ms       0.000us         0.00%       3.003ms       1.502ms             2  
#                                       aten::convolution         0.00%      32.000us        99.50%        1.556s     778.243ms       0.000us         0.00%       3.003ms       1.502ms             2  
#                                      aten::_convolution         0.00%      50.000us        99.49%        1.556s     778.227ms       0.000us         0.00%       3.003ms       1.502ms             2  
#                                 aten::cudnn_convolution         1.54%      24.089ms        99.49%        1.556s     778.159ms       2.403ms        32.10%       2.403ms       1.202ms             2  
# void implicit_convolve_sgemm<float, float, 1024, 5, ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.574ms        21.02%       1.574ms       1.574ms             1  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 1.564s
# Self CUDA time total: 7.487ms