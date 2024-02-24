import torch
from torchvision.models import resnet50

student = resnet50()
teacher = resnet50()

for param in student.parameters():
    param.data.fill_(0)
for param in teacher.parameters():
    param.data.fill_(1)

m = 0.999

for s_param, t_param in zip(student.parameters(), teacher.parameters()):
  
    assert s_param.shape == t_param.shape, "The shape is not compatible!"
    t_param.data = (1-m) * s_param.data + m * t_param.data
