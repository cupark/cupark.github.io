---
layout: post
title:  "BottleNeck"
summary: "BottleNeck Parameters"
author: cupark
date: '2023-1-9 10:56:23 +0530'
category: NeuralNetwork
thumbnail: /assets/img/posts/BottleNeck/thumb_bottleneck.png
keywords: BottleNeck
permalink: /blog/BottleNeck/
usemathjax: true
---

---
### BottleNeck
---
#### BottleNeck이 Convolution 연산량에 어떠한 영향을 미치는가 확인
---

<p align="center"><img src="/assets/img/posts/BottleNeck/bottleneck.png"></p>

---
##### Import Lib
```python
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
```

---
##### Standard Param Network
```python
class Standard_Param(nn.Module):
    def __init__(self):
        super(Standard_Param, self).__init__()
        
        self.Standard_Param = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        fx = self.Standard_Param(x)
        out = fx + x
        out = self.relu(out)
        return out
```
---
##### BottleNeck Param Network
```python
class BottleNeck_Param(nn.Module):
    def __init__(self, inputdim = 1, outputdim = 64):
        super(BottleNeck_Param, self).__init__()
        
        self.BottleNeck_Param = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=False),            
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        fx = self.BottleNeck_Param(x)
        out = fx + x
        out = self.relu(out)
        return out
```     

---
##### Select Device  
```python
device = torch.device('cpu')
```

---
##### Calculate Param  
```python
if __name__ == "__main__":
    model1 = Standard_Param()
    model2 = BottleNeck_Param()
    summary(model=model1, input_size= (256,320,320), device= device.type) # Total params: 294,912
    summary(model=model2, input_size= (256,320,320), device= device.type) # Total params: 69,632
```

<p align="center"><img src="/assets/img/posts/BottleNeck/result.png"></p>
