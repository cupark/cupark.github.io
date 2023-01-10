---
layout: post
title:  "Conv-Param"
summary: "Calculate Conv Param"
author: cupark
date: '2023-1-6 05:56:23 +0530'
category: NeuralNetwork
thumbnail: /assets/img/posts/ConvParam/thumb_convolution.png
keywords: ConvParam
permalink: /blog/ConvParam/
usemathjax: true
---

---

### **Convolution caculate Parameters**  

---

> #### 합성곱 연산에 따른 파라미터 계산  
>> - **Convolution Parameter**  
>> **1. Input Data Height : H**  
>> **2. Input Data Width : W**  
>> **3. Filter Height: FH**  
>> **4. Filter Width : WD**  
>> **5. Stride : S**  
>> **6. Padding : P**    
<p align="center"><img src="/assets/img/posts/ConvParam/outputsize.gif"></p>
<p align="center"><img src="/assets/img/posts/ConvParam/formula.png"></p>

---

> **Convolution Parameter Fomula**  
>> **InputChannel x KernelWidth x KernelHeight x OutputChannel + Bias(Filter)**   
>> **(Output Channel은 Filter의 값을 의미한다.)**   

 
---

> **BatchNormalization Parameter Configuration**   
>> **1. Gamma: Scaling Parameter**  
>> **2. Beta: Shift Parameter**  
>> **3. Mean: Non-Trainable Params**  
>> **4. Standard deviation: Non-Trainable Params**  


 
---

### **Example Calculate Conv Param on AlexNet**  

<p align="center"><img src="/assets/img/posts/ConvParam/alexnet_architecture.PNG"></p>

---

>### **AlexNet Architecture**  
>>- #### **Input: 227x227x3 크기의 컬러 이미지.**  
>> **1.Conv-1: 11x11 크기의 커널 96개, stride=4, padding=0**   
>> **2.MaxPool-1: stride 2, 3x3 max pooling layer**  
>> **3.Conv-2: 5x5 크기의 커널 256개, stride=1, padding=2**  
>> **4.MaxPool-2: stride 2, 3x3 max pooling layer**  
>> **5.Conv-3: 3x3 크기의 커널 384개, stride=1, padding=1**  
>> **6.Conv-4: 3x3 크기의 커널 384개, stride=1, padding=1**  
>> **7.Conv-5: 3x3 크기의 커널 256개, stride=1, padding=1**  
>> **8.Maxpool-3: stride 2, 3x3 max pooling layer**  
>> **9.FC-1: 4096개의 fully connected layer**  
>> **10.FC-2: 4096개의 fully connected layer**  
>> **11.FC-3: 1000개의 fully connected layer** 

---

>### **AlexNet Architecture Code**  

---

##### Import Lib  

```python
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
```

---

##### Class Define  
###### 논문에 W, H의 SIZE가 224로 잘못 표기 되어있어 227로 수정하여 사용한다.  
###### 파라미터의 계산 편의를 위하여 Relu와 LRN, Dropout은 제외하여 설계하였다. (전체버전은 Github에 업로드)  

```python
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=55, kernel_size=11, stride=4, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=55, out_channels=256, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1,bias=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1,bias=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1,bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.FC = nn.Sequential(
            nn.Linear(in_features= 9216, out_features= 4096),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.Linear(in_features= 4096, out_features= 1000),            
        )
        
        self.Init_Bias()
    
    def Init_Bias(self):
        for layer in self.Layer:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std= 0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.Layer[2].bias, 1)
        nn.init.constant_(self.Layer[4].bias, 1)
        nn.init.constant_(self.Layer[5].bias, 1)
    
    def forward(self, x):
        x = self.Layer(x)
        x = x.view(-1,256 * 6 *6)
        out = self.FC(x)
        return out
```

---
 
##### Main  

```python
device = torch.device('cpu')

if __name__ == "__main__":
    model1 = AlexNet()
    summary(model=model1, input_size= (3,227,227), device= device.type)
```

<p align="center"><img src="/assets/img/posts/ConvParam/result.PNG"></p>

