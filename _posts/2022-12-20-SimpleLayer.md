---
layout: post
title:  "Simple Layer!"
summary: "Simple Neural Network"
author: cupark
date: '2022-12-15 14:56:23 +0530'
category: NeuralNetwork
thumbnail: /assets/img/posts/SimpleLayer/thumb_simplelayer.png
keywords: SimpleLayer
permalink: /blog/SimpleLayer/
usemathjax: true
---

---
### 다중 레이어 구현
---
#### Input - First Layer

<p align="center"><img src="/assets/img/posts/SimpleLayer/input_layer.png"></p>

```python
import torch
import torch.nn as nn
import numpy as np
```
#### a1 = w11 * x1 + w12 * x2 +b1으로 나타낼 수 있다. 
#### A1 = a1, X = x1, x2 , W = w1, w2 , B = b 일 때 A1 = XW + B으로 구성할 수 있다.
```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print("X Shape: ", X.shape)    #(2,)
print("W1 Shape: ", W1.shape)  #(2,3)
print("B1 Shape: ", B1.shape)  #(3,)
A1 = np.dot(X,W1) + B1
print("A1 Shape: ", A1.shape) #(3,)
print("A1: ", A1)             # [0.3, 0.7, 1.1]
```
---
#### First Layer & Activation Function(Sigmoid)

<p align="center"><img src="/assets/img/posts/SimpleLayer/input_first_layer.png"></p>

---
#### Sigmoid()

<p align="center"><img src="/assets/img/posts/SimpleLayer/sigmoid.png"></p>

```python
class Sigmoid:
    def __init__(self): # 초기화
        self.out = None

    def forward(self, x): # 순전파
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout): # 역전파
        dx = dout * (1.0 - self.out) * self.out

        return dx

def sigmoid(x): # 시그모이드 식
    return 1 / (1 + np.exp(-x))    
sigmoid_func =  Sigmoid()
Z1 = sigmoid_func.forward(A1)
print("Z1 Shape: ", Z1.shape)  # (3,)
print("Z1: ", Z1)              # [0.57444252 0.66818777 0.75026011]
```
---
#### Fisrt-Second Layer

<p align="center"><img src="/assets/img/posts/SimpleLayer/first_second_layer.png"></p>

```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
print("A2 Shape: ", A2.shape)  # (2,)
print("A2 Shape: ", A2)        # [0.51615984 1.21402696]

Z2 = sigmoid_func.forward(A2)  
print("Z2 Shape: ", Z2.shape)  # (2,)
print("Z2 Shape: ", Z2)        # [0.62624937 0.7710107 ]
```
---
#### Second - Output Layer

<p align="center"><img src="/assets/img/posts/SimpleLayer/Second_ouput_layer.png"></p>

```python
def identify_fuction(x): # 항등함수
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
print("A3 Shape: ", A3.shape)  # (2,)
print("A3: ", A3)              # [0.31682708 0.69627909]

y = identify_fuction(A3)      # 출력값
print("y Shape: ", y.shape)  # (2,)
print("y: ", y)              # [0.31682708 0.69627909]
```
---
#### Network Code - Function
```python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['B2'] = np.array([0.1, 0.2])
    network['B3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    w1, w2, w3 = network['W1'],  network['W2'],  network['W3']
    b1, b2, b3 = network['B1'],  network['B2'],  network['B3']
    
    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2 
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = identify_fuction(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("y", y) # [0.31682708 0.69627909]
```
---
#### Network Code - Class
```python
class Simple_Network:
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.network['B1'] = np.array([0.1, 0.2, 0.3])
        self.network['B2'] = np.array([0.1, 0.2])
        self.network['B3'] = np.array([0.1, 0.2])

    def forward(self, x):
        self.w1, self.w2, self.w3 = self.network['W1'],  self.network['W2'],  self.network['W3']
        self.b1, self.b2, self.b3 = self.network['B1'],  self.network['B2'],  self.network['B3']
        
        self.a1 = np.dot(x,self.w1) + self.b1
        self.z1 = sigmoid(self.a1)
        self.a2 = np.dot(self.z1,self.w2) + self.b2 
        self.z2 = sigmoid(self.a2)
        self.a3 = np.dot(self.z2,self.w3) + self.b3
        self.y = identify_fuction(self.a3)
        return self.y

network = Simple_Network()
x = np.array([1.0, 0.5])
y = network.forward(x)
print("y", y) # [0.31682708 0.69627909]
```
