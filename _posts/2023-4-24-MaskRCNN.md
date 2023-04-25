---
layout: post
title:  "Mask-RCNN"
summary: "Mask-RCNN"
author: cupark
date: '2023-4-23 09:00:23 +0530'
category: NeuralNetwork
thumbnail: /assets/img/posts/MaskRCNN/thumb_maskrcnn.png
keywords: Masking
permalink: /blog/Mask-RCNN/
usemathjax: true
---


---
### **Mask RCNN**     
---
> #### About       
>>> **1. Backbone Network**     
>>>> - **ResNet, Feature Pyramid Network(FPN) 같은 기존 이미지 분류 네트워크를 통한 입력 이미지의 Feature 추출**       
>>>> - **Feature Extraction에서 Small Size Feature map은 작은 물체에 대한 검출이 용이하다.**       
>>>> - **Feature Extraction에서 Big Size Feature map은 큰 물체에 대한 검출이 용이하다.**       
>>>> - **Multi Scale Feature Pyramid Bottom-Up & Top-Down**       
>>>> - **Bottom-Up: 입력 이미지에 대한 저수준 정보를 추출하여 Feature Map 생성**       
>>>> - **Top-Down: 생성된 Feature map을 Up-Scaling하여 높은 Resolution에 대한 Feature map 생성**       
>>>> **∴ 결과적으로 Feature는 Scalable 할 수록 특성 추출에 용이하다.**     

>>> **2. RPN (Region Proposal Network)**       
>>>> - **ROI 영역내에 관심 객체 존재 가능성을 추정한다.**     


>>> **3. ROI Align**       
>>>> - **ROI Pooling을 사용하지 않고 ROI Align을 수행하는 이유.**     


>>> **4. RCNN (Region-based Convolution Neural Network**     
>>>> - **RPN에서 예측한 ROI을 사용하여 물체의 클래스를 예측한다.**     
>>>> - **RCNN은 ROI라는 고정된 크기에 대하여 Feature map을 추출하여 Fully Connected Layer로 확률값을 추정한다.**     

>>> **5. Mask Branch**     
>>>> - **ROI에 대한 Prediction을 수행한 RCNN에 대하여 Instance Segmentation을 수행한다.**     
>>>> - **여기서 기존 ROI Pooling 2차 대신, ROI Align을 사용한다.**     
>>>> - **ROI Align으로 추출된 ROI Feature에 대하여 1) Convolutional layer, 2) Deconvolutional layer를 거쳐 Binary Classification을 수행한다.**     


<p align="center"><img src="/assets/img/posts/MaskRCNN/Segmentation.PNG"></p>

---
#### About  Segmentation       
> **Semantic Segmentation vs Instance Segmentation**           
>> **Semantic Segmentation**         
>>> - **No object에 대하여 Localization을 수행하고 Pixel을 분류한다.**         
>>> - **기존방식(FCN)에서는 동일한 객체에 대하여 한번에 Masking을 수행한다.**         


>> **Instance Segmentation**      
>>> - **Multi class Object에 대하여 Detection을 수행하고 Pixel을 분류한다.**         
>>> - **동일한 객체에 대해서도 개별로 Masking을 진행한다.**         

---
#### Mask RCNN 관련 Model       
> **Encoder-Decoder / FCN / Mask RCNN**           
>> **Encoder-Decoder**         
>>> - **입력 이미지에 대하여 Encoder를 통하여 차원을 축소한다.**         
>>> - **축소된 차원의 이미지를 Decoder를 통하여 확장한다.**         
      


<p align="center"><img src="/assets/img/posts/MaskRCNN/FCN-Skipconnection.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/skip-connection.PNG"></p>

>> **FCN**      
>>> - **Encoder의 개념을 이어 입력 이미지에 대하여 차원을 축소한다.**         
>>> - **Decoder 과정이 축소된 차원에서 원본 이미지의 Resolution으로 급격하게 Up-Sampling 된다.**         
>>> - **이 때 경계가 모호해지는 문제를 예방하기 위하여 Skip Connection을 사용한다.**         
>>> - **Skip Connection은 ResNet에서 사용된 개념으로 Encoder 과정에 Pooling 부분에 대하여 Concatenate를 진행한다.**   
  
  
<p align="center"><img src="/assets/img/posts/MaskRCNN/MASKRCNN.PNG"></p>
  
>> **Mask RCNN**      
>>> - **Mask RCNN에서의 Instance Segmentation에 관한 2가지 사항**         
>>> **1. 기존 Faster RCNN에서 사용하는 ROI Pooling 대신 ROI Align을 사용한다**         
>>> **2. FCN을 통한 분류작업에 Multi-Class Classification을 사용하지 않고 Binary Classification을 사용한다.**        
  
---
#### What is ROI Align
> **ROI Pooling vs ROI Align**           
>> **RPN을 통하여 얻어진 ROI에 대하여 3 x 3 Convolution을 진행한다.**         
>> **ROI 영역에 대하여 일정한 크기의 Grid로 여러 Cell을 나누어 Feature map을 생성한다.**
<p align="center"><img src="/assets/img/posts/MaskRCNN/fast-rcnn_Quantization.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/roi-problem.PNG"></p>

>> **ROI Pooling**         
>>> - **동작 방식**   
>>> **1. ROI Pooling의 경우 추출된 ROI 영역에 대하여 7 x 7 Grid로 나눈다.**   
>>> **2. ROI 영역에 대하여 정수를 기준으로 Grid를 분할한다.**   
>>> **3. 분할된 영역에 대하여 Fully connected Layer를 수행한다.**   
>>> - **문제점**   
>>> **원인: 입력 이미지에 대하여 Grid 분할을 진행하는 과정에서 ROI 영역이 딱 맞아 떨어지지 않는다.**   
>>> **결과: 실수부에 걸쳐 있는 ROI에 7 x 7 Grid를 진행하고 정수영역에 대해서 Convolution이 진행된다. 따라서 실수부에 영역이 누락되거나 Quantization에 문제가 발생된다.**   

<p align="center"><img src="/assets/img/posts/MaskRCNN/roi-problem1.PNG"></p>

>> **ROI Align**         
>>> - **동작 방식**       
>>> - **1. 추출된 ROI영역에 대한 Scale Factor를 적용한다.**       
>>> - **2. Scale factor가 적용된 Feature를 매핑한다.**       
>>> - **3. 3 x 3 Grid 분할을 진행한다.**       
>>> - **4. 각 Gridcell에 4개의 좌표를 생성한다.** 
>>> - **5. 4개 좌표에 대하여 주변 좌표를 통하여 Bilinear Interpolation을 진행한다.**       
>>> - **6. Bilinear Interpolation의 결과로 Feature map을 생성한다.**       
>>> **결과: 실수 영역에 대하여 쌍선형 보간법을 사용하여 픽셀에 대한 정보를 보존하여 ROI Pooling 보다 높은 품질의 특성맵을 생성한다.**   
      

<p align="center"><img src="/assets/img/posts/MaskRCNN/roi-problem1.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/roi-align.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/roi-align1.PNG"></p>

  
<p align="center"><img src="/assets/img/posts/MaskRCNN/bilinear_Interpolation_dot1.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/bilinear_Interpolation_all.PNG"></p>
<p align="center"><img src="/assets/img/posts/MaskRCNN/roi_align_all.PNG"></p>
  
---
#### 결론     
> **Mask RCNN**           
>> **1. Mask RCNN의 FCN에서는 특정 픽셀값이 어떠한 Object를 어떠한 Class로 분류하는지 결정하는 Multi-Class classification을 사용하지 않는다.**              
>> **2. 특정 픽셀값이 어떠한 Class인지 고려하지 않고 Object가 Masking인가 아닌가 분류하는 Binary Masking Classification을 수행한다.**              
>> **3. Binary Masking Prediction을 거친 후 Prediction Masking을 원본 이미지의 Object 크기 만큼 복원하여 Overlap 한다.**              
>> **4. Loss Function**              
>>> **1) Faster RCNN: Object Classification Loss Function = Multi-class Cross-Entropy Loss**              
>>> **2) Bounding Box: L1 Smooth Loss**              
>>> **3) Masking FCN: Binary Cross-Entropy Loss**              

<p align="center"><img src="/assets/img/posts/MaskRCNN/Masking_Overlap.PNG"></p>



  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
