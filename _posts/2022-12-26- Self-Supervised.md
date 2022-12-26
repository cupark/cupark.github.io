---
layout: post
title:  "Self-Superivsed Learning!"
summary: "What is Self-Superivsed Learning"
author: cupark
date: '2022-12-26 16:05:13 +0530'
category: NeuralNetwork
thumbnail: /assets/img/posts/SelfSupervised/thumb_self.png
keywords: Self-Supervised
permalink: /blog/Self-Superivsed/
usemathjax: true
---

---
## Self-Supervied Learning  
---

### Self-Supervised Learning 이란  

라벨이 없는 Untagged data를 기반으로 한 학습이다.   
스스로 학습 데이터에 대한 분류를 수행하기 떄문에 Self라는 접두어가 붙었다.   
데이터에 tag가 존재하지 않는다. 따라서 개와 고양이 사진을 데이터로 넣어   
학습을 시키는 경우 개와 고양이가 무엇 인지 모르는채 이미지의 특성에 따라 분류를 한다.  

---

### Self-Supervised Learning 과정  

1. Pretext task - 연구자가 직접 만든 task를 정의한다.  
2. Label이 전혀 없는 데이터셋을 이용하여 Pretext task를 목표로 모델을 학습시킨다.  
   : Point - 데이터 자체의 정보를 Modify하여 지도 학습처럼 사용해야 한다. 
3. 학습된 모델을 Down-Stream task에 가지고 온다.  
   : Point - 가중치의 변동을 막은 뒤, 전이 학습을 수행(4.)
4. 3.에서 진행한 전이 학습 시 라벨을 전달하여 Supervised Learning을 수행하고 성능을 평가한다. 
<p align="center"><img src="/assets/img/posts/SelfSupervised/self-supervised-workflow.png"></p>
<p align="center"><img src="/assets/img/posts/SelfSupervised/self-supervised-finetuning.png"></p>

---
