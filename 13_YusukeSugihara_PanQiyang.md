---
documentclass: scrartcl
fontsize: 10pt
geometry: margin=0.7cm
author: YUSUKE SUGIHARA, Pan Qiyang 
title: Data Analytics Mini Project(No.13-Dejavu)
numbersections: true
toc: true
---
# Introduction
Nowadays, Quora is known as one of the most popular question-and-answer platforms on the internet. That allows people all over the world to learn from each other and a huge number of people utilize Quora every month indeed.　It is easy to imagine that there are tons of questions similar to each other on the platform. Therefore, it is considered beneficial for all users if there is an algorithm that can identify a new question that is similar to the existing questions on the platform that have already been answered by other users. Thus, our problem statement in this project is to predict whether a pair of questions are duplicates or not.

# Data Analytics

## Exploratory Data Analysis
```python
data = pd.read_csv('data/train.csv')
print(f'The number of datapoints is {data.shape}')
data.head(5)
```

## Set Representation, Minhashing and Locality Sensitive Hashing
```python
```

# Results

# Conclusion
