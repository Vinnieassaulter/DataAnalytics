---
documentclass: scrartcl
fontsize: 10pt
geometry: margin=0.7cm
author: Yusuke Sugihara, Pan Qiyang 
title: Data Analytics Mini Project(No.13-Dejavu)
numbersections: true
toc: true
---
# Introduction
Nowadays, Quora is known as one of the most popular question-and-answer platforms on the internet. That allows people all over the world to learn from each other and a huge number of people utilize Quora every month indeed.　It is easy to imagine that there are tons of questions similar to each other on the platform. Therefore, it is considered beneficial for all users if there is an algorithm that can identify a new question that is similar to the existing questions on the platform that have already been answered by other users. Thus, our problem statement in this project is to predict whether a pair of questions are duplicates or not.

# Data Analytics

## Exploratory Data Analysis
After we read the dataset, we first checked the basic information about the dataset by using the following code.
### *Basic information*
```python
data_train = pd.read_csv('data/train.csv')
data_train.head()
data_train.info()
data_train.describe()
data_train.shape
```

### *Missing Data*
Next, we investigated the missing data. We found that there are three missing data in the dataset. Since we aim to identify the duplication between two pair questions, for the missing data rows, We decided to drop these three rows as follows.

```python
data_train[data_train.isnull().any(axis=1)] 
data_train = data_train.dropna(how="any").reset_index(drop=True)
data_train.shape
```

### *Duplication*
```python

```

## Set Representation
```python
```

## Minhashing 
```python
```

## Locality Sensitive Hashing
```python
```

# Results

# Conclusion
