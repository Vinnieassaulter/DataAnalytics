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

## The way of data visualization in this project
Since the dataset that we will deal with is text data, we found tableau to be considerably complicated to utilize for visualization of this data set and decided to use Seaborn and Matplotlib.

## Exploratory Data Analysis

### *Basic information*
First of all, we displayed the data information, shape, and structure by using some appropriate methods in Pandas to better understand the datasets that we have.
Training data(train.csv) has 404290 rows and 6 columns, and test data (test.csv) has (2345796, 3). The data structure of train.csv is as follows:
- **id**: A simple row ID.
- **qid1, qid2**: The unique ID of each question in the pair.
- **question{1, 2}**: The actual textual contents of the questions.
- **is_duplicate**: The label is about whether the two questions are duplicates of each other.

On the other hand, the data structure of train.csv is as follows:
- **test_id**: A simple row ID.
- **question{1, 2}**: The actual textual contents of the questions.

### *Missing Data*

Next, we investigated whether there were any missing data in the dataset because missing values will have a bad influence on our data analysis. We found that there are three missing data in the dataset. Since we aim to identify the duplication between two pair questions, We decided to drop these three rows.

### *Duplication*

After removing missing values from the dataset, We also examined how many duplicated questions are in the dataset by counting the number of rows in the "is_duplicated" column. We found that there are 149263 duplicated questions out of 404287 in the dataset, which means 37% of the dataset can be considered as duplicated questions.

### *Number of unique questions and repeated questions*

Next, we counted the number of unique questions and repeated questions in the training dataset. The total number of unique questions is 537929. The number of questions that are repeated more than 1 time is 111778 which is 20.78%. The maximum number of times a question occurs is 157. The top 5 most repeated questions are as follows:
- What are the best ways to lose weight?
- How can you look at someone's private Instagram account without following them?
- How can I lose weight quickly?
- What's the easiest way to make money online?
- Can you see who views your Instagram?


## Feature Engineering
We reconstruct the data and add some new features:

- freq_qid1 = Frequency of qid1's (e.g. the number of times question1 occur)
- freq_qid2 = Frequency of qid2's
- q1len = Length of q1
- q2len = Length of q2
- q1_n_words = The number of words in Question 1
- q2_n_words = The number of words in Question 2
- word_Common = (The number of common unique words in Question 1 and Question 2)
- word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
- word_share = (word_common)/(word_Total)
- freq_q1+freq_q2 = the total frequency of qid1 and qid2
- freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

## Analyzing shared words
We can see that as the word share increases there is a higher chance the questions are similar. From the histogram, we can see that word share has some information differentiating similar and dissimilar classes.

## MinHash and Locality Sensitive Hashing




### *Create a dictionary of questions*


### *Set Representation*
we first represent the questions as set representations of k-shingles to guarantee that the probability of obtaining each shingle is low in the document space. We adopted a word-level shingle instead of a character-level shingle and I set k=1. The reason why I adopted k=1, in this case, is that the probability of finding each shingle in the union of shingles is lower in the second case with k=2. Additionally, since common English words are not useful for data analysis such as "the" and "and", we first import the English stopwords from the NLTK library and remove them from the set representation of the questions. The norm_dict dictionary maps a question to the actual question string. This dictionary can be used to evaluate the results of the MinHashLSH output.


## *MinHash signatures*
We used MinHash to generate "min hash signatures" for each question in the set_dict dictionary. The signatures will be stored in the min_dict dictionary, which maps each question to its corresponding min hash signature.


## *Locality Sensitive Hashing*
By using Minshashing, we now compressed the questions to numeric representation, and we’ve defined a signature metric. Since we would like to compare questions that are more likely similar to each other rather than comparing two completely different questions with each other, we can use Locality Sensitive Hashing (LSH) to find similar questions in a large set.


# Results


# Conclusion
