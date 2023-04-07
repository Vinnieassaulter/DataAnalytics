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

## The way to visualize the data
Since the dataset that we will deal with is text data, we found tableau to be considerably complicated to utilize for visualization of this data set and decided to use Seaborn and Matplotlib.

## Exploratory Data Analysis

### *Basic information*
First of all, we displayed the data information, shape, and structure by using some appropriate methods in Pandas to better understand the dataset that we have.
The dataset(train.csv) has 404290 rows and 6 columns. The data structure of train.csv is as follows:
- **id**: A simple row ID.
- **qid1, qid2**: The unique ID of each question in the pair.
- **question{1, 2}**: The actual textual contents of the questions.
- **is_duplicate**: The label is about whether the two questions are duplicates of each other.

<!-- On the other hand, the data structure of train.csv is as follows:
- **test_id**: A simple row ID.
- **question{1, 2}**: The actual textual contents of the questions. -->

### *Missing Data*

Next, we investigated whether there were any missing data in the dataset because missing values will have a bad influence on our data analysis. We found that there are three missing data in the dataset. Since we aim to identify the duplication between two pair questions, We decided to drop these three rows.

### *Duplication*

After removing missing values from the dataset, We also examined how many duplicated questions are in the dataset by counting the number of rows in the "is_duplicated" column. We found that there are 149263 duplicated questions out of 404287 in the dataset, which means 37% of the dataset can be considered as duplicated questions.

![](/png/duplicate.png)

### *Unique questions ＆ Repeated questions*

Next, we counted the number of unique questions and repeated questions in the dataset. By combining the two columns ”qid1” and ”qid2” in the dataset, we can count the number of unique questions by using `np.unique()` in Pandas. The total number of unique questions is 537929. 

The number of questions that are repeated more than 1 time is 111778 which is 20.78%. The maximum number of times a question occurs is 157. 
![](/png/Unique_repeated_quesions.png)

The top 5 most repeated questions are as follows:
- What are the best ways to lose weight?
- How can you look at someone's private Instagram account without following them?
- How can I lose weight quickly?
- What's the easiest way to make money online?
- Can you see who views your Instagram?

From a personal perspective, it makes sense that these questions make up the top five.

![](png/Log_Histogram_of_question_occurances.png)

We plotted a histogram of the number of question occurrences and the number of questions with that number of occurrences on a logarithmic scale. The majority of the questions in the dataset approximately have occurrences of less than 60. On the other hand, it can be seen that all questions that appear repeatedly more than 60 times are present only once each.

## Feature Engineering

In order to further understand other features behind the dataset, we reconstruct the data and add some new features. The definition of the new features is as follows:

- word_Common = (The number of common unique words in Question 1 and Question 2)
- word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
- word_share = (word_common)/(word_Total)

## Shared words
From the following diagram, we can notice that as the word share increases there is a higher chance the questions are similar. From the histogram, we can understand that word share has some information differentiating similar and dissimilar classes. This trend matches the intuition that similar questions will have more words in common.

![](png/word_share.png)



## MinHash and Locality Sensitive Hashing

### *Set Representation*
we first represent the questions as set representations of k-shingles to guarantee that the probability of obtaining each shingle is low in the document space. We adopted a word-level shingle instead of a character-level shingle and I set k=1. The reason why I adopted k=1, in this case, is that the probability of finding each shingle in the union of shingles is lower in the second case with k=2. Additionally, since common English words are not useful for data analysis such as "the" and "and", we first import the English stopwords from the NLTK library and remove them from the set representation of the questions. The norm_dict dictionary maps a question to the actual question string. This dictionary can be used to evaluate the results of the MinHashLSH output.


## *MinHash signatures*
We used MinHash to generate "min hash signatures" for each question in the set_dict dictionary. The signatures will be stored in the min_dict dictionary, which maps each question to its corresponding min hash signature.


## *Locality Sensitive Hashing*
By using Minshashing, we now compressed the questions to numeric representation, and we’ve defined a signature metric. Since we would like to compare questions that are more likely similar to each other rather than comparing two completely different questions with each other, we can use Locality Sensitive Hashing (LSH) to find similar questions in a large set.


# Trainning Model
## Data Split

# Conclusion
