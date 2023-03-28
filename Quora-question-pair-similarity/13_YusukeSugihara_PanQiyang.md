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
Since the dataset that we will deal with is text data, we found tableau to be considerably complicate to utilize for visualization of this data set and decided to use Seaborn and matplotlib as usual.

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
data_train['is_duplicate'].value_counts()
sns.countplot(data_train['is_duplicate'])
```
## MinHash and Locality Sensitive Hashing

### *Create a dictionary of questions*

```python
correct = data_train[data_train['is_duplicate']==1]
correct_dict = {}
for x,y in zip(correct['question1'],correct['question2']):
    if correct_dict.get(x)==None:
        correct_dict[x] = [y]
    else:
        correct_dict[x].append(y)
        correct_dict[x] = [i for i in set(correct_dict[x])]
for x,y in zip(correct['question2'],correct['question1']):
    if correct_dict.get(x)==None:
        correct_dict[x] = [y]
    else:
        correct_dict[x].append(y)
        correct_dict[x] = [i for i in set(correct_dict[x])]
```

### *Set Representation*
we first represent the questions as set representations of k-shingles to gurantee that the probability of obtaining each shingle is low in the document space. We adopted a word-level shingle instead of character-level shingle and I set k=1. The reason why I adopted k=1 in this case is because the probability of finding each shingle in the union of shingles is lower in the second case with k=2. Additionally, since common English words are not useful for data analysis such as "the" and "and", we first import the English stopwords from the NLTK library and remove them from the set representation of the questions. The norm_dict dictionary maps a question to the actual question string. This dictionary can be used to evaluate the results of the MinHashLSH output.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

set_dict={} # maps question to set representation of question
norm_dict={} # maps question to actual question. We may use this dictionary to evaluate results of LSH output.
count=1

for question in tqdm([x for x in com_data_train['question'] if type(x)==str]):
    temp_list = []
    for shingle in question.split(' '): # shingle is a word
        if shingle not in stop_words:
            temp_list.append(shingle.lower())
    set_dict["m{0}".format(count)] = set(temp_list)
    norm_dict["m{0}".format(count)] = question
    count +=1
```

## *MinHash signatures*
We used MinHash to generate "min hash signatures" for each question in the set_dict dictionary. The signatures will be stored in the min_dict dictionary, which maps each question to its corresponding min hash signature.

```python
num_perm = 256
min_dict = {} # maps question id (eg 'm1') to "min hash signatures"
count2 = 1
for val in tqdm (set_dict.values()): 
    m = MinHash(num_perm=num_perm)
    for shingle in val:
        m.update(shingle.encode('utf8'))
    min_dict["m{}".format(count2)] = m
    count2+=1
```

## *Locality Sensitive Hashing*
By using Minshashing, we now compressed the questions to numeric representation, and we’ve defined a signature metric. Since we would like to compare questions that are more likely similar with each other rather than comparing two completely different questions with each other, we can use Locality Sensitive Hashing (LSH) to find similar questions in a large set.

```python
lsh = MinHashLSH(threshold=0.4, num_perm=num_perm)
for key in tqdm(min_dict.keys()):
    lsh.insert(key,min_dict[key])
```
```python
def create_cand_pairs():
    big_list = []
    for query in min_dict.keys():
        bucket = lsh.query(min_dict[query])
        if len(bucket)==1:
            big_list.append([bucket[0],"None"])
        if len(bucket)>1:
            first_val = bucket[0]
            for val in bucket[1:]:
                second_val = val
                big_list.append([first_val,second_val])
    return big_list
```
# Results


# Conclusion
