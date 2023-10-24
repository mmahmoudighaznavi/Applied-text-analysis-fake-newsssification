# Fake News Classification using Deep Learning, Classical Algorithms, and Topic Modeling

--- 

## About <a name = "about"></a>

Fake news refers to misinformation or disinformation presented as news. It often has the aim of damaging the reputation of a person or entity or making money through advertising revenue. However, identifying the veracity of an article can be a complex task due to the quality of writing, inherent biases, or simply the nature of the information being reported.

In this project, we aim to tackle the problem of fake news detection using machine learning and deep learning approaches. We employ a diverse set of models including classical machine learning algorithms such as Logistic Regression, Decision Trees, Gradient Boosting and advanced Deep Learning techniques such as Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM), and Gated Recurrent Units (GRU).

---

## 🎈 Contributions

This project is collaborated by:
* Ruddy Simonpour <ruddys@sandiego.edu>
* Mohammad Mahmoudighaznavi <mmahmoudighaznavi@sandiego.edu>

---

## Prerequisites

### Necessary library packages 
```
import pandas as pd
import numpy as np
import csv

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Data Partitoning
from sklearn.model_selection import train_test_split

# Data Pre-Processing and Text Tokenizing
nltk.download('stopwords')
stop = set(stopwords.words('english'))
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ML Developement
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense

# Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_s

```

---

## Project Tree
```
|- README.md \  
|- Input Resources \   
|-     |- fake_news.csv
|- Python Script \
|-     |- Data_Ingestion 
|-     |- Viz_EDA
|-     |- Pre-processing
|-     |- Text Tokenizer
|-     |- Modeling
|-     |- Topic Modeling
|-     |- Web Scraping Aljazeera News Media 
|-     |- Topic Modeling for Aljazeera News       
```
---

## Dataset

- [ ] The first dataset used in this project is sourced from Kaggle, which includes different features like 'author', 'title', and 'text'. The 'text' column contains the content of the news articles. The dataset comprises 20800 rows and 5 columns.
- [ ] The second data resource for this project is sourced from Aljazeera website. Web scraping and selenium used to get fresh news from Aljazeera website. 

---

## Methodology

### Data Cleaning

The first step in our analysis involved cleaning the data. We removed unnecessary columns and handled missing data points. To mitigate the effects of class imbalance, we ensured that there's an equal distribution of the two classes - 'fake' and 'true' news.

The 'fake' news is labeled as '1' with 10413 instances, and the 'true' news labeled as '0' with 10387 instances, which provides us a balanced dataset for our models.

### Text Pre-processing
Text data needs to be processed and tokenized before we can use it for machine learning algorithms. Here is a plain language explanation of the text processing steps we followed:

**Removing Non-Alpha Characters:** We first cleaned up the text by removing any characters that aren't alphabets. This includes numbers, punctuation, and special characters.

**Lowercasing:** Next, we converted all the text to lowercase. This is done to ensure that the same word, when used in different cases, is considered the same. For example, 'Fake' and 'fake' are the same words and should be treated as such.

**Removing Stop Words and Stemming:** English has many common words like 'is', 'an', 'the', etc., which do not add much value when it comes to understanding the meaning of the text. These are called stop words and we removed them. Next, we performed a process called 'stemming'. This is where words are reduced to their base or root form. For example, 'driving', 'drives', 'drove' are variations of the word 'drive'.

### Embedding:
We utilized the Word2Vec algorithm to convert text data into numerical vectors which enabled our deep learning models to process and learn from the textual content efficiently. This technique facilitated the capture of semantic relationships between words, thereby enhancing the performance of our models in identifying fake news.

### Modeling: 
We employed various deep learning models including Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and Recurrent Neural Networks (RNN) to analyze the temporal dependencies in the text data, which is crucial for understanding the context and semantics. In addition, we also incorporated classical machine learning algorithms like Decision Tree, Support Vector Machines (SVM), and Logistic Regression to compare and analyze their performance against deep learning models in the classification task. This combination of models provided a comprehensive analysis of different machine learning approaches to fake news classification.

### Evaluation:
For evaluating the performance of our deep learning models, we used the accuracy metric alongside Area Under the Receiver Operating Characteristic Curve (AUC-ROC) values. These metrics provided insight into the true positive rate versus false positive rate, offering a holistic view of model performance. In the case of classical algorithms, we expanded the evaluation to include confusion matrices, which provided a detailed breakdown of true positive, true negative, false positive, and false negative rates, alongside the AUC-ROC curve values and accuracy. This multifaceted evaluation approach enabled a thorough assessment of each model’s strengths and weaknesses in tackling fake news classification.

### Topic Modeling: 
Topic modeling was incorporated into our project to extract and understand the underlying topics present in the news data. This technique helped in categorizing the news articles into distinct topics, which further aided in discerning the authenticity of the news. By analyzing the distribution of topics and their coherence, we could better understand the thematic structure of the news data, which is essential for identifying patterns associated with fake news. Through topic modeling, we could enrich our feature set and improve the robustness of our classification models in detecting fake news.



