import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure you've downloaded the required NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')

# Create global instances and sets for efficiency
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))


class NewsPreprocessor:
    
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))

    def preprocess(self, content):
        content = self._remove_non_alpha(content)
        content = self._lowercase(content)
        content = self._remove_stopwords_and_lemmatize(content)
        return content

    def _remove_non_alpha(self, text):
        return re.sub('[^a-zA-Z]+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_stopwords_and_lemmatize(self, text):
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stopwords_set]
        return ' '.join(lemmatized_words)
    

# Deep learning tokenizing

class DeepLearningTokenizer:
    
    def __init__(self, max_words=20000, oov_token="<OOV>", max_length=1037):
        self.tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
        self.max_length = max_length
        print("Init Max Length:", self.max_length)
        
    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Only change max_length if it was not explicitly set during initialization
        if self.max_length is None:
            self.max_length = int(np.percentile([len(seq) for seq in sequences], 95))
        print("Fit Max Length:", self.max_length) 
        
    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.max_length = 1037
        print("Transform Max Length before padding:", self.max_length)  # Add this
        return pad_sequences(sequences, padding='post', maxlen=self.max_length)
    

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
