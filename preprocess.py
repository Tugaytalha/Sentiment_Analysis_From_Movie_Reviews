#!/usr/bin/env python
# coding: utf-8

# # Preprocessing
# In this notebook, we will preprocess the data to make it ready for the model. The steps are as follows:

# In[8]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import contractions
from autocorrect import Speller


# ## 0. Setting variables
# 

# In[9]:


# Set the path to the dataset
dataset_path = 'data/merged_data.csv'

# Set the path to save the preprocessed dataset
preprocessed_dataset_path = 'data/preprocessed_dataset.csv'


# ## 1. Preparing the libraries 

# In[10]:


# Download stopwords for deleting unnecessary words
nltk.download('stopwords')
# Download lemmatizer for converting words to their base form
nltk.download('wordnet')
# Download spacy model for tokenization
nlp = spacy.load('en_core_web_sm')
# Download punkt for tokenization
nltk.download('punkt')


# Initialize lemmatizer, stopwords, and spell checker
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words = set(stopwords.words('english')) - {'not', 'no'}
spell = Speller()


# # 2. Preprocessing the given text
# The following steps are performed to preprocess the text:
# 1. Lowercasing
# 2. Removing URLs
# 3. Removing email addresses
# 4. Removing punctuation
# 5. Removing numbers
# 6. Correcting misspellings
# 7. Removing extra whitespaces
# 8. Tokenization
# 9. Removing stopwords
# 10. Lemmatization
# 11. Joining tokens back to a single string
# 
# These steps are purify the text from unnecessary information that does not contribute to the meaning of the text.

# In[11]:


def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation (excluding apostrophes for contractions)
    text = re.sub(r"[^\w\s']", '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Correct misspellings
    text = spell(text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords (excluding 'not' and 'no') and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back to a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


# ### Example of preprocessing

# In[12]:


# Example usage
text_data = [
    "I don't like this movie! It's terrible.", 
    "Great movie! I loved it.", 
    "Check out this website: https://example.com",
    "I'm not sure if I enjoyed the play; it was a bit too long.",
    "The product is good, but the service is not.",
    "They've done a fantastic job! Highly recommended.",
    "It's okay, not the best I've seen.",
    "Wow! What an incredible performance!",
    "I wouldn't buy this again; it's a waste of money.",
    "Email me at example@example.com for more information.",
    "Visit us at http://www.example.com for details.",
    "The food was great, but the waiter was rude.",
    "Amazing concert last night! Can't wait for the next one.",
    "Ugh, I hated the ending of that book.",
    "Can you believe what happened in the latest episode?"
]
preprocessed_data = [preprocess_text(text) for text in text_data]

print(preprocessed_data)


# # 3. Loading the dataset

# In[13]:


import pandas as pd

# Load the dataset
dataset = pd.read_csv(dataset_path)


# # 4. Preprocessing the dataset

# In[14]:


# Preprocess the text data
dataset['text'] = dataset['text'].apply(preprocess_text)


# # 5. Saving the preprocessed dataset

# In[15]:


# Save the preprocessed dataset
dataset.to_csv(preprocessed_dataset_path, index=False)


# In[ ]:




