#!/usr/bin/env python
# coding: utf-8

# # Train Model
# In this notebook, we will create and train a model to predict the sentiment of a movie review.

# In[58]:


import pandas as pd
import numpy as np
import re
from keras_nlp.models import Tokenizer


# In[59]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.losses import CategoricalCrossentropy
from tqdm import tqdm


# In[60]:


# Load the data
df = pd.read_csv('data/preprocessed_dataset.csv')


# In[61]:


df.drop(columns=['text']).head(5)


# In[62]:


import matplotlib.pyplot as plt

# Plot the frequency of each sentiment from the most frequent to the least frequent
# This will help us determine which sentiments to drop to balance the dataset
collsWoText = df.columns.drop('text')

# Create a mapping of sentiment to frequency
sentiment_freq = { i : df[i].sum() for i in collsWoText}
# Sort the mapping by frequency
sentiment_freq = dict(sorted(sentiment_freq.items(), key=lambda item: item[1], reverse=True))
# Plot the frequency of each sentiment
plt.bar(sentiment_freq.keys(), sentiment_freq.values())
plt.xticks(rotation=90)
plt.show()


# In[63]:


# Drop the some rows that belong to the most frequent sentiments which neutral, approval and admiration to balance the dataset
df = df.drop(df[df['neutral'] == 1].sample(frac=.8).index)
df = df.drop(df[df['approval'] == 1].sample(frac=.2).index)
df = df.drop(df[df['admiration'] == 1].sample(frac=.17).index)


# In[64]:


# Plot the frequency of each sentiment after balancing the dataset
sentiment_freq = { i : df[i].sum() for i in collsWoText}
sentiment_freq = dict(sorted(sentiment_freq.items(), key=lambda item: item[1], reverse=True))
plt.bar(sentiment_freq.keys(), sentiment_freq.values())
plt.xticks(rotation=90)
plt.show()
# Balanced enough


# In[65]:


df_new = df.drop(columns=['text'])


# In[66]:


# Find frequency of easch sentiment
df_new.sum()


# In[67]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df_new, test_size=0.2, random_state=42)

# Initialize TextVectorization layer
max_features = 5000  # Maximum number of words to consider
sequence_length = 28  # Maximum length of a sequence

# Ensure X_train and X_test are of type str
X_train = X_train.astype(str)
X_test = X_test.astype(str)

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int', # float?
    output_sequence_length=sequence_length
)

# Adapt the vectorize layer to the training data
vectorize_layer.adapt(X_train.values)

# Vectorize the training and testing data
X_train_vectorized = vectorize_layer(X_train.values)
X_test_vectorized = vectorize_layer(X_test.values)

# Convert y_train and y_test to numpy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[67]:





# In[68]:


# Find longest review
max_len = 0
for review in tqdm(X_train.values):
    max_len = max(max_len, len(review.split()))
max_len


# In[69]:


#llok up x_train vector
X_train_vectorized[125236]


# In[74]:


# Build the model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=sequence_length))
model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid'))
model.add(Dense(30, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(X_train_vectorized, y_train, epochs=4, batch_size=32, validation_data=(X_test_vectorized, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_vectorized, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


# In[71]:


# Train with whole dataset

# Ensure x is of type string
df['text'] = df['text'].astype(str)

# Vectorize the whole dataset
X_vectorized = vectorize_layer(df['text'].values)


# Convert y to numpy arrays
y = df_new.to_numpy()

# Compile the model
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(X_vectorized, y, epochs=4, batch_size=32)


# # Save the model
# 

# In[ ]:


from tensorflow.keras.models import save_model

# Save the model
save_model(model, 'models/sentiment_model')


# In[ ]:


# Print versions 
import tensorflow as tf
import keras_nlp
print(f'Tensorflow version: {tf.__version__}')
print(f'Keras version: {tf.keras.__version__}')
print(f'Keras NLP version: {keras_nlp.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')


# In[ ]:




