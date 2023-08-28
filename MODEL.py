

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Step 1: Load and preprocess the dataset
data = pd.read_csv('Precily.csv')

# Step 2: Feature extraction
vectorizer = TfidfVectorizer()
text1_features = vectorizer.fit_transform(data['text1'])
text2_features = vectorizer.transform(data['text2'])

# Step 3: Calculate similarity
batch_size = 1000  
num_samples = len(data)
similarity_scores = np.zeros(num_samples)

for i in range(0, num_samples, batch_size):
    batch_end = min(i + batch_size, num_samples)
    batch_scores = cosine_similarity(
        text1_features[i:batch_end],
        text2_features[i:batch_end]
    ).diagonal()
    similarity_scores[i:batch_end] = batch_scores

# Step 4: Convert similarity scores to a value between 0 and 1
normalized_similarity_scores = (similarity_scores + 1) / 2

# Step 5: Save similarity scores to a new column in the dataset
data['similarity'] = normalized_similarity_scores

# Step 6: Save the updated dataset with similarity scores
data.to_csv('dataset_with_similarity.csv', index=False)

# Step 7: Save the vectorizer and model as pickle files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(normalized_similarity_scores, f)
