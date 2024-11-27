import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pickle

# Load datasets
collab_df = pd.read_csv('smaller_dataset.csv')
content_df = pd.read_csv('steam_dataset.csv')

# Content-based filtering preparation
content_df.drop_duplicates(inplace=True)
content_features = ['name', 'short_description', 'categories', 'genres', 'developers', 'tags', 'publishers']
content_df[content_features] = content_df[content_features].fillna('')

# Combine features for content-based filtering
def combine_features(row):
    return (str(row['name']) + " " +
            str(row['short_description']) + " " +
            str(row['publishers']) + " " +
            str(row['categories']) + " " +
            str(row['genres'])*2 + " " +
            str(row['developers']) + " " +
            str(row['tags'])*2)

content_df['combined_features'] = content_df.apply(combine_features, axis=1)

# Vectorize combined features
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(content_df['combined_features']).astype('float32')

# Initialize NearestNeighbors model
content_nbrs = NearestNeighbors(n_neighbors=12, metric='cosine')
content_nbrs.fit(tfidf_matrix)

# Collaborative filtering preparation
collab = collab_df.drop_duplicates(subset=['user_id', 'app_id']).dropna(subset=['user_id', 'app_id'])
merged_df = collab.merge(content_df[['AppID', 'name']], left_on='app_id', right_on='AppID', how='left')
filtered_recommend = merged_df.groupby('user_id').filter(lambda x: x['is_recommended'].count() >= 1)
data = filtered_recommend.pivot_table(index='name', columns='user_id', values='is_recommended').fillna(0)
collab_similarity_scores = cosine_similarity(data)

# Save precomputed data
with open('precomputed_data.pkl', 'wb') as f:
    pickle.dump({
        'content_df': content_df,
        'tfidf_matrix': tfidf_matrix,
        'content_nbrs': content_nbrs,
        'collab_similarity_scores': collab_similarity_scores,
        'data': data
    }, f)
