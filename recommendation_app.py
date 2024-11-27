import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process

# Load precomputed data
with open('precomputed_data.pkl', 'rb') as f:
    precomputed_data = pickle.load(f)

content_df = precomputed_data['content_df']
tfidf_matrix = precomputed_data['tfidf_matrix']
content_nbrs = precomputed_data['content_nbrs']
collab_similarity_scores = precomputed_data['collab_similarity_scores']
data = precomputed_data['data']

# Function to recommend games
def recommend_hybrid(game_name):
    # Try to match the game in the content-based system
    matched_game, score, _ = process.extractOne(game_name, content_df['name'])
    
    # If the match score is below 90, return None for both game and recommendations
    if score < 90:
        return None, None
    
    recommendations = {}

    # Content-based recommendations (from content_df)
    if game_name in content_df['name'].values:
        content_index = content_df[content_df['name'] == matched_game].index[0]
        
        # Ensure content_nbrs and tfidf_matrix are available
        if 'content_nbrs' in globals() and 'tfidf_matrix' in globals():
            content_distances, content_indices = content_nbrs.kneighbors(tfidf_matrix[content_index], n_neighbors=12)

            # Collect content-based recommendations
            for idx, distance in zip(content_indices[0][1:], content_distances[0][1:]):  # Skip the first as it’s the original game
                game = content_df.iloc[idx]['name']
                recommendations[game] = recommendations.get(game, 0) + (1 - distance)
        else:
            # If there are missing data structures, return None
            return None, None
    else:
        # If the game isn't found in content_df, return None for recommendations
        recommendations = {}

    # Collaborative-based recommendations (from collaborative data)
    if matched_game in data.index:
        collab_index = data.index.get_loc(matched_game)
        
        # Ensure collab_similarity_scores is available
        if 'collab_similarity_scores' in globals():
            collab_similar_items = sorted(list(enumerate(collab_similarity_scores[collab_index])), key=lambda x: x[1], reverse=True)[1:12]
            
            # Collect collaborative-based recommendations
            for idx, similarity in collab_similar_items:
                game = data.index[idx]
                recommendations[game] = recommendations.get(game, 0) + similarity
        else:
            # If collaborative-based data isn't available, return None
            return None, None
    else:
        # If the game isn't found in the collaborative dataset
        recommendations = {}

    # Sort the recommendations by their score in descending order and return top 12
    ranked_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:12]
    
    # If no recommendations were found, return None
    if not ranked_recommendations:
        return None, None
    
    return matched_game, ranked_recommendations

# Streamlit app
st.title("Hybrid Game Recommendation System")
st.write("Enter the name of a game to get personalized recommendations.")

# User input
game_name = st.text_input("Enter a game you like:")
process_button = st.button("Get Recommendations")

if process_button and game_name:
    with st.spinner("Processing..."):
        matched_game, recommendations = recommend_hybrid(game_name)

    if matched_game and recommendations:
        # Display recommendations
        st.success(f"Recommendations based on '{matched_game}':")
        cols = st.columns(3)
        for i, (game, score) in enumerate(recommendations):
            with cols[i % 3]:
                game_row = content_df[content_df['name'] == game].iloc[0]
                image_url = game_row['header_image'] if not pd.isna(game_row['header_image']) else 'path/to/default_image.jpg'
                st.image(image_url, caption=f"{game} ({score:.4f})", use_column_width=True)
    else:
        st.error("No close match found. Please try another game.")
