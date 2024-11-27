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
import numpy as np
from fuzzywuzzy import process

# Function to perform fuzzy matching of the game name
def fuzzy_match_game(game_name, content_df, threshold=70):
    # Fuzzy match the input game name to the closest game name from content_df
    matched_game, score, _ = process.extractOne(game_name, content_df['name'])  # Unpack the result into three variables
    if score < threshold:  # Define a threshold for the fuzzy match score (you can adjust this value)
        print(f"No close match found for '{game_name}' with fuzzy matching!")
        return None
    return matched_game

# Token-based game name matching (for fallback when fuzzy match score is below threshold)
def recommend_game_based_on_input(user_input, game_names, threshold=70):
    # Tokenize the input to extract key terms (this is very basic tokenization)
    input_tokens = set(user_input.lower().split())
    
    # Find fuzzy matches for the input
    matched_game, score, _ = process.extractOne(user_input, game_names)
    
    if score >= threshold:
        print(f"Exact match found: {matched_game} (Score: {score})")
        return matched_game
    else:
        # If no exact match, find games that match key terms in the name
        recommended_games = []
        for game in game_names:
            game_tokens = set(game.lower().split())
            common_tokens = input_tokens.intersection(game_tokens)
            if len(common_tokens) > 0:
                # Calculate score based on the number of matching tokens
                match_score = len(common_tokens) * 10  # Basic scoring, adjust as needed
                recommended_games.append((game, match_score))

        # Sort recommendations based on match score and return the top matches
        recommended_games = sorted(recommended_games, key=lambda x: x[1], reverse=True)
        print(f"Recommended games based on your input (with token match): {recommended_games[:5]}")  # Showing top 5 recommendations
        return [game for game, _ in recommended_games]

# Collaborative recommendation function with fuzzy matching integration
def recommend_hybrid(game_name):
    # Ensure game_name is treated as a string
    game_name = str(game_name)
    threshold=70
    # Fuzzy matching to find the closest game name in content_df
    matched_game = fuzzy_match_game(game_name, content_df, threshold)
    if not matched_game:
        # If no match found via fuzzy matching, fall back to token-based matching
        matched_game = recommend_game_based_on_input(game_name, content_df['name'], threshold)
        if not matched_game:
            return None, None
    
    recommendations = {}

    # Content-based recommendations (from content_df)
    content_index = content_df[content_df['name'] == matched_game].index[0]
    content_distances, content_indices = content_nbrs.kneighbors(tfidf_matrix[content_index], n_neighbors=12)
    
    # Content-based similarity scores (from content-based system)
    for idx, distance in zip(content_indices[0][1:], content_distances[0][1:]):  # Skip the first as it’s the original game
        game = content_df.iloc[idx]['name']
        recommendations[game] = recommendations.get(game, 0) + (1 - distance)  # Convert distance to similarity
    
    # Add collaborative recommendations if the game exists in the collaborative dataset
    if matched_game in data.index:
        collab_index = np.where(data.index == matched_game)[0][0]
        collab_similar_items = sorted(list(enumerate(collab_similarity_scores[collab_index])), key=lambda x: x[1], reverse=True)[1:12]
        
        # Collaborative-based similarity scores
        for idx, similarity in collab_similar_items:
            game = data.index[idx]
            recommendations[game] = recommendations.get(game, 0) + similarity
    
    # Sort by combined score in descending order and get top 12
    ranked_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:12]
    
    # If no recommendations were found, return None
    if not ranked_recommendations:
        return None, None
    
    return matched_game, ranked_recommendations

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
