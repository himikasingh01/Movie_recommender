import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies and preprocess genres before merging
movies = pd.read_csv('movies.csv')
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Load ratings
ratings = pd.read_csv('ratings.csv')

# Merge after genre cleanup
movie_data = pd.merge(ratings, movies, on='movieId')

# Keep only unique movies (drop duplicates)
movies_unique = movies.drop_duplicates(subset='movieId')[['movieId', 'title', 'genres']].reset_index(drop=True)

# Optional: limit for performance
movies_unique = movies_unique.head(9000)

# TF-IDF & Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_unique['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
title_to_index = pd.Series(movies_unique.index, index=movies_unique['title'])

# Recommendation function
def recommend_movie(title, cosine_sim=cosine_sim):
    if title not in title_to_index:
        return ["Movie not found."]
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_unique['title'].iloc[movie_indices].tolist()


def recommend_by_genre1(genre_name):
    # Find movies containing the specified genre
    similar_movies = movies_unique[movies_unique['genres'].str.contains(genre_name, case=False, na=False)]

    if similar_movies.empty:
        return ["No movies found for this genre."]

    return similar_movies['title'].head(10).tolist()




# Streamlit UI
st.title("🎬 Movie Recommender System")

movie_list = movies_unique['title'].sort_values().tolist()
selected_movie = st.selectbox("Select a movie to get recommendations", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie)
    st.write("Top 10 Recommendations:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")


st.markdown("---")
st.title("🎭 Recommend by Genre")

# Input box for genre name
genre_input = st.text_input("Enter a genre (e.g., Drama, Comedy, Action):")

if st.button("Recommend by Genre"):
    if genre_input:
        genre_recommendations = recommend_by_genre1(genre_input)
        st.write("Top 10 Genre-based Recommendations:")
        for i, movie in enumerate(genre_recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("Please enter a genre.")


