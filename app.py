import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ---- Fungsi untuk Rekomendasi ----
def load_data():
    # Load dataset Spotify
    data = pd.read_csv('dataset.csv')  # Sesuaikan path dataset
    selected_features = ['danceability', 'energy', 'tempo', 'valence', 'loudness', 'acousticness']
    data_cleaned = data[selected_features + ['track_name', 'artists']].dropna()
    return data_cleaned, selected_features

def recommend_songs(song_name, data, features_scaled, top_n=5):
    try:
        # Pencocokan sebagian untuk nama lagu
        song_index = data[data['track_name'].str.lower().str.contains(song_name.lower())].index[0]
        song_vector = features_scaled[song_index]
        
        # Menghitung kemiripan
        similarity = cosine_similarity(features_scaled, [song_vector])
        similar_songs = np.argsort(similarity.flatten())[::-1][1:top_n+1]
        
        # Mengambil rekomendasi
        recommendations = data.iloc[similar_songs][['track_name', 'artists']]
        return recommendations
    except IndexError:
        return None

# ---- Fungsi Visualisasi ----
def visualize_recommendations(recommendations, data, selected_features):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, row in recommendations.iterrows():
        song_name = row['track_name']
        song_data = data[data['track_name'] == song_name][selected_features]
        
        # Jika ada lebih dari 1 baris, ambil baris pertama
        if len(song_data) > 1:
            song_data = song_data.iloc[0]
        else:
            song_data = song_data.squeeze()
        
        ax.plot(selected_features, song_data.values, label=song_name)
    ax.set_title('Fitur Audio Lagu-Lagu Rekomendasi')
    ax.set_xlabel('Fitur')
    ax.set_ylabel('Nilai Fitur')
    ax.legend()
    st.pyplot(fig)

# ---- Aplikasi Streamlit ----
def app():
    # Load data
    data_cleaned, selected_features = load_data()
    
    # Preprocessing
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data_cleaned[selected_features])
    
    st.set_page_config(page_title="Sistem Rekomendasi Lagu", layout="wide")
    st.title("🎵 Sistem Rekomendasi Lagu Berbasis Fitur Audio")

    # Input Nama Lagu
    st.sidebar.header("Cari Lagu Berdasarkan Nama")
    song_name_input = st.sidebar.text_input("Masukkan nama lagu:", "")

    # Form untuk fitur lagu baru
    st.sidebar.header("Sesuaikan Preferensi Fitur Audio")
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider("Tempo", 60, 200, 120)
    valence = st.sidebar.slider("Valence (positivity)", 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider("Loudness", -40, 0, -5)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5)

    # Input data lagu baru
    input_song = np.array([danceability, energy, tempo, valence, loudness, acousticness]).reshape(1, -1)
    input_song_scaled = scaler.transform(input_song)

    # Prediksi Rekomendasi
    st.sidebar.header("Rekomendasi Lagu")
    if st.sidebar.button('Dapatkan Rekomendasi'):
        recommendations = recommend_songs(song_name_input, data_cleaned, features_scaled)
        if recommendations is not None:
            st.write("### 🎶 Rekomendasi Lagu yang Serupa:")
            st.write(recommendations)
            visualize_recommendations(recommendations, data_cleaned, selected_features)
        else:
            st.error("Lagu tidak ditemukan dalam database.")

    # Distribusi fitur
    st.header("📊 Distribusi Fitur Lagu")
    st.write("Grafik berikut menunjukkan distribusi fitur audio dari dataset.")
    fig, ax = plt.subplots(figsize=(10, 6))
    data_cleaned[selected_features].hist(bins=20, ax=ax)
    ax.set_title('Distribusi Fitur Audio')
    st.pyplot(fig)

if __name__ == "__main__":
    app()
