import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import subprocess

# Vérifier si joblib est installé
try:
    import joblib
    st.write("✅ joblib est bien installé sur Streamlit Cloud")
except ModuleNotFoundError:
    st.write("❌ joblib n'est PAS installé. Tentative d'installation...")
    subprocess.run(["pip", "install", "joblib"])
 


# Charger le modèle entraîné
model = joblib.load("weather_model20.pkl") 
scaler = joblib.load("minmax_scaler.pkl") 


# Titre de l'application
st.title("🌤️ Application de Prédiction DE LA TEMPERATURE")

# Interface utilisateur pour saisir les données
st.sidebar.header("Entrez les paramètres météo")

# Entrées pour chaque variable du modèle
wind_dir = st.sidebar.number_input("Direction du vent (°)", min_value=0, max_value=360, value=90)
app_temp = st.sidebar.number_input("Température Ressentis (°C)", min_value=-50.0, max_value=50.0, value=20.0)
wind_spd = st.sidebar.number_input("Vitesse du vent (m/s)", min_value=0.0)
wind_gust_spd = st.sidebar.number_input("Rafales de vent (m/s)", min_value=0.0)
rh = st.sidebar.slider("Humidité relative (%)", min_value=0, max_value=100, value=50)
wind_cdir = st.sidebar.selectbox("Wind Cardinal Direction", options=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
# Bouton pour faire la prédiction
if st.sidebar.button("Prédire la température"):
    # Initialiser le LabelEncoder pour la direction du vent
    encoder = LabelEncoder()
    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    encoder.fit(wind_directions)

    # Encoder la direction du vent sélectionnée
    wind_cdir_encoded = encoder.transform([wind_cdir])[0]
    
    # Transformer les données en format utilisable par le modèle
    features = np.array([[wind_dir, wind_spd, wind_gust_spd, rh,app_temp, wind_cdir_encoded]])

    # Appliquer la normalisation (scalage MinMax) aux caractéristiques d'entrée
    features_scaled = scaler.transform(features)

    # Prédiction du modèle
    prediction = model.predict(features_scaled)

    # Affichage du résultat
    st.write("📊 Valeurs envoyées au modèle (normalisées) :", features_scaled)
    st.write("Shape des données envoyées :", features_scaled.shape)

    st.subheader("🌦️ Prévision de la température :")
    temperature = prediction[0]  # Récupérer uniquement la température
    st.write(f"🌡️ Température prévue : **{temperature:.2f}°C**")
