import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Flight Delay Predictor Home",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

## App title
st.title("✈️ FLIGHT DELAY PREDICTION APP")

# Load image 
img_path = Path(__file__).parent / "misc" / "flight_header.png"
image = Image.open(img_path)

# Display image with fixed width 
st.image(image, use_container_width=True)

# Introduction text
st.markdown("""
Welcome aboard the **Flight Delay Prediction App**!  

Every year, nearly **1 in 5 flights** around the world don’t take off on time, costing the airline industry **over $30 billion** and throwing countless travel plans into chaos. Behind these frustrating delays lies a fascinating story — one told by data.

In this project, *you* become the data detective. Journey through real-world flight delay patterns and discover the hidden factors that cause delays — from timing to routes, and more.  

On the **Exploratory Data Analysis (EDA)** page, dive into interactive visuals that bring the data to life and reveal surprising trends.  

Then, step into the shoes of a data scientist in the **Predictor** section, where you’ll use a powerful machine learning model to forecast whether a flight might be delayed.  

Your journey into the world of flight delay analytics starts now. Navigate through the sidebar and let the data tell its story!
""")