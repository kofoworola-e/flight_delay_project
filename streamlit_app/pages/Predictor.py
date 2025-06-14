import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os
import plotly.graph_objects as go
import shap

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🤖 PREDICT FLIGHT DELAY")
st.write("Enter flight details below to predict if your flight will be delayed.")

# ------ Look up table data--------
@st.cache_data(show_spinner=False)
def load_lookups_from_drive():
    airline_delay_file_id = "1ed2CeYXgwrWEc-aecGBfwRTBbR-Rkilu"
    route_density_file_id = "1-1LadBkeEYEsCfC6LdK1Ssv89Hthhxls"
    dest_cluster_file_id = "17DMA5-fWipMqQPGCNYD_cXIIuIphTB8Y"
    route_cluster_file_id = "1H8I0YOC6zIIHARBIkumuxcVe0njoo04g"

    base_url = "https://drive.google.com/uc?id="

    # Construct URLs
    airline_url = f"{base_url}{airline_delay_file_id}"
    route_url = f"{base_url}{route_density_file_id}"
    dest_cluster_url = f"{base_url}{dest_cluster_file_id}"
    route_cluster_url = f"{base_url}{route_cluster_file_id}"

    # Read CSVs
    airline_delay_lookup = pd.read_csv(airline_url)
    route_density_lookup = pd.read_csv(route_url)
    dest_cluster_lookup = pd.read_csv(dest_cluster_url)
    route_cluster_lookup = pd.read_csv(route_cluster_url)

    return airline_delay_lookup, route_density_lookup, dest_cluster_lookup, route_cluster_lookup

# Load all four lookup tables
airline_delay_lookup, route_density_lookup, dest_cluster_lookup, route_cluster_lookup = load_lookups_from_drive()   

# ------- Mapping Functions
def map_airline_delay_features(airline_name):
    row = airline_delay_lookup[airline_delay_lookup['airline_name'] == airline_name]
    return (
        row.iloc[0]['airline_avg_arr_delay'] if not row.empty else 0,
        row.iloc[0]['airline_avg_dep_delay'] if not row.empty else 0
    )

def map_route_density(route):
    row = route_density_lookup[route_density_lookup['route'] == route]
    return row.iloc[0]['route_density'] if not row.empty else 0

def map_dest_cluster(dest):
    row = dest_cluster_lookup[dest_cluster_lookup['dest'] == dest]
    return row.iloc[0]['dest_cluster'] if not row.empty else 0

def map_route_cluster(route):
    row = route_cluster_lookup[route_cluster_lookup['route'] == route]
    return row.iloc[0]['route_cluster'] if not row.empty else 0    

def get_time_block(dep_hour):
    if 0 <= dep_hour < 6:
        return "12am–6am"
    elif 6 <= dep_hour < 9:
        return "6am–9am"
    elif 9 <= dep_hour < 12:
        return "9am–12pm"
    elif 12 <= dep_hour < 15:
        return "12pm–3pm"
    elif 15 <= dep_hour < 18:
        return "3pm–6pm"
    elif 18 <= dep_hour < 21:
        return "6pm–9pm"
    else:
        return "9pm–12am"

# Define mapping based on delay trend
time_score_map = {
    "12am–6am": 1, "6am–9am": 2, "9am–12pm": 3, "12pm–3pm": 4,
    "3pm–6pm": 5, "6pm–9pm": 6, "9pm–12am": 7
}

month_score_map = {
    'Sep': 1, 'Oct': 2, 'Nov': 3, 'Jan': 4, 'Feb': 5, 'Mar': 6, 'May': 7,
    'Aug': 8, 'Apr': 9, 'Jun': 10, 'Jul': 11, 'Dec': 12
}

dow_score_map = {
    'Thu': 1, 'Fri': 2, 'Mon': 3, 'Sun': 4,
    'Sat': 5, 'Wed': 6, 'Tue': 7
}        
   
# ------------ Preprocessing User's Input ----------
# Cache pipeline loading for efficiency
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(os.path.dirname(__file__), "..", "logreg_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        return cloudpickle.load(f)

def preprocess_user_input(user_input_dict):
    # Extract airline and route
    airline = user_input_dict.pop("airline_name")
    route = user_input_dict.pop("route")

    # Split route into origin and destination
    origin, dest = route.split(" - ")

    # Get distance haul
    #dist_haul = user_input_dict.get("dist_haul", 'Short')

    # Map airline and route-level features
    arr_avg, dep_avg = map_airline_delay_features(airline)
    route_density = map_route_density(route)
    dest_cluster = map_dest_cluster(dest)
    route_cluster = map_route_cluster(route)

    # Derived feature: is_redeye
    dep_hour = user_input_dict.get("dep_hour", 12)
    is_redeye = int(dep_hour >= 22 or dep_hour <= 5)

    # Time block → score
    time_block = get_time_block(dep_hour)
    time_block_score = time_score_map.get(time_block, 4)  

    # Month & Day of Week scores
    month = user_input_dict.get("month", "Jan")
    day_of_week = user_input_dict.get("day_of_week", 'Mon')

    month_score = month_score_map.get(month, 6)
    dow_score = dow_score_map.get(day_of_week, 4)

    # Add all new features
    user_input_dict.update({
        "origin": origin,
        "dest": dest,
        #"dist_haul": dist_haul,
        "airline_avg_arr_delay": arr_avg,
        "airline_avg_dep_delay": dep_avg,
        "route_density": route_density,
        "dest_cluster": dest_cluster,         
        "route_cluster": route_cluster,
        "is_redeye": is_redeye,
        "time_block_score": time_block_score,
        "month_delay_score": month_score,
        "dow_delay_score": dow_score
    })

    # Convert to DataFrame
    df = pd.DataFrame([user_input_dict])

    return df

def predict_with_pipeline(df):
    pipeline = load_pipeline()
    try:
        predictions = pipeline.predict(df)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ------ SHAP Values -------
def get_shap_values(df_input, pipeline_path="logreg_pipeline.pkl"):
    with open(pipeline_path, "rb") as f:
        pipeline = cloudpickle.load(f)

    preprocessor = pipeline.named_steps['preprocessor']
    multi_clf = pipeline.named_steps['classifier']

    # Transform input to numeric features
    X_transformed = preprocessor.transform(df_input)

    shap_values_dict = {}

    # Explain each estimator inside MultiOutputClassifier
    for i, estimator in enumerate(multi_clf.estimators_):
        explainer = shap.LinearExplainer(estimator, X_transformed, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_transformed)
        shap_values_dict[i] = shap_values

    # Get feature names after preprocessing (if possible)
    try:
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(cols)
            else:
                names = cols
            feature_names.extend(names)
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    return shap_values_dict, feature_names


def generate_recommendations(shap_values_dict, feature_names, df_input, model_preds, threshold=0.01):
    all_recommendations = {}

    # Extract input values once
    val_dep_hour = df_input["dep_hour"].values[0]
    #val_dist_haul = df_input["dist_haul"].values[0]
    val_month = df_input["month_delay_score"].values[0]
    val_day = df_input["dow_delay_score"].values[0]
    val_airline_arr = df_input["airline_avg_arr_delay"].values[0]
    val_airline_dep = df_input["airline_avg_dep_delay"].values[0]
    val_route_density = df_input["route_density"].values[0]
    val_is_redeye = df_input["is_redeye"].values[0]

    feature_msgs = {
        "dep_hour": (
            f"Your flight's scheduled departure hour ({val_dep_hour}:00) makes it a candidate for delay.",
            "Consider booking flights earlier or later to avoid peak delay times."
        ),
        "is_redeye": (
            "Your flight is a red-eye flight, which tends to have a higher risk of delay." if val_is_redeye else "Your flight is not a red-eye flight, which usually helps avoid delays.",
            "If possible, consider non-red-eye flights for better punctuality."
        ),
        "airline_avg_arr_delay": (
            f"The airline you chose has an average arrival delay of {val_airline_arr:.1f} minutes historically.",
            "Trying a different airline might reduce your delay risk."
        ),
        "airline_avg_dep_delay": (
            f"The airline you chose has an average departure delay of {val_airline_dep:.1f} minutes historically.",
            "Trying a different airline might reduce your delay risk."
        ),
        "route_density": (
            f"This route has a traffic density score of {val_route_density}, indicating heavy traffic which can increase delay chances.",
            "Flying on less busy routes could improve your chances of on-time flights."
        ),
        "month_delay_score": (
            "This month tends to experience more delays historically.",
            "If your travel is flexible, consider off-peak months."
        ),
        "dow_delay_score": (
            "Flights on this day of the week tend to be more prone to delays.",
            "Traveling on less busy days may reduce delay risk."
        ),
        #"dist_haul": (
           # f"Your flight is classified as a '{val_dist_haul}' haul, which affects delay likelihood.",
            #"Sometimes shorter or longer haul flights have different risk patterns."
       # ),
    }

    for output_index, shap_values in shap_values_dict.items():
        pred = model_preds[output_index]
        shap_frame = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0]
        })

        if pred == 1:
            # Get feature with highest absolute SHAP value above threshold
            top_feature = shap_frame.loc[
                shap_frame['shap_value'].abs() > threshold
            ].sort_values(by="shap_value", key=abs, ascending=False).head(1)

            if not top_feature.empty:
                feat = top_feature.iloc[0]['feature']
                if feat in feature_msgs:
                    msg1, msg2 = feature_msgs[feat]
                    all_recommendations[output_index] = [
                        f"• {msg1}",
                        f"  👉 {msg2}"
                    ]
                else:
                    all_recommendations[output_index] = ["Delay risk identified, but no specific recommendation available."]
            else:
                all_recommendations[output_index] = ["Delay predicted, but no major risk factor stood out."]
        else:
            all_recommendations[output_index] = ["No major delay factors identified."]

    return all_recommendations     

# ------------ USER INPUT FORM -----------
with st.form("flight_form"):
    airline_name = st.selectbox("Airline", sorted(airline_delay_lookup["airline_name"].unique()))
    route = st.selectbox("Route", sorted(route_density_lookup["route"].unique()))
    month = st.selectbox("Month", list(month_score_map.keys()))
    day_of_week = st.selectbox("Day of Week", list(dow_score_map.keys()))
    dep_hour = st.number_input("Scheduled Departure Hour (0–23)", min_value=0, max_value=23, value=12)
    #distance_group = st.selectbox("Distance", ["Short", "Medium", "Long"])

    submitted = st.form_submit_button("Predict Delay")
if submitted:
    user_input = {
        "airline_name": airline_name,
        "route": route,
        "month": month,
        "day_of_week": day_of_week,
        "dep_hour": dep_hour,
        #"dist_haul": distance_group
    }

    df_input = preprocess_user_input(user_input)

    with st.spinner("Predicting delay, please wait..."):
        prediction = predict_with_pipeline(df_input)  # e.g. [[1, 0]]

    if prediction is not None:
        pred_dep, pred_arr = prediction[0]  # unpack departure and arrival delay predictions

        st.markdown("### ✈️ Prediction Result")
        st.write(f"**Departure delay predicted:** {'🟥 Yes' if pred_dep == 1 else '🟩 No'}")
        st.write(f"**Arrival delay predicted:** {'🟥 Yes' if pred_arr == 1 else '🟩 No'}")

        shap_vals, feature_names = get_shap_values(df_input)

        # Pass model predictions to recommendations
        recs_dict = generate_recommendations(
            shap_vals, feature_names, df_input, model_preds=prediction[0]
        )

        st.subheader("✈️ Recommendations Based on Prediction")

        labels = {0: "Departure Delay", 1: "Arrival Delay"}

        for i, recs in recs_dict.items():
            st.markdown(f"**🔹 {labels.get(i, f'Output {i}')}**")
            for r in recs:
                st.write(r)

        st.markdown("### 📊 Raw SHAP Values for Departure Delay Model")
        # shap_vals[0] corresponds to departure delay model's SHAP values
        shap_dep = shap_vals[0][0]
        shap_dep_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_dep
        }).sort_values(by="SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_dep_df)

        st.markdown("### 📊 Raw SHAP Values for Arrival Delay Model")
        # shap_vals[1] corresponds to arrival delay model's SHAP values
        shap_arr = shap_vals[1][0]
        shap_arr_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_arr
        }).sort_values(by="SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_arr_df)
