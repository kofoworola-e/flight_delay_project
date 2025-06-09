# ✈️ Understanding Flight Delays in the U.S.

##### *A real-world ML project using 327k+ U.S. flight records to uncover delay patterns and predict late departures and arrivals using a MultiOutput Logistic Regression model.*

---

## 🔦 Summary

This project began as part of a [DataCamp competition](https://app.datacamp.com/learn/competitions/understanding-flight-delays), where I was challenged to explore and predict flight delays using a large U.S. domestic flight dataset.

I extended the challenge by:

✅ Performing full **EDA**, **feature engineering**, and **modeling** in Jupyter

✅ Building a **MultiOutput classification model** for predicting both departure and arrival delays

✅ Designing and deploying a **Streamlit app** to share data-driven insights and make live predictions

Key insights include delay patterns across **routes**, **airlines**, and **times of day**, plus an interactive tool to explore and predict delays based on flight details.

The app is deployed and available here:
👉 [flightdelayanalytics.streamlit.app](https://flightdelayanalytics.streamlit.app)

---
## 🧰 Tech Stack & Tools

<p align="left"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge"/> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Badge"/> <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib Badge"/> <img src="https://img.shields.io/badge/Seaborn-0D4068?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn Badge"/> <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Badge"/> <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/> <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Badge"/> <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab Badge"/> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge"/> </p>

## 📌 Project Overview

Originally part of a DataCamp Competition on Understanding Flight Delays, I decided to take it a step further. I wanted to explore delay patterns more deeply, build a prediction tool, and turn it all into a user-friendly app.

The project is split into two parts:

1. **Jupyter Notebook** for full EDA, feature engineering, and model building
2. **Streamlit App** for interactive insights and real-time delay prediction

Along the way, I also got hands-on practice with **MultiOutput classification**, evolving a simple task into a dual-target model for both **departure** and **arrival** delays.

---

## 🎯 Goals

### Business Goals

* Help minimize the impact of delays on travelers and airline operations
* Surface key risk factors by time, route, and carrier
* Provide a prediction tool for users to anticipate delays ahead of time

### Data Science Goals

* Explore delay patterns across airlines, months, weekdays, and time blocks
* Build a multi-output model to predict both departure and arrival delays ≥15 mins
* Visualize insights clearly and deploy an easy-to-use prediction tool
* Gain practical experience with multi-label modeling and deployment

---

## 💾 Dataset Summary

* Source: [Kaggle Dataset – flights.csv](https://www.kaggle.com/datasets/mahoora00135/flights/data)
* Includes 327,000+ completed domestic U.S. flights in 2023.

**Key Features:**

* Date & time breakdowns (month, day, hour, etc.)
* Departure/arrival delays (in minutes)
* Airline and airport codes
* Engineered delay flags (`dep_delayed_15`, `arr_delayed_15`)
* Routes and traffic volume

---

## 📓 Notebook: EDA & Modeling

Located in: `notebooks/flight_delay_analysis.ipynb`

### Highlights

* Frontier and ExpressJet had the highest delay rates
* Delays spike in **June, July, December**, and **after 3 PM daily**
* Newark (EWR) was a major hub for delay-prone routes
* Most punctual carriers: **Hawaiian Airlines**, **Alaska Airlines**

### Features & Targets

* Categorical features (airline, route, time of day)
* Binary targets: `dep_delayed_15`, `arr_delayed_15`
* MultiOutputClassifier used for training

---

## 🤖 Model Summary

The deployed model is a **Logistic Regression** classifier wrapped in a **MultiOutputClassifier**.

| Metric    | Departure Delay | Arrival Delay |
| --------- | --------------- | ------------- |
| Recall    | 0.62            | 0.61          |
| Precision | 0.33            | 0.34          |
| F1 Score  | 0.43            | 0.44          |
| PR AUC    | 0.37            | 0.37          |

* **Hamming Loss:** 0.33
* **Subset Accuracy:** 0.59
* **Average PR AUC:** 0.37

> This baseline performs reasonably well in terms of recall — useful for alerting users to potential delays.

---

## 🌐 Streamlit App
Deployed at: [flightdelayanalytics.streamlit.app](https://flightdelayanalytics.streamlit.app)

Located in: `streamlit_app/Home.py`

Dependencies in: `streamlit_app/requirements.txt`

### 🔍 App Features

#### ✈️ Delay Explorer

* Delay trends by **month**, **weekday**, **hour**, **airport**, and **airline**
* Interactive visuals for identifying high-risk routes

#### 🔮 Delay Predictor

* Input flight details (airline, origin, destination, time)
* Get predictions for **departure and arrival delays** ≥15 minutes
* Backed by the logistic regression model

---

## ⚙️ How to Run Locally

``` bash
#  Clone the Repository
git clone https://github.com/kofoworola-e/flight_delay_project.git
cd flight_delay_project

# Set Up Environment
conda create -n flight-delay python=3.10 -y
conda activate flight-delay

pip install -r streamlit_app/requirements.txt

# Run Streamlit app
streamlit run streamlit_app/Home.py

# Open Jupyter notebook
jupyter notebook notebooks/flight_delay_analysis.ipynb
```

---

## 📁 Folder Structure

```
flight_delay_project/
│
├── streamlit_app/              ← Streamlit app files
│   ├── Home.py                 ← Main entry point
│   └── requirements.txt        ← App dependencies
│
├── notebooks/                  ← Data analysis & modeling
│   └── flight_delay_analysis.ipynb
│             
├── data/                       ← Raw and cleaned datasets
└── README.md                   ← You're here!
```

---

## 🔮 Future Plans

1. **Context-Aware Recommendations**
   Integrate tailored suggestions in the app (e.g., "Consider early morning flights" or "Avoid EWR–CAE on Fridays") using NLP or LLMs to generate user-friendly advice based on predicted risks.

2. **Model Improvement**
  Experiment with advanced models (e.g., Random Forest, XGBoost, or stacking) and hyperparameter tuning to boost **Recall** and **PR-AUC**. I also plan to revisit **feature engineering** to create more informative and predictive variables that can enhance the model’s ability to detect delay risks.

3. **Deploy Enhancements**
   The app is already live, but I plan to improve its responsiveness, error handling, and UI polish (e.g., airline logos, route maps).

4. **API Integration**
   Build a lightweight REST API or use tools like **FastAPI** to expose the prediction engine for external use or integration into mobile/booking platforms.

5. **Real-Time Data Support**
   Explore integrating live flight status APIs (e.g., FAA or FlightAware) to enrich predictions with real-time context such as weather, gate changes, or air traffic alerts.

6. **Continuous Learning Pipeline**
   Implement a feedback loop where new data can be used to retrain and recalibrate the model periodically to maintain prediction quality over time.

---

## 📚 Lessons & Reflections

Working on this project was both enriching and exciting. Here are some key things I learned and reflected on:

### ✅ Technical Growth

* I got hands-on experience with **MultiOutput classification**, transitioning from single-label prediction to a more realistic dual-label setup for both departure and arrival delays.
* I deepened my understanding of how to **balance model metrics** — especially when working with imbalanced data and needing to prioritize **recall** over other scores in high-stakes use cases like delay detection.
* Building the Streamlit app helped me strengthen my skills in **turning raw models into usable tools** that non-technical users can interact with.

### 🧠 Insights Gained

* Delays often follow **predictable patterns** across time of day, seasons, routes, and specific carriers — and even a simple model can reveal useful, actionable insights.
* Visual storytelling is just as important as the model — I found that clear, interactive charts help surface patterns that even good metrics can hide.

### 💡 Reflections

* This project reminded me how valuable it is to **go beyond the minimum**. Starting from a competition brief and turning it into a full product taught me a lot about the **end-to-end data science lifecycle**.
* I enjoyed blending analysis, modeling, and app design — and I’m looking forward to exploring how **LLMs and real-time data APIs** can further enhance this tool.

---

## 🙋🏽‍♀️ About Me

I’m **Kofoworola Egbinola**, a passionate, self-taught data scientist blending numbers, narratives, and intuition to solve real-world problems.

Connect with me: [LinkedIn](https://www.linkedin.com/in/kofoworola-egbinola-m)

> This project was created with ❤️ and curiosity.
