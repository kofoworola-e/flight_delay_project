import streamlit as st
import os
import pandas as pd
import gdown
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ----- Streamlit Page Config -----
st.set_page_config(
    page_title="Flight Delay EDA",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----- Title and Introduction -----
st.title("📊 FLIGHT DELAY DATA ANALYSIS")

st.markdown("""
✈️ **Welcome to the Flight Delay Explorer!**  

Imagine planning a cross-country trip in 2023. Which airlines should you trust to be on time? How often do flights actually get delayed?  

We analyzed **327,346 flights** across the United States throughout the entire year — uncovering real stories hidden behind the numbers.
""")

@st.cache_data
def load_data():
    output = "flight_data.csv"
    if not os.path.exists(output):
        url = "https://drive.google.com/uc?id=1-2YlSUqC4XE_DIOanrZabDWHTm1j_FSp"
        gdown.download(url, output, quiet=True)
    df = pd.read_csv(output)
    return df

# ----- Load Data -----
flights_df = load_data()

# --------- Customizations ----------
# ----- Custom Color Palette -----
custom_palette = [
    "#05192d",  # navy (highlight)
    "#03ef62",  # green (highlight)
    "#d9d9e2",  # grey400 (neutral)
    "#e8e8ea",  # grey300 (neutral)
    "#333333",  # dark grey (text),
    "#efefef"
]

# ----- Dataset Summary -----
st.markdown("---")
st.markdown("### ➡️ What’s in the Dataset?")

st.markdown("""
Our dataset captures flights from 3 major airports to 104 destinations, operated by 16 airlines, over 12 months and all days of the week.
""")

st.markdown("#### Overview of Key Features")
features_df = pd.DataFrame({
    'Feature': ['Origin', 'Destination', 'Airline Name', 'Month', 'Day of Week'],
    'Unique Values': [3, 104, 16, 12, 7],
    'Most Frequent': ['Newark (EWR)', 'Atlanta (ATL)', 'United Airlines Inc.', 'August', 'Saturday'],
    'Frequency': [117127, 16837, 57782, 28756, 49301]
})
st.dataframe(features_df)

# ----- Flight Delay Reality Check -----
st.markdown("---")
st.markdown("### ✈️ How Common Are Flight Delays?")

st.markdown("""
Let’s face it — delays happen. But how often?  
Here’s a quick reality check on departure and arrival delays, especially those that are 15 minutes or longer, which is the standard industry threshold for being “late.”
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Departure Delays", "39% delayed", "61% on-time")
    st.metric("Departure Delays ≥ 15 min", "22% delayed", "78% on-time")
with col2:
    st.metric("Arrival Delays", "41% delayed", "59% on-time")
    st.metric("Arrival Delays ≥ 15 min", "24% delayed", "76% on-time")

st.markdown("""
➡️ While most flights are punctual, roughly **1 in 4 flights** face delays that can disrupt plans. 
But wait — do these delays vary by season or day of the week? Let's take a closer look at how delay rates shift across time.
""")
st.markdown("---")

# ----------Slides Tracker ------
# Slide tracker for Time Patterns
if "slide_1" not in st.session_state:
    st.session_state.slide_1 = 1

# Slide tracker for Airline-Level Performance
if "slide_2" not in st.session_state:
    st.session_state.slide_2 = 1

# Slide tracker for Airports Performance
if "slide_3" not in st.session_state:
    st.session_state.slide_3 = 1    

# ------------ 1. Time Patterns in Flight Delays -------------
st.markdown("### ⏰ Timing and Delays: Uncovering Time-Based Risk Patterns")
st.markdown("""
Flight delays often follow predictable time-based trends.  
We analyzed delay rates by **month**, **day of the week**, and **time of day** to identify when disruptions are most and least likely to occur.  

These insights can help travelers plan smarter — and support data-driven decisions in scheduling and operations.
""")

# Prepare data
# Define correct 3-letter orderings
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

time_order = ['12am–6am', '6am–9am', '9am–12pm', '12pm–3pm', '3pm–6pm', '6pm–9pm', '9pm–12am']

# Ensure the columns are categorical with the correct order
flights_df['month'] = pd.Categorical(flights_df['month'], categories=month_order, ordered=True)
flights_df['day_of_week'] = pd.Categorical(flights_df['day_of_week'], categories=dow_order, ordered=True)
flights_df['time_block'] = pd.Categorical(flights_df['time_block'], categories=time_order, ordered=True)

# Monthly Delay Trends
monthly_delay = flights_df.groupby('month', observed=False).agg({
    'dep_delayed_15': 'mean',
    'arr_delayed_15': 'mean'
}).reset_index()

# DOW Delay Trends
dow_delay = flights_df.groupby('day_of_week', observed=False).agg({
    'dep_delayed_15': 'mean',
    'arr_delayed_15': 'mean'
}).reset_index()

# Hourly Delay Trends
time_delay = (
    flights_df.groupby('time_block', observed=False)[['dep_delayed_15', 'arr_delayed_15']]
    .mean().round(2)
    .reset_index()
)

# -------- Slide 1: Monthly Delay Trends Plot -------------
def time_slide_1(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(data=monthly_delay, x='month', y='dep_delayed_15', label='Departure Delay ≥15 min', ax=ax, color=custom_palette[0])
    sns.lineplot(data=monthly_delay, x='month', y='arr_delayed_15', label='Arrival Delay ≥15 min', ax=ax, color=custom_palette[1])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncols=2, bbox_to_anchor=(0.7, 1.05))

    ax.set_ylabel("Proportion of Flights Delayed ≥15 min", fontsize=10)
    ax.set_xlabel("Month")

    return fig

# -------- Slide 2: Weekly Delay Patterns Plot -------------
def time_slide_2(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=dow_delay, x='day_of_week', y='dep_delayed_15', label='Departure Delay ≥15 min', ax=ax, color=custom_palette[0])
    sns.lineplot(data=dow_delay, x='day_of_week', y='arr_delayed_15', label='Arrival Delay ≥15 min', ax=ax, color=custom_palette[1])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncols=2, bbox_to_anchor=(0.7, 1.08))

    ax.set_ylabel("Proportion of Flights Delayed ≥15 min", fontsize=10)
    ax.set_xlabel("Day of Week")

    return fig    

# -------- Slide 3: Hourly Delay Patterns Plot -------------
def time_slide_3(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(data=time_delay, x='time_block', y='dep_delayed_15', label='Departure Delay ≥15 min', ax=ax, color=custom_palette[0])
    sns.lineplot(data=time_delay, x='time_block', y='arr_delayed_15', label='Arrival Delay ≥15 min', ax=ax, color=custom_palette[1])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncols=2, bbox_to_anchor=(0.7, 1.05))

    ax.set_ylabel("Proportion of Flights Delayed ≥15 min", fontsize=10)
    ax.set_xlabel("Time Block")

    return fig        

# --------- Navigation buttons
col1a, _, col2a = st.columns([1, 6, 1])
with col1a:
    if st.button("◀️ Back", key="back_a") and st.session_state.slide_1 > 1:
        st.session_state.slide_1 -= 1
with col2a:
    if st.button("Next ▶️", key="next_a") and st.session_state.slide_1 < 3:
        st.session_state.slide_1 += 1

if st.session_state.slide_1 == 1:
    st.markdown("##### 🗓️ Monthly Delay Trends: Tracking Seasonal Peaks and Dips")
    st.write("How do delays shift throughout the year?")
    
    fig = time_slide_1(monthly_delay)
    st.pyplot(fig)

    st.write("""
    - Delays **peak in summer (June–July)** and **December**, driven by **high travel demand** and **weather-related disruptions**.
    - The **fall months (September–November)** see the **lowest delay rates**, thanks to calmer weather and lighter travel loads.
    """)

    st.write("""
    Let’s zoom in from months to **days of the week** to uncover more timing-based insights.
    """)

elif st.session_state.slide_1 == 2:
    st.markdown("##### 📅 Weekly Delay Patterns: Midweek Mayhem, Thursday Calm")
    st.write("Which days of the week see the most or least delays?")

    fig = time_slide_2(dow_delay)
    st.pyplot(fig)

    st.write("""
    - **Tuesdays and Wednesdays** are the most delay-prone, likely due to **business travel surges** and midweek congestion.
    - **Thursdays** are the most punctual, offering a sweet spot before the weekend rush.
    """)

    st.write("""
    Now, let’s drill down to **hourly patterns** — when in the day are delays most likely?
    """)

elif st.session_state.slide_1 == 3:
    st.markdown("##### 🕑 Hourly Delay Patterns: Evening Rush vs. Early Bird Advantage")
    st.write("Do delays depend on what time of day you fly?")

    fig = time_slide_3(time_delay)
    st.pyplot(fig)

    st.write("""
    - **Delays climb steadily throughout the day**, peaking during **evening hours (6 PM – 12 AM)** due to cascading operational delays.
    - **Early morning flights (12 AM – 9 AM)** are most reliable — low traffic, rested crews, and clean schedules all contribute.
    """)

    st.write("""
    ###### 🧾 Summary & Recommendations

    **For Travelers:**  
    - Your best bet for on-time flights? **Book early departures**, especially on **Thursdays**.
    - Avoid **evening flights** and **midweek peaks** (Tuesdays & Wednesdays) when delays are most likely.

    **For Airlines & Airports:**  
    - Use these patterns to **optimize flight schedules**, **adjust staffing**, and **streamline operations** during high-delay windows.
    - Prioritizing early-day efficiency and midweek resilience could greatly improve on-time performance and customer satisfaction.
    """)

    st.markdown("""
    **But when and how often delays occur isn't the full story —**  
    Let’s now explore how **different airlines** stack up in delay performance.
    """)

st.markdown("---")    

# ------------------ 2. AIRLINE-LEVEL PERFORMANCE --------------------
st.markdown("### 🛫 Airline-Level Performance: Who’s On Time?")
st.markdown("""
**Which airlines are the champions of punctuality, and which are the frequent offenders?**  
We dug into flight performance across all carriers to uncover how often planes are delayed.

Each airline’s performance is visualized across six key slides:
- Overall delay levels
- The biggest offenders
- The most punctual players
- Breakdown of delay types (Departure vs Arrival)
- Delay composition for the worst performers
- Delay composition for the most reliable

Let’s dive into the skies of data to see who’s making passengers wait—and who’s keeping them moving.
""")

# ----- Prepare data for airline delay analysis -----
airline_delay = flights_df.groupby('airline_name').agg({
    'dep_delayed_15': 'mean',
    'arr_delayed_15': 'mean',
}) * 100 

airline_delay = airline_delay.round(1).reset_index().rename(columns={
    'airline_name': 'Airline Name',
    'dep_delayed_15': 'Departure Delay ≥15m',
    'arr_delayed_15': 'Arrival Delay ≥15m'
})

airline_delay['Total Delay'] = airline_delay['Departure Delay ≥15m'] + airline_delay['Arrival Delay ≥15m']
airline_delay['Dep %'] = round(airline_delay['Departure Delay ≥15m'] / airline_delay['Total Delay'] * 100, 2)
airline_delay['Arr %'] = round(airline_delay['Arrival Delay ≥15m'] / airline_delay['Total Delay'] * 100, 2)

airline_delay = airline_delay.sort_values('Total Delay', ascending=False)

#st.dataframe(airline_delay)
# -------- Slide 1: Total Delay Trends -------------
def airline_slide_1(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(airline_delay['Airline Name'], airline_delay['Total Delay'], color=custom_palette[-4])

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | TOTAL FLIGHTS DELAYED", fontsize=9)
    ax.xaxis.set_label_coords(0.09, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig

# ---------- Slide 2: Top Latecomers overall
def airline_slide_2(df):
    top_4 = airline_delay['Total Delay'].head(4).index

    colors= []
    for i in airline_delay.index:
        if i in top_4:
            colors.append('grey')  # navy
        else:
             colors.append(custom_palette[-4])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(airline_delay['Airline Name'],
     airline_delay['Total Delay'], color=colors)         

    # Add total delay annotation
    for i in top_4:
        airline = airline_delay.loc[i, 'Airline Name']
        total = airline_delay.loc[i, 'Total Delay']
        ax.text(
            total - 6, airline, f"{total:.0f}%",
            va='center', ha='left',
            color='white', fontsize=9
        )  

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | TOTAL FLIGHTS DELAYED", fontsize=9)
    ax.xaxis.set_label_coords(0.09, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig         

 # ---------- Slide 3: Top Puntuals overall
def airline_slide_3(df):
    bottom_4 = airline_delay['Total Delay'].tail(4).index

    colors= []
    for i in airline_delay.index:
        if i in bottom_4:
            colors.append('grey')  # navy
        else:
             colors.append(custom_palette[-4])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(airline_delay['Airline Name'],
     airline_delay['Total Delay'], color=colors)         

    # Add total delay annotation
    for i in bottom_4:
        airline = airline_delay.loc[i, 'Airline Name']
        total = airline_delay.loc[i, 'Total Delay']
        ax.text(
            total - 6, airline, f"{total:.0f}%",
            va='center', ha='left',
            color='white', fontsize=9
        )  

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | TOTAL FLIGHTS DELAYED", fontsize=9)
    ax.xaxis.set_label_coords(0.09, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig    

# ------Slide 4: Delay Type
def airline_slide_4(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Departure % first
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Dep %'], 
        label='Departure Delay ≥15m',
        color=custom_palette[2]
    )

    # Then plot Arrival % on top of Departure %
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Arr %'], 
        left=airline_delay['Dep %'], 
        label='Arrival Delay ≥15m',
        color=custom_palette[3]
    )  
         
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | DEPARTURE DELAY ≥15M | ARRIVAL DELAY ≥15M", fontsize=9)
    ax.xaxis.set_label_coords(0.21, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig   

# ---------- Slide 5-----
def airline_slide_5(df):
    # ----- Highlight top 4 airlines -----
    top_4 = airline_delay.head(4).index

    dep_colors, arr_colors = [], []
    for i in airline_delay.index:
        if i in top_4:
            dep_colors.append(custom_palette[0])  # navy
            arr_colors.append(custom_palette[1])  # green
        else:
            dep_colors.append(custom_palette[2])  # grey400
            arr_colors.append(custom_palette[3])  # grey300

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Departure % first
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Dep %'], 
        label='Departure Delay ≥15m',
        color=dep_colors
    )

    # Then plot Arrival % on top of Departure %
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Arr %'], 
        left=airline_delay['Dep %'], 
        label='Arrival Delay ≥15m',
        color=arr_colors
    )  

    # Add total delay annotation
    for i in top_4:
        airline = airline_delay.loc[i, 'Airline Name']
        dep = airline_delay.loc[i, 'Dep %']
        arr = airline_delay.loc[i, 'Arr %']
        y = airline
        if dep > 1:
            ax.text(0.5, y, f"{dep:.0f}%", va='center', ha='left',
             color='white', fontsize=9)
        if arr > 1:
            ax.text(dep + 0.5, y, f"{arr:.0f}%", va='center', ha='left',
             color='white', fontsize=9)        

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | DEPARTURE DELAY ≥15M | ARRIVAL DELAY ≥15M", fontsize=9)
    ax.xaxis.set_label_coords(0.21, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig   

# ---------- Slide 6-----
def airline_slide_6(df):
    # ----- Highlight bottom 4 airlines -----
    bottom_4 = airline_delay.tail(4).index

    dep_colors, arr_colors = [], []
    for i in airline_delay.index:
        if i in bottom_4:
            dep_colors.append(custom_palette[0])  # navy
            arr_colors.append(custom_palette[1])  # green
        else:
            dep_colors.append(custom_palette[2])  # grey400
            arr_colors.append(custom_palette[3])  # grey300

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Departure % first
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Dep %'], 
        label='Departure Delay ≥15m',
        color=dep_colors
    )

    # Then plot Arrival % on top of Departure %
    ax.barh(
        airline_delay['Airline Name'], 
        airline_delay['Arr %'], 
        left=airline_delay['Dep %'], 
        label='Arrival Delay ≥15m',
        color=arr_colors
    )  

    # Add total delay annotation
    for i in bottom_4:
        airline = airline_delay.loc[i, 'Airline Name']
        dep = airline_delay.loc[i, 'Dep %']
        arr = airline_delay.loc[i, 'Arr %']
        y = airline
        if dep > 1:
            ax.text(0.5, y, f"{dep:.0f}%", va='center', ha='left',
             color='white', fontsize=9)
        if arr > 1:
            ax.text(dep + 0.5, y, f"{arr:.0f}%", va='center', ha='left',
             color='white', fontsize=9)        

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel("AIRLINE | DEPARTURE DELAY ≥15M | ARRIVAL DELAY ≥15M", fontsize=9)
    ax.xaxis.set_label_coords(0.21, 1.07)
    ax.set_ylabel("")

    ax.margins(y=0.015)
    return fig   

# --------- Navigation buttons
col1b, _, col2b = st.columns([1, 6, 1])
with col1b:
    if st.button("◀️ Back", key="back_b") and st.session_state.slide_2 > 1:
        st.session_state.slide_2 -= 1
with col2b:
    if st.button("Next ▶️", key="next_b") and st.session_state.slide_2 < 6:
        st.session_state.slide_2 += 1

if st.session_state.slide_2 == 1:
    st.markdown("##### ✈️ Airline-Level Delay Performance")
    st.write("How do airlines compare in terms of delays?")
    
    fig = airline_slide_1(airline_delay)
    st.pyplot(fig)

    st.write("""
        There's a broad spread in total delay rates across carriers, which reveals major operational differences. Identifying top and bottom performers helps spotlight reliability gaps.
    """)

    st.write("""
        Let's now focus on **who exactly are the worst offenders** when it comes to delays.
    """)

elif st.session_state.slide_2 == 2:
    st.markdown("##### 🚨 Most Delayed Airlines")
    st.markdown("Which carriers have the highest delay rates?")

    fig = airline_slide_2(airline_delay)
    st.pyplot(fig)

    st.markdown("""
        Frontier, ExpressJet, AirTran and Mesa Airlines lead in delays, each with total delay rates above 60%. These patterns suggest persistent operational or route-specific issues.
    """)

    st.markdown("""
        But while some airlines struggle, others excel — let's now highlight the **most punctual carriers**.
    """)

elif st.session_state.slide_2 == 3:
    st.markdown("##### ✅ Most Punctual Airlines")
    st.markdown("Which airlines are consistently on time?")

    fig = airline_slide_3(airline_delay)
    st.pyplot(fig)

    st.markdown("""
        Hawaiian, Alaska, and US Airways maintain low delay rates, possibly due to less congested routes, efficient ground operations, and favorable scheduling.
    """)

    st.markdown("""
        To understand the **nature of these delays**, let's break them down by type — departure vs arrival.
    """)

elif st.session_state.slide_2 == 4:
    st.markdown("##### 📊 Delay Breakdown by Type")
    st.markdown("How do airlines perform across Departure vs Arrival delays (≥15 minutes)?")

    fig = airline_slide_4(airline_delay)
    st.pyplot(fig)  

    st.markdown("""
        Some airlines tend to struggle more with **arrival** delays than departures — or vice versa.
        Understanding where delays accumulate helps identify bottlenecks (e.g. gate availability vs boarding logistics).
    """)

    st.markdown("""
        Let's zoom in further and see how **the worst-performing airlines** stack up across these two delay types.
    """) 

elif st.session_state.slide_2 == 5:
    st.markdown("##### 🟥 Delay Split: Most Delayed Airlines")
    st.markdown("Do worst-performing airlines struggle more with departures or arrivals?")

    fig = airline_slide_5(airline_delay)
    st.pyplot(fig)

    st.markdown("""
        - **Frontier** and **AirTran** experience **disproportionately higher arrival delays**, which may point to issues like turnaround inefficiencies or destination airport constraints.
        - **ExpressJet** and **Mesa** show more balanced but still elevated delays.
    """)

    st.markdown("""
        Now, let’s flip the lens again and see how the **most punctual airlines manage both types** of delays.
    """)

elif st.session_state.slide_2 == 6:
    st.markdown("##### 🟩 Delay Split: Most Punctual Airlines")
    st.markdown("What kind of delays are most avoided by top performers?")

    fig = airline_slide_6(airline_delay)
    st.pyplot(fig)

    st.markdown("""
        - **Hawaiian Airlines** leads with the lowest departure delay rate (35%) and highest arrival delay rate (65%) among the top four.
        - **US Airways** follows with a departure delay rate of 40% and arrival delay rate of 60%.
        - **American** and **Alaska Airlines** maintain relatively balanced delay patterns, though their arrival delays still exceed 50%.
    """)

    st.markdown("""
        In summary, while top airlines perform better overall, **arrival delays remain a common challenge** — even among the most punctual carriers.
    """)

    st.markdown("""
        ###### ✅ Recommendations for Improvement:
        - **Benchmark operational practices** of Hawaiian and US Airways, particularly in departure scheduling and turnaround management.
        - **Investigate persistent arrival delays**, which could stem from external airport constraints or late inbound connections.
        - **Implement strategies to smoothen arrivals**, such as buffer scheduling or early departure leeway for high-traffic routes.
    """)

    st.markdown("""  
    **Airline performance tells part of the story — but geography plays a major role too.**  
    Let’s explore which **airports** are driving the most delays at both **departure and arrival** ends.
    """)
     
st.markdown("---")   

# ------------ 3. Airport-Level Delays -------------
st.markdown("### 🗺️ Airport-Level Delays: Where Trouble Takes Off and Lands")
st.markdown("""
**Are some airports magnets for delays?**  
We analyzed both departure and arrival delays by airport to uncover which hubs tend to cause — or suffer — the most disruption.
""")

# ---- Origin Airport Delay Trends ----
origin_airport_delay = (
    flights_df.groupby(['origin']).agg({
        'dep_delayed_15': 'mean',
        'arr_delayed_15': 'mean',
        'flight': 'count'
    }).round(2).reset_index()
)
origin_airport_delay.columns = ['Airport', 'Departure Delay ≥15m', 'Arrival Delay ≥15m', 'Total Flights']
origin_airport_delay.sort_values(by='Departure Delay ≥15m', ascending=False, inplace=True)

# ------- Destination Airport Delay Trends -------
dest_airport_delay = (
    flights_df.groupby(['dest']).agg({
        'dep_delayed_15': 'mean',
        'arr_delayed_15': 'mean',
        'flight': 'count'
    }).round(2).reset_index()
)
dest_airport_delay.columns = ['Airport', 'Departure Delay ≥15m', 'Arrival Delay ≥15m', 'Total Flights']
dest_airport_delay.sort_values(by='Departure Delay ≥15m', ascending=False, inplace=True)

# Clusters
dest_cluster_map = (
    flights_df[['dest', 'dest_cluster']]
    .drop_duplicates()
    .set_index('dest')['dest_cluster']
)

dest_airport_delay['Cluster'] = dest_airport_delay['Airport'].map(dest_cluster_map)

# Map clusters to intuitive short labels with explicit order
cluster_map = {
    4: "Very Low Delay, Low Traffic",
    3: "Low Delay, Avg Traffic", 
    1: "Low Delay, High Traffic",
    0: "Moderate Delay, Low Traffic",
    2: "High Delay, Low Traffic"
}
dest_airport_delay['Cluster Label'] = dest_airport_delay['Cluster'].map(cluster_map)

# Define the desired order explicitly
cluster_order = [
    "Very Low Delay, Low Traffic",
    "Low Delay, Avg Traffic",
    "Low Delay, High Traffic",
    "Moderate Delay, Low Traffic",
    "High Delay, Low Traffic",
]

# Make 'Cluster Label' a categorical variable with order
dest_airport_delay['Cluster Label'] = pd.Categorical(dest_airport_delay['Cluster Label'],
                                                     categories=cluster_order,
                                                     ordered=True)

# -------- Slide 1: Origin Airport Delay Trends -------------
def airport_slide_1(df):
    fig, ax = plt.subplots(figsize=(8, 3))
    # Departure bars
    ax.barh(
        origin_airport_delay['Airport'],
        origin_airport_delay['Departure Delay ≥15m'],
        label='Departure Delay ≥15m',
        color=custom_palette[0]
    )
    # Arrival bars
    ax.barh(
        origin_airport_delay['Airport'],
        origin_airport_delay['Arrival Delay ≥15m'],
        left=origin_airport_delay['Departure Delay ≥15m'],
        label='Arrival Delay ≥15m',
        color=custom_palette[1]
    )

    for i in range(len(origin_airport_delay)):
        dep = origin_airport_delay['Departure Delay ≥15m'].iloc[i]
        arr = origin_airport_delay['Arrival Delay ≥15m'].iloc[i]
        total = dep + arr
        y = i

    # Departure delay number, aligned at start of the departure bar (slightly inside)
        if dep > 0.01:
            ax.text(0.01, y, f"{dep:.0%}", va='center', ha='left', color='white', fontsize=9)

        # Arrival delay number, aligned at start of the arrival bar (which starts at dep)
        if arr > 0.01:
            ax.text(dep + 0.01, y, f"{arr:.0%}", va='center', ha='left', color='white', fontsize=9)

        # Total delay number, aligned just past the total bar
        ax.text(total + 0.01, y, f"{total:.0%}", va='center', ha='left', color='black', fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 0.61, 0.2))
    ax.set_xlim(0, 0.6)
    ax.tick_params(axis='y', left=False)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines[['left', 'right', 'bottom']].set_visible(False)
    ax.legend(ncols=2, bbox_to_anchor=(0.54, 1.6), prop={'size': 9})

    #ax.set_xlabel("AIRPORT | DEPARTURE DELAY ≥15M | ARRIVAL DELAY ≥15M", fontsize=9)
    ax.xaxis.set_label_coords(0.186, 1.25)
    ax.set_ylabel("")

    plt.tight_layout()

    return fig

# -------- Slide 2: Destination Airport Delay Trends -------------
def airport_slide_2(df):
   
    fig = px.scatter(
    dest_airport_delay,
    x='Departure Delay ≥15m',
    y='Arrival Delay ≥15m',
    size='Total Flights',
    color='Total Flights',
    hover_name='Airport',
    size_max=45,
    labels={
        'Departure Delay ≥15m': 'Departure Delay Rate',
        'Arrival Delay ≥15m': 'Arrival Delay Rate',
        "Total Flights": "Number of Flights"
    },
    color_continuous_scale='Blues'
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>" +
            "Departure Delay: %{x:.0%}<br>" +
            "Arrival Delay: %{y:.0%}<br>" +
            "Number of Flights: %{marker.size:,}<extra></extra>"
        ),
        marker=dict(opacity=0.8, 
            line=dict(width=1, color='DarkSlateGrey'))
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickformat='.0%', range=[0, 0.6]),
        yaxis=dict(tickformat='.0%', range=[0, 0.6]),
        height=600
    )

    return fig   

# -------- Slide 3: Destination Airport Cluster Trends -------------
def airport_slide_3(dest_airport_delay, cluster_order):
    # 👇 Cluster selector using Streamlit
    selected_cluster = st.selectbox("Select a cluster:", ["All Clusters"] + cluster_order)

    # Set up colors
    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    max_flights = dest_airport_delay['Total Flights'].max()

    # Filter clusters based on selection
    clusters_to_plot = cluster_order if selected_cluster == "All Clusters" else [selected_cluster]

    for i, cluster in enumerate(cluster_order):
        if cluster not in clusters_to_plot:
            continue

        df_cluster = dest_airport_delay[dest_airport_delay['Cluster Label'] == cluster]

        fig.add_trace(go.Scatter(
            x=df_cluster['Departure Delay ≥15m'],
            y=df_cluster['Arrival Delay ≥15m'],
            mode='markers',
            marker=dict(
                size=df_cluster['Total Flights'],
                sizemode='area',
                sizeref=2.*max_flights/(50.**2.25),
                sizemin=4,
                color=colors[i % len(colors)],
                line=dict(width=1, color='black'),
                opacity=0.7
            ),
            name=cluster,
            customdata=df_cluster[['Airport', 'Departure Delay ≥15m', 'Arrival Delay ≥15m', 'Total Flights']].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Departure Delay ≥15m: %{customdata[1]:.0%}<br>" +
                "Arrival Delay ≥15m: %{customdata[2]:.0%}<br>" +
                "Total Flights: %{customdata[3]:,}<extra></extra>"
                )
        ))

    # Final layout tweaks
    fig.update_layout(
        xaxis_title='Departure Delay ≥15 min',
        yaxis_title='Arrival Delay ≥15 min',
        margin=dict(r=50),
        xaxis=dict(tickformat='.0%', range=[-0.01, 0.70]),
        yaxis=dict(tickformat='.0%', range=[-0.01, 0.70]),
        height=600,
        width=1000,
        legend=dict(
        orientation='h',       # horizontal layout
        yanchor='bottom',
        y=1.02,                # slightly above the chart
        xanchor='center',
        x=0.5,                 # centered horizontally
        title=None             
    )
    )

    return fig

# --------- Navigation buttons
col1c, _, col2c = st.columns([1, 6, 1])
with col1c:
    if st.button("◀️ Back", key="back_c") and st.session_state.slide_3 > 1:
        st.session_state.slide_3 -= 1
with col2c:
    if st.button("Next ▶️", key="next_c") and st.session_state.slide_3 < 3:
        st.session_state.slide_3 += 1

if st.session_state.slide_3 == 1:
    st.markdown("##### 🛫 Origin Airport Performance: How Do NYC Airports Stack Up?")
    st.markdown("""
    **Three busy airports, one big question — who handles delays best?**  
    We looked at delay rates from EWR, JFK, and LGA to compare how often flights take off and land late.
    """)
    
    fig = airport_slide_1(origin_airport_delay)
    st.pyplot(fig)

    st.markdown("""
    **What the data shows:**  
    - **Newark (EWR)** tops the delay chart with 25% departure and 26% arrival delays out of 117k+ flights.  
    - **John F. Kennedy (JFK)** follows with 21% departure and 24% arrival delays over 109k flights.  
    - **LaGuardia (LGA)** is slightly better but still challenged — 19% departure and 23% arrival delays across 101k flights.

    These airports operate in one of the most congested airspaces in the U.S., where weather, volume, and traffic control bottlenecks make delays a persistent issue.

    ✈️ **So what?**  
    - For travelers: Expect delays when flying out of NYC — plan layovers with extra buffer time.  
    - For airports: There’s room to improve ground operations and traffic coordination to ease schedule pressure.
    
    Let’s now explore where flights are headed — and how delays vary across **destination airports**.
    """)

elif st.session_state.slide_3 == 2:
    st.markdown("##### 📍 Destination Airport Delays: Where Do Flights Land Late?")
    st.markdown("""
    **Destination airports tell a mixed story — some struggle, others shine.**  
    We analyzed delays across 104 destination airports to spot patterns in arrival and departure timeliness.
    """)
    
    fig = airport_slide_2(dest_airport_delay)
    st.plotly_chart(fig)

    st.markdown("""
    **What the data shows:**  
    - **High-delay regional airports** like Jackson Hole (50%+ delays, 21 flights) and South Bend (50%+, 10 flights) face major delay challenges.  
    - **Major hubs** such as Atlanta (20% departure, 26% arrival delays) and Chicago O’Hare (23%, 24%) handle heavy traffic with fewer delays.  
    - Some airports stand out for **exceptional punctuality**, like Salt Lake City and Seattle-Tacoma, despite varying flight volumes.

    ✈️ **So what?**  
    - For passengers: Regional airports can be surprisingly delay-prone — check stats before booking connections.  
    - For operators: Volume isn’t the only factor — infrastructure and planning play a huge role in performance.

    With 100+ airports in the mix, patterns aren't always obvious.  
    To dig deeper, we clustered destination airports based on their characteristics — and the results were telling.
    """)  

elif st.session_state.slide_3 == 3:
    st.markdown("##### 🧭 Clustering Destination Airports: Who’s Efficient, and Who’s Struggling?")
    st.markdown("""
    **We grouped 104 destination airports based on departure delays, arrival delays, and flight volumes to reveal distinct operational profiles.**

    """)

    fig = airport_slide_3(dest_airport_delay, cluster_order)
    st.plotly_chart(fig)

    st.markdown("""
    **What the data shows:**  
    - **Cluster 4: Very low delay, low traffic airports** like Anchorage (ANC) and Palm Springs (PSP) operate with minimal delays.  
    - **Cluster 1: Low delay, average traffic airports** such as Chicago O’Hare (ORD) and Fort Lauderdale (FLL) handle moderate traffic with strong efficiency.  
    - **Cluster 3: Low delay, high traffic hubs** including Denver (DEN) and Seattle (SEA) manage heavy volumes with few delays.  
    - **Cluster 0: Moderate delay, low traffic airports** like South Bend (SBN) and Birmingham (BHM) face emerging operational challenges.  
    - **Cluster 2: High delay, low traffic airports** such as Jackson Hole (JAC) and Columbia (CAE) suffer severe delays despite low flight volumes.

    ✈️ **So what?**  
    - The least delayed airports often have low traffic but strong operational smoothness.  
    - Busy hubs with good infrastructure keep delays down despite high volumes.  
    - Smaller airports with moderate or high delays need targeted improvements to reduce disruptions.

    **Next up:** Let’s shift focus to the routes — uncover the problematic origin-dest pairs?
    """)

st.markdown("---")

# ------------ 4. Route Activites in Flight Delays -------------
st.markdown("### 🗺️ Route-Level Delay Insights: Which City Pairs Are Most Impacted?")
st.markdown("""
**Some routes are simply more delay-prone than others.**  
Whether it’s busy corridors between major hubs or long-haul routes affected by upstream issues, certain city pairs rack up delays far more often than others.

Let’s explore which origin-destination combinations are most impacted — and what might be driving the patterns.
""")

# Prepare data
# Create Route column
flights_df['route'] = flights_df['origin'] + ' - ' + flights_df['dest']

# Route-level delay aggregation
route_delay = (
    flights_df.groupby('route').agg({
        'dep_delayed_15': 'mean',     # proportion of flights delayed on departure
        'arr_delayed_15': 'mean',     # proportion of flights delayed on arrival
        'flight': 'count',
        'dep_delay': 'mean',          # average delay in minutes
        'arr_delay': 'mean'
    }).round(2).reset_index()
)

# Rename columns for clarity
route_delay.columns = [
    'Route',
    'Departure Delay ≥15m',
    'Arrival Delay ≥15m',
    'Total Flights',
    'Avg Dep Delay',
    'Avg Arr Delay'
]

# Calculate delay rate (mean of dep + arr delay proportions)
route_delay['Delay Rate'] = ((route_delay['Departure Delay ≥15m'] + route_delay['Arrival Delay ≥15m']) / 2).round(2)

# Compute delay score (delay rate * total avg delay)
route_delay['Delay Score'] = (route_delay['Delay Rate'] * (route_delay['Avg Dep Delay'] + route_delay['Avg Arr Delay'])).round(2)

# Airline-by-route level aggregation
airline_routes = (
    flights_df.groupby(['airline_name', 'route']).agg({
        'dep_delayed_15': 'mean',
        'arr_delayed_15': 'mean',
        'flight': 'count',
        'dep_delay': 'mean',
        'arr_delay': 'mean'
    }).round(2).reset_index()
)

# Rename columns
airline_routes.columns = [
    'Airline',
    'Route',
    'Departure Delay ≥15m',
    'Arrival Delay ≥15m',
    'Total Flights',
    'Avg Dep Delay',
    'Avg Arr Delay'
]

# Calculate delay rate
airline_routes['Delay Rate'] = ((airline_routes['Departure Delay ≥15m'] + airline_routes['Arrival Delay ≥15m']) / 2).round(2)

# Compute delay score
airline_routes['Delay Score'] = (airline_routes['Delay Rate'] * (airline_routes['Avg Dep Delay'] + airline_routes['Avg Arr Delay'])).round(2)


# ------ Bubble Plot
fig = px.scatter(
    route_delay,
    x="Departure Delay ≥15m",
    y="Arrival Delay ≥15m",
    size="Total Flights",
    color="Delay Rate",
    hover_name="Route",
    size_max=60,
    color_continuous_scale='RdBu_r',
    labels={
        'Departure Delay ≥15m': 'Departure Delay Rate',
        'Arrival Delay ≥15m': 'Arrival Delay Rate',
        "Total Flights": "Number of Flights",
    }
)

fig.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br><br>" +
        "Departure Delay: %{x:.0%}<br>" +
        "Arrival Delay: %{y:.0%}<br>" +
        "Number of Flights: %{marker.size:,}<br>" +
        "Delay Rate: %{marker.color:.0%}"
    ),
    marker=dict(opacity=0.7, 
        line=dict(width=1, color='DarkSlateGrey'))
)

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(tickformat='.0%', range=[-0.01, 0.7]),
    yaxis=dict(tickformat='.0%', range=[-0.01, 0.7]),
    coloraxis_colorbar=dict(tickformat=".0%", title="Delay Rate"),
    height=600
)

st.markdown("##### Route Delay Performance: Departure vs Arrival")
st.markdown("*The further to the top-right a bubble is, the more problematic the route. Bigger bubbles = more flights, so they affect more passengers.*")

st.plotly_chart(fig)

# ---- Heatmap
# Split Route into Origin and Destination for heatmap axes
route_delay[['Origin', 'Destination']] = route_delay['Route'].str.split('-', expand=True)

# Create pivot table for heatmap
heatmap_data = route_delay.pivot_table(
    values='Delay Score',
    index='Destination',
    columns='Origin'
)

# Create heatmap
fig = px.imshow(
    heatmap_data,
    color_continuous_scale='RdBu_r',
    aspect='auto',
    labels=dict(color="Delay Score"),
)

fig.update_layout(
    xaxis=dict(side='top'),
    height=1000
)

fig.update_xaxes(tickangle=0)  # Make destination labels horizontal

st.markdown("##### Route-Level Delay Heatmap (Delay Score)")
st.markdown("*Scan across a column to see how one origin airport performs across destinations. Dark red squares mean consistent delays — the worst routes by delay behavior jump right out at you.*")

st.plotly_chart(fig)

# ------ Airline Comparison on Selected Route ------
st.markdown("##### Compare Airlines on Same Route")
st.markdown("Even on the same route, your experience may vary widely depending on the airline.")

route = st.selectbox("Choose a route:", flights_df["route"].unique())

# Filter for selected route
filtered = flights_df[flights_df['route'] == route]

# Compute average delays per airline for this route
airline_delay = (
    filtered.groupby('airline_name').agg({
        'dep_delayed_15': 'mean',
        'arr_delayed_15': 'mean',
        'flight': 'count'
        })
    .round(2)
    .reset_index()
)

# Rename columns for cleaner legend labels
airline_delay = airline_delay.rename(columns={
    'dep_delayed_15': 'Departure Delay ≥15m',
    'arr_delayed_15': 'Arrival Delay ≥15m',
     'flight': 'Total Flights'
})

airline_delay = airline_delay.sort_values('Total Flights', ascending=False)


# Calculate overall average delay for the route
overall_avg_delay = flights_df[['dep_delayed_15', 'arr_delayed_15']].values.mean()

# Create grouped bar chart
fig = px.bar(
    airline_delay,
    y='airline_name',
    x=['Departure Delay ≥15m', 'Arrival Delay ≥15m'],
    orientation='h',
    barmode='group',
    title=f"Delays by Airline for Route {route}",
    color_discrete_map={
        'Departure Delay ≥15m': custom_palette[0], 
        'Arrival Delay ≥15m': custom_palette[1]   
    }
)

# Add vertical line
fig.add_vline(
    x=overall_avg_delay,
    line_dash="dash",
    line_color="gray",
    annotation_text="Overall Avg Delay ≥15m",
    annotation_position="top",
    annotation_font_size=12,
    opacity=0.7
)

# Format layout
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    yaxis_title="Airline",
    xaxis_title="Proportion of Delayed Flights",
    xaxis=dict(tickformat=".0%", range=[0, 1]),
    legend_title_text=None,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=0.3
    ),
    height=500
)

# Add custom hovertemplate including Total Flights
fig.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>Total Flights: %{customdata[0]}<br>Delay Rate: %{x:.0%}<extra></extra>"
    ),
    customdata=airline_delay[['Total Flights']].values
)

st.plotly_chart(fig)