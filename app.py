import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Water Quality Prediction",page_icon=":droplet:")

st.markdown(
    """
    <style>    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #e6f2ff; /* Light blue background */
        color: #003366;  /* Dark blue text */
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #003366 !important;  /* Apply dark text to all headings and labels */
    }

    .stButton>button {
        text-align: center;
        background-color: #4da6ff;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: 1px solid #004d99;
    }

    .stButton>button:hover {
        background-color: #1a8cff;
        border: 1px solid #004d99;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


# Load the dataset
df = pd.read_csv("cleaned-data.csv")

#Load the model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

#pollutants and thresholds
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
safety_thresholds = {
    'O2': 5.0,
    'NO3': 10,
    'NO2': 0.1,
    'SO4': 250,
    'PO4': 0.1,
    'CL': 250
}

def classify_water(row, thresholds):
    for pollutant, threshold in thresholds.items():
        value = row[pollutant]
        if pollutant == 'O2':
            if value < threshold:
                return 'Not Safe'
        elif value > threshold:
            return 'Not Safe'
    return 'Safe'

st.markdown(
    """
    <h1 style='text-align: center;'>Water Quality Prediction App</h1>
    <h5 style='text-align: center;'>Predict pollutant levels for a specific station and year.</h5>
    """,
    unsafe_allow_html=True
)

st.write("")
with st.expander("Show Safety Thresholds"):
        st.table(pd.DataFrame(safety_thresholds, index=["Threshold"]).T)

with st.expander("How is Water Quality Determined?"):
    st.markdown("""
    - **O₂** must be **above 5.0** for safe water.
    - All other pollutants must be **below their thresholds**.
    - The model predicts pollutant levels based on historical trends at each station.
    """)
st.write("")

#user inputs
valid_station_ids = sorted(df['id'].unique())
station_id = st.selectbox("Select Station ID:", valid_station_ids)
year = st.number_input("Enter Year:", min_value=2000, max_value=2030, placeholder="Enter the year")
st.write("")
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    predict_btn = st.button("Predict")

if predict_btn:
    with st.spinner("Predicting pollutant levels..."):
        input_data = pd.DataFrame({"year": [year], "id": [station_id]})
        input_encoded=pd.get_dummies(input_data, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        #predict
        predicted_pollutants = model.predict(input_encoded)[0]
        predicted_dict = {p: val for p, val in zip(pollutants, predicted_pollutants)}
        safety_status = classify_water(predicted_dict, safety_thresholds)

        st.write(f"#### Predicted pollutant levels for station '{station_id}' in {year}:")
        st.dataframe(pd.DataFrame(predicted_dict, index=["Predicted Value"]).T)
        if safety_status == "Safe":
            st.success("🟢 Water Quality Status: Safe — within environmental safety limits.")
        else:
            st.error("🔴 Water Quality Status: Not Safe — pollutant levels exceed thresholds.")
        
        comparison_df = pd.DataFrame({
        "Pollutant": pollutants,
        "Predicted": [predicted_dict[p] for p in pollutants],
        "Threshold": [safety_thresholds[p] for p in pollutants],
        "Status": [
            "Safe" if (p == 'O2' and predicted_dict[p] >= safety_thresholds[p]) or 
                    (p != 'O2' and predicted_dict[p] <= safety_thresholds[p]) 
            else "Unsafe"
            for p in pollutants
            ]
        })
        st.write("")
        st.markdown("##### Pollutant Risk Assessment")
        st.dataframe(comparison_df, use_container_width=True)


        st.write("")
        # Plot Bar Chart
        st.markdown("##### Predicted Pollutant Levels")
        pred_df = pd.DataFrame({
            'Pollutant': pollutants,
            'Predicted Value': predicted_pollutants
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=pred_df, x='Pollutant', y='Predicted Value', hue='Pollutant', palette='coolwarm', legend=False, ax=ax)

        # Annotate unsafe pollutants
        for i, pollutant in enumerate(pred_df['Pollutant']):
            val = predicted_dict[pollutant]
            threshold = safety_thresholds[pollutant]
            if (pollutant == 'O2' and val < threshold):
                ax.text(i, threshold + 0.5, f"Limit >  {threshold}", ha='center', color='red', fontsize=8)
            elif (pollutant != 'O2' and val > threshold):
                ax.text(i, threshold + 0.5, f"Limit < {threshold}", ha='center', color='red', fontsize=8)

        ax.set_title("Predicted Pollutants vs Safety Thresholds")
        ax.set_ylabel("Concentration")
        ax.set_xlabel("Pollutants")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("")
        st.markdown("#####  Water Quality Trend Analysis")

        station_df = df[df['id'] == int(station_id)]

        yearly_safety = station_df.groupby(['year'])['Water Quality'].value_counts().unstack().fillna(0)
        safe = yearly_safety.get('Safe', pd.Series(0, index=yearly_safety.index))
        not_safe = yearly_safety.get('Not Safe', pd.Series(0, index=yearly_safety.index))
        yearly_safety['% Safe'] = safe / (safe + not_safe) * 100

        st.markdown(f"#####  Year-wise % of Safe Samples for Station {station_id}")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly_safety, x=yearly_safety.index, y='% Safe', marker='o', ax=ax1)
        ax1.set_title(f"Year-wise % of Safe Samples - Station {station_id}")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Percent Safe")
        ax1.grid(True)
        st.pyplot(fig1)

        monthly_safety = station_df.groupby('month')['Water Quality'].value_counts().unstack().fillna(0)
        safe = monthly_safety.get('Safe', pd.Series(0, index=monthly_safety.index))
        not_safe = monthly_safety.get('Not Safe', pd.Series(0, index=monthly_safety.index))
        monthly_safety['% Safe'] = safe / (safe + not_safe) * 100
        st.write("")
        st.markdown(f"#####  Month-wise (Seasonal) % of Safe Samples for Station {station_id}")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=monthly_safety, x=monthly_safety.index, y='% Safe', marker='o', ax=ax2)
        ax2.set_title(f"Month-wise % of Safe Samples - Station {station_id}")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Percent Safe")
        ax2.set_xticks(range(1, 13))
        ax2.grid(True)
        st.pyplot(fig2)

        st.write("")
        with st.expander("🔍 Show Raw Yearly Safety Data (Station-wise)"):
            st.dataframe(yearly_safety.reset_index())

        with st.expander("🔍 Show Raw Monthly Safety Data (Station-wise)"):
            st.dataframe(monthly_safety.reset_index())

        
 




