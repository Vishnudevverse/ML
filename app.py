import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

if 'comparison_list' not in st.session_state:
    st.session_state.comparison_list = []
if 'prediction_state' not in st.session_state:
    st.session_state.prediction_state = {}

@st.cache_data
def load_assets():
    """Loads the ML model, column data, and the cleaned dataset with currency conversion."""
    model = joblib.load('car_price_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    
    df = pd.read_csv('used_cars.csv')


    df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')


    conversion_rate = 83 
    df['price'] = df['price'] * conversion_rate
    

    df_for_options = df.dropna()
    
    return model, model_columns, df_for_options

model, model_columns, df_for_options = load_assets()
features = ['model_year', 'milage', 'fuel_type', 'transmission', 'brand', 'accident']


st.sidebar.header('Enter Car Details')
brand = st.sidebar.selectbox('Brand', sorted(df_for_options['brand'].unique()))
model_year = st.sidebar.slider('Model Year', 2000, 2024, 2018)
milage = st.sidebar.slider('Milage (km)', 0, 300000, 50000)
fuel_type = st.sidebar.selectbox('Fuel Type', df_for_options['fuel_type'].unique())
transmission = st.sidebar.selectbox('Transmission', df_for_options['transmission'].unique())
accident = st.sidebar.selectbox('Accident History', df_for_options['accident'].unique())

st.sidebar.info("💡 **Tip:** Adjust the sliders and dropdowns to see how they affect the car's price.")

st.title('🚗 Advanced Car Price Predictor')

if st.sidebar.button('Predict Price', type="primary"):
    input_data = {
        'model_year': [model_year], 'milage': [milage], 'fuel_type': [fuel_type],
        'transmission': [transmission], 'brand': [brand], 'accident': [accident]
    }
    input_df = pd.DataFrame(input_data)
    input_df_encoded = pd.get_dummies(input_df)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df_aligned)[0]
    

    st.session_state.prediction_state = {
        "prediction": prediction,
        "inputs": {'brand': brand, 'model_year': model_year, 'milage': milage, 
                   'fuel_type': fuel_type, 'transmission': transmission, 'accident': accident}
    }

if st.session_state.prediction_state:

    pred_data = st.session_state.prediction_state
    prediction = pred_data["prediction"]
    inputs = pred_data["inputs"]

    st.header("Prediction Results")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Predicted Price", value=f"₹ {prediction:,.0f}")
    
        if st.button("Add to Comparison"):
            car_details = {
                'Brand': inputs['brand'], 'Year': inputs['model_year'], 'Milage': inputs['milage'],
                'Fuel': inputs['fuel_type'], 'Transmission': inputs['transmission'], 
                'Predicted Price': f"₹ {prediction:,.0f}"
            }
            st.session_state.comparison_list.append(car_details)
            st.success("Car added to comparison list below!")

    with col2:
        feature_imp = pd.Series(model.feature_importances_, index=model_columns).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(); sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette='viridis')
        ax.set_title("Top Features Influencing Price"); ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    
    st.markdown("---")
    

    st.subheader("Market Analysis")
    market_cars = df_for_options[(df_for_options['brand'] == inputs['brand']) & (df_for_options['model_year'] == inputs['model_year'])]
    if not market_cars.empty:
        fig, ax = plt.subplots(figsize=(10, 4)); sns.histplot(market_cars['price'], kde=True, ax=ax, color='skyblue', bins=20)
        ax.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Your Predicted Price: ₹{prediction:,.0f}')
        ax.set_title(f"Price Distribution for {inputs['brand']} ({inputs['model_year']})"); ax.set_xlabel("Price (₹)"); ax.set_ylabel("Number of Cars")
        ax.legend(); st.pyplot(fig)
    
    st.markdown("---")


    st.subheader("Similar Real Listings")
    price_tolerance = 0.15
    similar_cars = df_for_options[
        (df_for_options['brand'] == inputs['brand']) &
        (df_for_options['model_year'] == inputs['model_year']) &
        (df_for_options['price'].between(prediction * (1 - price_tolerance), prediction * (1 + price_tolerance)))
    ].copy()
    if not similar_cars.empty:
        similar_cars_encoded = pd.get_dummies(similar_cars[features])
        similar_cars_aligned = similar_cars_encoded.reindex(columns=model_columns, fill_value=0)
        similar_cars['predicted_price'] = model.predict(similar_cars_aligned)
        
        deal_threshold = 0.90
        similar_cars['deal_status'] = similar_cars.apply(
            lambda row: "🔥 Good Deal!" if row['price'] < row['predicted_price'] * deal_threshold else "-",
            axis=1
        )
        st.dataframe(similar_cars[['brand', 'model', 'model_year', 'milage', 'price', 'deal_status']])
    else:
        st.warning("No similar cars found in the dataset for this price range.")

else:
    st.info("Enter car details in the sidebar and click 'Predict Price' to see the results.")

st.markdown("---")
st.header("Comparison List")
if st.session_state.comparison_list:
    comparison_df = pd.DataFrame(st.session_state.comparison_list)
    st.table(comparison_df)
    if st.button("Clear Comparison List"):
        st.session_state.comparison_list = []
        st.session_state.prediction_state = {}
        st.rerun()
else:
    st.info("You haven't added any cars to the comparison list yet.")