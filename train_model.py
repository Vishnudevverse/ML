import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('used_cars.csv')

features = ['model_year', 'milage', 'fuel_type', 'transmission', 'brand', 'accident']
target = 'price'
df_selected = df[features + [target]].copy()

df_selected['price'] = df_selected['price'].str.replace('$', '').str.replace(',', '')
df_selected['price'] = pd.to_numeric(df_selected['price'], errors='coerce')

df_selected['milage'] = df_selected['milage'].str.replace(' mi.', '').str.replace(',', '')
df_selected['milage'] = pd.to_numeric(df_selected['milage'], errors='coerce')

conversion_rate = 83 
df_selected['price'] = df_selected['price'] * conversion_rate

df_cleaned = df_selected.dropna()

print(f"Original rows: {len(df)}, Rows after cleaning: {len(df_cleaned)}")

df_encoded = pd.get_dummies(df_cleaned, columns=['fuel_type', 'transmission', 'brand', 'accident'], drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

model_columns = X.columns
joblib.dump(model_columns, 'model_columns.joblib')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining the model... This might take a moment.")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")

joblib.dump(model, 'car_price_model.joblib')
print("\n✅ Model and columns saved successfully!")