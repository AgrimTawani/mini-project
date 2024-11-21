import pandas as pd
import matplotlib.pyplot as plt     
from sklearn.model_selection import train_test_split
diamond_data = pd.read_csv('Diamond.csv')
diamond_data.describe()
diamond_data.info()
print(diamond_data['cut'].unique())
print(diamond_data['color'].unique())
print(diamond_data['clarity'].unique())

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded_columns = encoder.fit_transform(diamond_data[['cut', 'color', 'clarity']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['cut', 'color', 'clarity']))


diamond_data = diamond_data.drop(['cut', 'color', 'clarity'], axis=1)

diamond_data = pd.concat([diamond_data, encoded_df], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit and transform the price column
diamond_data['price_scaled'] = scaler.fit_transform(diamond_data[['price']])
diamond_data = diamond_data.drop(columns=['price'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = diamond_data.drop(columns=['price_scaled'])
y = diamond_data['price_scaled']

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")