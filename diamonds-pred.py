# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load the dataset
    diamond_data = pd.read_csv('Diamond.csv')

    # Exploratory Data Analysis (EDA)
    print("Dataset Description:")
    print(diamond_data.describe())
    print("\nDataset Information:")
    diamond_data.info()

    print("\nUnique values in 'cut':", diamond_data['cut'].unique())
    print("Unique values in 'color':", diamond_data['color'].unique())
    print("Unique values in 'clarity':", diamond_data['clarity'].unique())

    # Prepare figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diamond Data Visualizations", fontsize=16)

    # Visualize the distribution of 'price'
    sns.histplot(diamond_data['price'], kde=True, bins=30, color='blue', ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Diamond Prices")
    axes[0, 0].set_xlabel("Price")
    axes[0, 0].set_ylabel("Frequency")

    # Visualize the distribution of 'cut' categories
    sns.countplot(x='cut', data=diamond_data, palette='viridis', ax=axes[0, 1])
    axes[0, 1].set_title("Count of Diamond Cuts")
    axes[0, 1].set_xlabel("Cut")
    axes[0, 1].set_ylabel("Count")

    # Data Preprocessing
    encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = encoder.fit_transform(diamond_data[['cut', 'color', 'clarity']])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['cut', 'color', 'clarity']))

    # Drop original categorical columns and add encoded ones
    diamond_data = diamond_data.drop(['cut', 'color', 'clarity'], axis=1)
    diamond_data = pd.concat([diamond_data, encoded_df], axis=1)

    # Scale the 'price' column
    scaler = StandardScaler()
    diamond_data['price_scaled'] = scaler.fit_transform(diamond_data[['price']])
    diamond_data = diamond_data.drop(columns=['price'])

    # Train-Test Split
    X = diamond_data.drop(columns=['price_scaled'])
    y = diamond_data['price_scaled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate Model Performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Visualization of Predictions
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6, color='purple')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[1, 0].set_title("Actual vs Predicted Values")
    axes[1, 0].set_xlabel("Actual Prices (scaled)")
    axes[1, 0].set_ylabel("Predicted Prices (scaled)")

    # Visualize feature importance
    coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    coefficients.plot(kind='bar', color='orange', ax=axes[1, 1])
    axes[1, 1].set_title("Feature Importance (Linear Regression Coefficients)")
    axes[1, 1].set_ylabel("Coefficient Value")
    axes[1, 1].set_xlabel("Feature")

    # Adjust layout and display all graphs in one window
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
