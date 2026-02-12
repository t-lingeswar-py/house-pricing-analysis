import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
data = pd.read_csv("data.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Assume columns are Area and Price (adjust if needed)
X = data.iloc[:, :-1]   # feature
y = data.iloc[:, -1]    # target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
print("\nMean Squared Error:", mse)

# Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, predictions, color="red", linewidth=2, label="Predicted Line")
plt.xlabel("Feature")
plt.ylabel("House Price")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()
