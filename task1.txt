import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(r"C:\Users\HP\Documents\boston.csv")

print("Dataset Loaded Successfully")
print(df.head())


X = df[["RM", "LSTAT", "PTRATIO", "TAX"]]
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Train Linear Regression Model
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 5. Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 6. Evaluation Metrics
# ===============================
print("\nModel Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# ===============================
# 7. Predict New House Price
# ===============================
new_house = [[6.5, 15.0, 18.0, 300]]
predicted_price = model.predict(new_house)

print("\nPredicted House Price (MEDV):", predicted_price[0])

# ===============================
# 8. GRAPH 1: Actual vs Predicted
# ===============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices (MEDV)")
plt.ylabel("Predicted House Prices (MEDV)")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ===============================
# 9. GRAPH 2: Simple Regression Visualization (RM vs MEDV)
# ===============================
plt.figure()
plt.scatter(df["RM"], df["MEDV"])
plt.plot(df["RM"], model.predict(df[["RM", "LSTAT", "PTRATIO", "TAX"]]), linewidth=2)
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price (MEDV)")
plt.title("House Price vs Number of Rooms")
plt.show()
