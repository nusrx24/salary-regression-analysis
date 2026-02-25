import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

csv_path = r"C:\Users\Nusair\Downloads\Salary_Data.csv"
ds = pd.read_csv(csv_path)

X = ds[["YearsExperience"]].values
y = ds["Salary"].values

model = LinearRegression()
model.fit(X, y)

new_years = np.array([[1.5], [3.0], [5.0], [7.0], [10.0]])
predicted_salaries = model.predict(new_years)

print("Linear Regression Model Trained")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}\n")

print("Predicted Salary for New Years of Experience:")
for years, salary in zip(new_years.flatten(), predicted_salaries):
	print(f"{years:.1f} years -> ${salary:,.2f}")

plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.scatter(new_years, predicted_salaries, color="green", marker="x", s=100, label="Predictions")
plt.title("Salary vs Experience (Linear Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()