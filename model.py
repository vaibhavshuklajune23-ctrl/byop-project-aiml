import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[["study_hours"]]
y = data["marks"]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])

print("Predicted Marks:", predicted_marks[0])
