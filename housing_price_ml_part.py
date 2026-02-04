#1, import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2. load dataset
df = pd.read_csv(r"C:\Users\91808\OneDrive\Desktop\ML\housing_eda_ready.csv")

#3. select the feature and target
x = df[['Area_sqft']]
y = df[['Price_Lakhs']]

#4. train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#5. Build Simple Linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#6. predictions
y_pred = model.predict(x_test)

#7. Model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("RMSE:",rmse)
print("R2 score:",r2)

#8. Visualize regression line
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple linear regression")
plt.show()

#9. Multiple linear regression

#feature selection
X = df[['Area_sqft','Bedrooms','Bathrooms']]
Y = df['Price_Lakhs']
#train model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train,Y_train)
Y_pred_multi = model_multi.predict(X_test)
#evaluation multiple regression
print("R2 score:",r2_score(Y_test,Y_pred_multi))
print("RMSE:",np.sqrt(mean_squared_error(Y_test,Y_pred_multi)))
#Interpret coefficients
coeff_df = pd.DataFrame(
    model_multi.coef_,
    X.columns,
    columns=['Coefficient']
)
print(coeff_df)