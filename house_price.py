#1. required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. Load dataset
df = pd.read_csv(r"C:\Users\91808\OneDrive\Desktop\ML\Dataset\housing_prices.csv")

#3. Basic data inspection
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
df.info()

#4. Statistical summary 
print(df.describe())

#5. check missing values
df.isnull().sum()

#6. Univariate Analysis
#6.1 Histogram(Distribution)
plt.figure()
plt.hist(df['Price_Lakhs'],bins=15)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#7. Bivariate Analysis
#7.1 Acatter plot(Area vs Price)
plt.get_fignums()
plt.scatter(df['Area_sqft'],df['Price_Lakhs'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs price')
plt.show()

#7.2 Bar plot(Location vs Avg Price)
df.groupby('Location')['Price_Lakhs'].mean().plot(kind='bar')
plt.xlabel('Location')
plt.ylabel('Average Price')
plt.title('Average price by location')
plt.show()

#7.3 Bedroom vs price
df.groupby('Bedrooms')['Price_Lakhs'].mean().plot(kind='bar')
plt.xlabel('Bedrooms')
plt.ylabel('Average Price')
plt.title('Average price by Number of bedrooms')
plt.show()

#8. Multi variatenAnalysis
#8.1 Correlation matrix
corr = df.corr(numeric_only=True)

#8.2 heatmap
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True)
plt.title('Correlation Heatmap')
plt.show()

#9 Save Dataset for modeling (optional)
df.to_csv('housing_eda_ready.csv', index=False)


#Feature level conclusion
#1. Most important feature: Area
#2. Moderately important : Bedrooms, Bathrooms
#3. Highly influential categorical featue : Location