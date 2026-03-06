import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.figure(figsize=(10,6))
sns.set(style='whitegrid')

# %%
df = pd.read_csv('demand_forecasting.csv')
df.head(5)

# %%
df.dtypes

# %%
df['Date'] = pd.to_datetime(df['Date'])

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %%
df.describe().T

# %%
df.describe(include='object').T

# %%
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()
df

# %%
df.columns

# %%
df['Discounted_Price'] = df['Price']*(1-df['Discount']/100)
df['Sell through rate'] = df['Units Sold']/df['Inventory Level']
df

# %%
df.groupby("Category")["Demand"].agg(['mean','sum','std']).sort_values(by='sum',ascending=False).reset_index()

# %%
df.groupby(['Region','Seasonality'])['Demand'].mean().T

# %%
df.groupby("Promotion")["Demand"].mean()

# %%
pd.pivot_table(df,values='Demand',index='Month',columns='Category',aggfunc='mean')

# %%
sns.histplot(df['Demand'],bins=20,kde=True)
plt.title('Demand Distribution')
plt.show()

# %%
sns.scatterplot(data=df, x='Inventory Level',y='Units Sold')
plt.title('Inventory vs Units Sold')
plt.show()

# %%
sns.boxplot(data=df,x='Category',y='Demand')
plt.xticks(rotation=45)
plt.title('Demand by Category')
plt.show()

# %%
sns.boxplot(data=df,x='Weather Condition',y='Demand')
plt.xticks(rotation=45)
plt.title('Demand by Weather Condition')
plt.show()

# %%
monthly_demand = df.groupby('Month')['Demand'].mean()
monthly_demand.plot(kind='bar')
plt.title('Average Demand by Month')
plt.show()

# %%
daily_demand = df.groupby('Date')['Demand'].sum()
daily_demand.plot()
plt.title('Total Daily Demand over Time ')
plt.show()

# %%
sns.barplot(data=df,x='Promotion',y='Demand')
plt.title('Promotion Impact on Demand')
plt.show()

# %%
sns.scatterplot(data=df,x='Discounted_Price',y='Demand')
plt.title('Discounted Price vs Demand')
plt.show()

# %%
df.groupby('Seasonality')['Demand'].mean().plot(kind='bar',title='Demand by Season')
plt.show()

# %%
df.groupby('Epidemic')['Demand'].mean().plot(kind='bar',title='Epidemic Impact on Demand')
plt.show()

# %%
df.to_csv('preprocessed_demand_forecasting.csv',index=False)