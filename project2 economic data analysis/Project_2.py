#PROJECT 2: ECONOMIC DATA ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-v0_8')

df = pd.read_csv("D:\projects_data\salesforcourse-4fe2kehu.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.drop('column1', axis=1, inplace=True)
print(df.dtypes)
# print(df.head())
print('Missing values\n',df.isnull().sum())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['customer_age'] = df['customer_age'].astype(int)
df['year'] = df['year'].astype(int)
df['quantity'] = df['quantity'].astype(int)

df['date']= pd.to_datetime(df['date'])
#Extract the year and month from the "Date" column
df['year']=df['date'].dt.year
df['year_month'] = df['date'].dt.strftime('%Y-%m')

# Profit
df['profit'] = df['revenue'] - df['cost']
# Profit Margin
df['profit_margin'] = (df['profit'] / df['revenue'] * 100).round(2)

# Create Age Groups
def age_group(age):
    if age < 26:
        return '18-25'
    elif age < 36:
        return '26-35'
    elif age < 46:
        return '36-45'
    elif age < 56:
        return '46-55'
    elif age < 66:
        return '56-66'
    else:
        return '65+'

df['age_group'] = df['customer_age'].apply(age_group)

# Key metrics
total_revenue = df['revenue'].sum()
total_cost = df['cost'].sum()
total_profit = df['profit'].sum()
profit_margin_overall = (total_profit / total_revenue * 100)
num_transactions = len(df)
avg_order_value = df['revenue'].mean()
total_quantity = df['quantity'].sum()

print(f"\n FINANCIAL METRICS:")
print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Total Cost: {total_cost:,.2f}")
print(f"Total Profit: {total_profit:,.2f}")
print(f"Profit Margin: {profit_margin_overall:.2f}%")

print(f"\n TRANSACTION METRICS:")
print(f"Number of Transactions: {num_transactions:,}")
print(f"Average Order Value: {avg_order_value:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")

# Revenue by Product Category
category_revenue = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
print("\nRevenue by Product Category:\n", category_revenue)

# Revenue by Sub Category (top 10)
subcat_revenue = df.groupby('sub_category')['revenue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Sub Categories by Revenue:\n", subcat_revenue)

# Best selling by quantity
quantity_by_product = df.groupby('product_category')['quantity'].sum().sort_values(ascending=False)
print("\nQuantity Sold by Product Category:\n", quantity_by_product)

# Most profitable products
profit_by_category = df.groupby('product_category')['profit'].sum().sort_values(ascending=False)
print("\nProfit by Product Category:\n", profit_by_category)

# Product with best profit margin
avg_margin_by_category = df.groupby('product_category')['profit_margin'].mean().sort_values(ascending=False)
print("\nAverage Profit Margin by Category:\n",avg_margin_by_category.round(2))


# LOCATION ANALYSIS

# Revenue by Country
country_revenue = df.groupby('country')['revenue'].sum().sort_values(ascending=False)
print("\nRevenue by Country:\n", country_revenue)

# Revenue by State (top 10)
state_revenue = df.groupby('state')['revenue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 States by Revenue:\n", state_revenue)

# Profit by Country
country_profit = df.groupby('country')['profit'].sum().sort_values(ascending=False)
print("\nProfit by Country:\n", country_profit)

# Top 5 locations (state) by profit
top_states_profit = df.groupby('state')['profit'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 States by Profit:\n", top_states_profit)

# CUSTOMER ANALYSIS

# Revenue by Gender
gender_revenue = df.groupby('customer_gender')['revenue'].sum().sort_values(ascending=False)
print("\nRevenue by Gender:\n",gender_revenue)

# Average order value by Gender
gender_avg = df.groupby('customer_gender')['revenue'].mean().sort_values(ascending=False)
print("\nAverage Order Value by Gender:\n", gender_avg.round(2))

# Revenue by Age Group
age_revenue = df.groupby('age_group')['revenue'].sum().sort_values(ascending=False)
print("\nRevenue by Age Group:\n",age_revenue)

# Customer count by demographics
gender_count = df['customer_gender'].value_counts()
age_count = df['age_group'].value_counts()
print("\nCustomer Count by Gender:\n", gender_count)
print("\nCustomer Count by Age Group:\n", age_count)


# Age distribution
plt.figure()
plt.hist(df["customer_age"], bins=30)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Group data by month

Month_Revenue = df.groupby(['year_month'])['revenue'].sum().reset_index()
sns.relplot(data=Month_Revenue, x="year_month", y="revenue",kind="line", height =10, aspect = 2.1)
plt.tight_layout()
plt.show()

monthly_cost =df.groupby(['year_month'])['cost'].sum().reset_index()
sns.relplot(data=monthly_cost, x="year_month", y="cost", kind="line", height =10, aspect = 2.1)
plt.tight_layout()
plt.show()

monthly_data = (df.groupby("year_month")[["revenue", "cost", "profit"]].sum().reset_index())
print('monthly data\n', monthly_data)
plt.figure(figsize=(9,6))
plt.plot(monthly_data["year_month"], monthly_data["revenue"], label="Revenue")
plt.plot(monthly_data["year_month"], monthly_data["cost"], label="Cost")
plt.plot(monthly_data["year_month"], monthly_data["profit"], label="Profit")
plt.title("Monthly Revenue, Cost & Profit Trend")
plt.xlabel("Year-Month")
plt.ylabel("Amount")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Group by sub category and calculate total quantity sold
category_sales = df.groupby(['product_category','sub_category'])['quantity'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=category_sales,x="sub_category",y="quantity",hue="product_category")
plt.title("Top Selling Sub-Category")
plt.xlabel("Sub Category")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.legend(title="Category")
plt.tight_layout()
plt.show()

subcategory_profit= df.groupby(['sub_category'])['profit'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=subcategory_profit,x="sub_category",y="profit")
plt.title("Profit per Sub-Category")
plt.xlabel("Sub Category")
plt.ylabel("Profit")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

subcategory_margin = df.groupby('sub_category')['profit_margin'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=subcategory_margin,x="sub_category",y="profit_margin")
plt.title("Profit margin per Sub-Category")
plt.xlabel("Sub Category")
plt.ylabel("profit margin")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Top selling product for each Customer Age by Product Category

df_grouped = (df.groupby(["customer_age", "product_category"])["quantity"].sum().reset_index())

top_products = (df_grouped.loc[df_grouped.groupby("customer_age")["quantity"].idxmax()])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_products,x="customer_age",y="quantity",hue="product_category")
plt.title("Top Selling Category Products by Customer Age")
plt.xlabel("Customer Age")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.legend(title="Category")
plt.tight_layout()
plt.show()


# Top selling product for each Customer Age by Sub category
df_subgrouped = (df.groupby(["customer_age", "sub_category"])["quantity"].sum().reset_index())

top_products_sub = (df_subgrouped.loc[df_subgrouped.groupby("customer_age")["quantity"].idxmax()])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_products_sub,x="customer_age",y="quantity",hue="sub_category")
plt.title("Top Selling Sub-Category Products by Customer Age")
plt.xlabel("Customer Age")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.legend(title="Sub Category")
plt.tight_layout()
plt.show()

# country having the highest profit
country_sales =df.groupby('country')['profit'].sum().reset_index()

plt.figure(figsize=(8, 8))
plt.pie(country_sales['profit'],labels=country_sales['country'],autopct='%1.1f%%',startangle=90,  colors=plt.cm.Set3.colors)
plt.title('Profit Distribution by Country')
plt.tight_layout()
plt.show()

# products purchased the most in each country 
df_country = df.groupby(["country", "sub_category"])["quantity"].sum().reset_index()

top_products_country = (df_country.loc[df_country.groupby("country")["quantity"].idxmax()])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_products_country,x="country",y="quantity",hue="sub_category")
plt.title("Top Selling Sub-Category Products by Country")
plt.xlabel("Country")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.legend(title="Sub Category")
plt.tight_layout()
plt.show()
