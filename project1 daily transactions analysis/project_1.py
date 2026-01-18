# PROJECT 1: DAILY TRANSACTIONS ANALYSIS

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
# plt.style.use('fivethirtyeight')

#######u have the data set na uski link neeche wale me paste kardo
#similarly plt.save fig me bhi address lagega 

df = pd.read_csv("D:\Daily Household Transactions.csv")
df.dropna()


#data filtering

print(df.describe().round(2))
print("\nMissing values:")
print(df.isnull().sum())

df['Category'].fillna('Unknown', inplace=True)
df.dropna(subset=['Date','Amount'],  inplace=True)

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Removed {duplicates} duplicates")


df['Amount'] = df['Amount'].astype(float)


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
null_dates = df['Date'].isnull().sum()
print('number of null date',null_dates)
df = df.dropna(subset=['Date'])
df['Day']=df['Date'].dt.day
df['Day_Name']=df['Date'].dt.day_name()
df['Month']=df['Date'].dt.month
# df['Month_Name']= df['Date'].dt.month_name()
df['Month_Name'] = pd.to_datetime(df['Month'], format='%m').dt.month_name()
print(df.head())
print(df.dtypes)

#Data analysis

total_expense = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
print("Total Expense-" ,total_expense)

total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
print("Total Income- ", total_income)

#C1- Number of transaction
sns.countplot(data = df, x = "Income/Expense")
plt.title('Number of transaction')
plt.show()
plt.savefig('01_Number_of_transaction.png')

#C2- 
sns.boxplot(data = df, x = "Amount", y = "Income/Expense")
plt.show()
plt.savefig('02_category_spending_bar.png')

total_saving= total_income-total_expense
print('Total savings-', total_saving)
num_transactions = len(df)
print('Number of transactions- ', num_transactions)

highest = df['Amount'].max()
lowest = df['Amount'].min()
average= df['Amount'].mean().round(2)

print('Highest Transaction-', highest)
print('Lowest Transaction-', lowest)
print('Average transaction-', average)

#Category Analysis

catergory_wise= df[df['Income/Expense'] == 'Expense'].groupby(['Category'])['Amount'].agg([
                ('Total','sum'),('Average', 'mean'),('Count', 'count'),('Max', 'max'),('Min', 'min')
]).round(2)

catergory_wise['Percentage'] = (catergory_wise['Total'] / total_expense * 100).round(2)
catergory_wise = catergory_wise.sort_values('Total', ascending=False)
print('Category wise Summary\n', catergory_wise)

#C3-Number of Transactions per Payment Mode
sns.countplot(data = df, x = "Mode", order = df["Mode"].value_counts().iloc[:3].index)
plt.title('Number of Transactions per Payment Mode')
plt.xlabel('Mode of Transaction')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

#C4- Number of Transactions per Category
sns.countplot(data = df, x = "Category", order = df["Category"].value_counts().iloc[:5].index)
plt.title('Number of Transactions per Category')
plt.xlabel('Category of Transaction')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

#C5- Number of Transactions per SubCategory
sns.countplot(data = df, x = "Subcategory", order = df["Subcategory"].value_counts().iloc[:10].index)
plt.xticks(rotation= 90)
plt.title('Number of Transactions per Subcategory')
plt.xlabel('Subcategory of Transaction')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

category_totals= catergory_wise['Total'].sort_values(ascending= False)

#C6- Total spending by category
# plt.figure(figsize=(12, 6))
plt.bar(category_totals.index, category_totals.values, color='steelblue')
plt.title('Total Spending by Category')
plt.xlabel('Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#C7- Spending Distribution by Category(PIE)
plt.figure(figsize=(8,8))
category_totals_updated= category_totals.sort_values(ascending=False).head(10)
plt.pie(category_totals_updated.values, labels=category_totals_updated.index, 
        autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
plt.title('Spending Distribution by Category', fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

#C8- Spending Distribution by Category(BOX) 
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['Income/Expense'] == 'Expense'], y='Category', x='Amount', order = df["Category"].value_counts().iloc[:5].index, palette='Set2')
plt.title('Amount Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Amount')
# plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#C9-Spending Distribution by SubCategory
plt.figure(figsize = (12,6))
sns.boxplot(data = df, x = "Amount", y = "Subcategory", order =df["Subcategory"].value_counts().iloc[:10].index)
plt.title('Amount Distribution by SubCategory')
plt.xlabel('SubCategory')
plt.ylabel('Amount')
# plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Monthly Analysis

monthly_expenses= (df[df['Income/Expense'] == 'Expense'].groupby(['Month_Name'])['Amount'].sum().reindex(['January','February','March','April','May','June',
           'July','August','September','October','November','December']).reset_index())
print('Monthly Expenses \n', monthly_expenses)

#c10- Expenses per month
sns.barplot(data= monthly_expenses,x='Month_Name',y='Amount')
plt.xticks(rotation= 90)
plt.title('Expenses Per Month') 
plt.xlabel('Month')
plt.ylabel('Expense Amount')
plt.tight_layout()
plt.show()

#C11- MOnthly spending Trend
plt.plot(monthly_expenses['Month_Name'],monthly_expenses['Amount'], 
         marker='o', linewidth=3, markersize=10, color='#2ecc71')
plt.title('Monthly Spending Trend', fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Daily trends
daily_data = df[df['Income/Expense'] == 'Expense'].groupby(['Date'])['Amount'].sum()

#C12- Daily Transaction Amount
plt.figure(figsize = (12,6))
plt.plot(daily_data.index, daily_data.values, marker='o')
plt.title('Daily Transaction Amounts')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.tight_layout()
plt.show()

#payment mode anlysis


payment_summary = df[df['Income/Expense'] == 'Expense'].groupby('Mode')['Amount'].agg([
    ('Total', 'sum'),('Average', 'mean'),('Count', 'count')
]).round(2)

payment_summary = payment_summary.sort_values('Total', ascending=False)
print("\nPayment Mode Summary:\n ", payment_summary)

#C13- Number of Transactions by Payment Mod
# plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['Income/Expense'] == 'Expense'], x='Mode', palette='viridis',
              order=df[df['Income/Expense'] == 'Expense']['Mode'].value_counts().index)
plt.title('Number of Transactions by Payment Mode')
plt.xlabel('Payment Mode')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.tight_layout()
plt.show()

#C14-
sns.scatterplot(data = df,x = "Income/Expense", y = "Mode")
#plt.title('')
plt.ylabel('Payment Mode')
plt.xlabel('Income/Expense')
plt.tight_layout()
plt.show()

#neeche wala chal raha h but muessy
#C15-Correlation Heatmap of Transaction Categories
'''
pivot_table = df.pivot_table(index='Date', columns='Category', values='Amount',aggfunc='sum', fill_value=0)
correlation_matrix = pivot_table.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Transaction Categories')
plt.show()
'''
top_categories = (df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(10).index)
pivot_table = df.pivot_table(index='Date',columns='Category',values='Amount',aggfunc='sum',fill_value=0)

pivot_top = pivot_table[top_categories]

correlation_matrix = pivot_top.corr().round(2)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap='coolwarm',linewidths=0.5,square=True)

plt.title('Correlation Heatmap of Top 10 Spending Categories')
plt.tight_layout()
plt.show()