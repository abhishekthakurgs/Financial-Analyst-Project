# PROJECT 3: DATA ANALYST JOBS ANALYSIS

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
#print(plt.style.available)
plt.style.use('seaborn-v0_8')

#Insert the CSV in the line below
df = pd.read_csv("D:\projects_data\DataAnalyst.csv")   #insert the CSV between these brackets

pd.set_option("display.max_columns", None)

#df.drop(index)

# Handle missing values and -1/Unknown
df = df.replace('-1', np.nan)
df = df.replace('Unknown', np.nan)

print('No of missing values:', {sum(df.isna().sum())})
df.dropna(inplace=True)

print(f"Duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

print('Total_jobs: ', len(df))

df.columns = (df.columns.str.strip().str.replace(" ", "_"))

# Extract Min and Max Salary from Salary Estimate
# Remove "(Glassdoor est.)" text
df['Salary_Estimate'] = df['Salary_Estimate'].str.replace(r'\(Glassdoor est.\)', '', regex=True)

# Extract min and max salary numbers
df[['MinSalary', 'MaxSalary']] = df['Salary_Estimate'].str.extract(r'\$(\d+)K-\$(\d+)K')

# Convert to numeric
df['MinSalary'] = pd.to_numeric(df['MinSalary'], errors='coerce')
df['MaxSalary'] = pd.to_numeric(df['MaxSalary'], errors='coerce')

# Calculate average salary
df['AvgSalary'] = (df['MinSalary'] + df['MaxSalary']) / 2


#print(df[['Salary_Estimate', 'MinSalary', 'MaxSalary', 'AvgSalary']].head())


# Key metrics
total_jobs = len(df)
avg_salary = df['AvgSalary'].mean().round(2)
median_salary = df['AvgSalary'].median()
min_salary = df['MinSalary'].min()
max_salary = df['MaxSalary'].max()
num_companies = df['Company_Name'].nunique()
num_locations = df['Location'].nunique()
avg_rating = df['Rating'].mean().round(2)


print(f"\n JOB MARKET METRICS:")
print(f"Total Job Postings: {total_jobs:,}")
print(f"Number of Companies: {num_companies:,}")
print(f"Number of Locations: {num_locations:,}")

print(f"\nSALARY INSIGHTS:")
print(f"Average Salary:  {avg_salary}")
print(f"Median Salary: ${median_salary}")
print(f"Salary Range: ${min_salary} - ${max_salary}")

print("\n COMPANY RATINGS:")
print(f"Average Company Rating: {avg_rating}/5.0")


# Extract City and State from Location
df[['City', 'State']] = df['Location'].str.split(',', n=1, expand=True)
df['City'] = df['City'].str.strip()
df['State'] = df['State'].str.strip()

# Location Feature Engineering
df["city"] = df["Location"].str.split(",", expand=True)[0]
df["state"] = df["Location"].str.split(",", expand=True)[1]

# Ratings by Industry
plt.figure(figsize=(14, 7))
sns.boxplot(x='Industry', y='Rating', data=df)
plt.xticks(rotation=90)
plt.title("Company Ratings by Industry")
plt.tight_layout()
plt.show()

# Salary distribution
plt.figure(figsize=(14, 6))
sns.histplot(df['Salary_Estimate'], kde=True, bins=20)
plt.title("Salary Estimate Distribution")
plt.xlabel("Salary")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Skill Extraction (Text Analysis)
df["python_skill"] = df["Job_Description"].str.contains("python", case=False, na=False).astype(int)
df["sql_skill"] = df["Job_Description"].str.contains("sql", case=False, na=False).astype(int)
df["excel_skill"] = df["Job_Description"].str.contains("excel", case=False, na=False).astype(int)

df["tech_skill_score"] = df["python_skill"] + df["sql_skill"] + df["excel_skill"]


df['Job_Title']= df['Job_Title'].replace(
                ['Data Analyst I','Junior Data Analyst1', 'data analyst i', 'Data Analyst Junior', 'data analyst junior',
                'Junior Data Analyst', 'Junior Data AnalystI', 'Junior Data Analystl','Entry Level / Jr. Data Analyst'],
                'Junior Data Analyst', regex=True)

df['Job_Title']= df['Job_Title'].replace(
                ['Sr. Data Analyst', 'sr. data analyst','Sr Data Analyst', 'sr data analyst','senior data analyst', 
                'Senior Data Analyst','Data Analyst III', 'data analyst iii', 'senior data analyst'],
                'Senior Data Analyst', regex=True)

df['Job_Title']=df['Job_Title'].replace(['Data Analyst II','data analyst ii', 'Middle Data Analyst'],
                'Middle Data Analyst', regex=True)

#print(df['Job_Title'].head(50))
# print(df['Job_Title'].unique())

# Average Salary
sns.boxenplot(data=df, x='AvgSalary')
plt.xlabel('Average Salary')
plt.ylabel('Count')
plt.title('Distribution of Average Salary')
plt.tight_layout()
plt.show()

top_jobs = df['Job_Title'].value_counts().head(10)
print(top_jobs)
plt.figure(figsize=(14,7))
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.xlabel('Count')
plt.ylabel('Job Title')
plt.title('Top 10 Jobs')
plt.tight_layout()
plt.show()

# Salary and Job Title
# Average salary by job title (top 10)
salary_by_title = df.groupby('Job_Title')['AvgSalary'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Job Titles by Average Salary:")
print(salary_by_title.round(2))

df_sorted = df.sort_values(by='AvgSalary', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='AvgSalary', y='Job_Title', data=df_sorted, orient='h', order=df_sorted['Job_Title'].value_counts().head(10).index)
plt.xlabel('Average Salary')
plt.ylabel('Job Title')
plt.title('Average Salary by Job Title')
plt.tight_layout()
plt.show()

df.drop(['Salary_Estimate', 'MinSalary','MaxSalary',"python_skill","sql_skill", "excel_skill", 'tech_skill_score'], axis=1, inplace=True)
print(df.head())

# Salary Trends by Location

# Average salary by industry (top 10)
salary_by_industry = df.groupby('Industry')['AvgSalary'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Industries by Average Salary:")
print(salary_by_industry.round(2))

job_location =df.groupby('Location')["AvgSalary"].mean().reset_index()
top_10 = job_location.sort_values(by = "AvgSalary",ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=top_10, x='AvgSalary', y='Location')
plt.title('Salary Trends by Location')
plt.xlabel('AVG Salary (USD)')
plt.ylabel('Location')
plt.tight_layout()
plt.show()

# Top work locations among interviewed

top_locations =df['Location'].value_counts().head(20)

plt.figure(figsize=(13,6))
sns.barplot(x=top_locations.values, y=top_locations.index)
plt.xlabel('Count')
plt.ylabel('Location')
plt.title('Top 20 Locations')
plt.tight_layout()
plt.show()

top_headquarters =df['Headquarters'].value_counts().head(20)
plt.figure(figsize=(13,6))
sns.barplot(x=top_locations.values, y=top_locations.index)
plt.xlabel('Count')
plt.ylabel('Headquarters')
plt.title('Top 20 Locations')
plt.tight_layout()
plt.show()

# Companies by Amount of Employees
df_size = df['Size'].value_counts().head(20)
plt.figure(figsize=(12,6))
sns.barplot(x=df_size.index,y=df_size.values)
plt.xlabel('Size')
plt.ylabel('Count')
plt.title('Size Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Salary by company size
salary_by_size = df.groupby('Size')['AvgSalary'].mean().sort_values(ascending=False)
print("\nAverage Salary by Company Size:")
print(salary_by_size.round(2))

df_sizeXsalary =df.groupby('Size')['AvgSalary'].mean().reset_index()
df_sizeXsalary =df_sizeXsalary.sort_values(by='AvgSalary',ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Size', y='AvgSalary',data=df_sizeXsalary)
plt.xlabel('Company Size')
plt.ylabel('Average Salary')
plt.title('Average Salary by Company Size')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Company Rating distribution
rating_dist = df['Rating'].value_counts().sort_index()
print("\nCompany Rating Distribution:")
print(rating_dist)

sns.boxenplot(data=df, x='Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Rating')
plt.tight_layout()
plt.show()

# Salary by ownership type
salary_by_ownership = df.groupby('Type_of_ownership')['AvgSalary'].mean().sort_values(ascending=False)
print("\nAverage Salary by Ownership Type:")
print(salary_by_ownership.head().round(2))

TOP =df['Type_of_ownership'].value_counts().head(20)
plt.figure(figsize=(9, 6))
sns.barplot(x=TOP.values, y=TOP.index)
plt.xlabel('Count')
plt.ylabel('Type of Ownership')
plt.title('Top 20 Types of Ownership')
plt.tight_layout()
plt.show()

# Top Sectors
df_sector = df['Sector'].value_counts().head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=df_sector.values,y=df_sector.index)
plt.xlabel('Count')
plt.ylabel('Sector')
plt.title('Sector Distribution')
plt.tight_layout()
plt.show()

# Salary by Sector
salary_by_sector = df.groupby('Sector')['AvgSalary'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Sectors by Average Salary:")
print(salary_by_sector.round(2))

average_salary_by_sector =df.groupby('Sector')['AvgSalary'].mean().reset_index()
average_salary_by_sector =average_salary_by_sector.sort_values(by='AvgSalary',ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(y='Sector', x='AvgSalary',data=average_salary_by_sector)
plt.xticks(rotation=90)
plt.xlabel('Sector')
plt.ylabel('Average Salary (Thousands Dollars)')
plt.title('Average Salary by Sector')
plt.tight_layout()
plt.show()