# PROJECT 4: DATA SCIENCE JOB SALARIES

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
#print(plt.style.available)
plt.style.use('seaborn-v0_8')

# Insert the CSV location in the line below
df = pd.read_csv("D:\projects_data\Data_Science_Job_Salaries.csv")

pd.set_option("display.max_columns", None)
df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.head())

print('No of missing values:', {sum(df.isna().sum())})
df.dropna(inplace=True)

print(f"Duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

print('Total_jobs: ', len(df))

# Standardize column names
df.columns = (df.columns.str.strip().str.replace(" ", "_"))

df["experience_level"] = df["experience_level"].map({
    "EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"
})

df["employment_type"] = df["employment_type"].map({
"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"
})

df["company_size"] = df["company_size"].map({
    "S": "Small", "M": "Medium", "L": "Large"
})

df["job_type"] = df["remote_ratio"].map({
    0: "Onsite", 50: "Hybrid",100: "Remote"
})

# Add a salary ratio feature
df['salary_ratio'] = df['salary'] / df['salary_in_usd']

# Key metrics
total_jobs = len(df)
avg_salary = df['salary_in_usd'].mean()
median_salary = df['salary_in_usd'].median()
min_salary = df['salary_in_usd'].min()
max_salary = df['salary_in_usd'].max()
num_countries = df['company_location'].nunique()
num_titles = df['job_title'].nunique()
years_covered = df['work_year'].unique()

print(f"\n DATASET OVERVIEW:")
print(f"Total Jobs Analyzed: {total_jobs:,}")
print(f"Number of Job Titles: {num_titles}")
print(f"Number of Countries: {num_countries}")
print(f"Years Covered: {sorted(years_covered)}")

print(f"\n SALARY INSIGHTS:")
print(f"Average Salary: ${avg_salary:,.0f}")
print(f"Median Salary: ${median_salary:,.0f}")
print(f"Salary Range: ${min_salary:,.0f} - ${max_salary:,.0f}")


# drop salary and salary_currency features 
df.drop(['salary', 'salary_currency'], axis=1, inplace=True)
df.rename(columns={'salary_in_usd': 'salary'}, inplace=True)

plt.figure()
plt.hist(df["salary"], bins=40)
plt.title("Salary Distribution (USD)", fontsize=16)
plt.xlabel("Salary", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.tight_layout()
plt.show()

#Salary vs Experience level

exp_stats = df.groupby('experience_level')['salary'].agg([
    ('Average', 'mean'),('Median', 'median'),
    ('Count', 'count'),('Min', 'min'),('Max', 'max')
]).round(0)
print("\nComplete Experience Level Statistics:")
print(exp_stats)

salary_by_experience = (df.groupby("experience_level")["salary"].mean().sort_values())

# plt.figure()
# plt.bar(salary_by_experience.index, salary_by_experience.values)
# plt.title("Average Salary by Experience Level")
# plt.xlabel("Experience Level")
# plt.ylabel("Average Salary")
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

sns.barplot(x=salary_by_experience.index,y=salary_by_experience.values,ax=axes[0])
axes[0].set_title('Mean Salary vs Experience Level', fontsize=16)
axes[0].set_xlabel('Experience Level', fontsize=13)
axes[0].set_ylabel('Salary', fontsize=13)

sns.violinplot(data=df,x='experience_level',y='salary',ax=axes[1])
axes[1].set_title('Experience Level vs Salary', fontsize=16)
axes[1].set_xlabel('Experience Level', fontsize=13)

plt.tight_layout()
plt.show()

# Salary vs Employment Type
salary_by_employment = (df.groupby("employment_type")["salary"].mean().sort_values())
print('Average salary per employee type\n ', salary_by_employment)

# plt.figure()
# plt.bar(salary_by_employment.index, salary_by_employment.values)
# plt.title("Average Salary by Employment Type")
# plt.xlabel("Employment Type")
# plt.ylabel("Average Salary")
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

sns.barplot(x=salary_by_employment.index, y=salary_by_employment,ax=axes[0])
axes[0].set_title('Mean Salary Vs Employment Type', fontsize=16)
axes[0].set_xlabel('Employment Type', fontsize=13)
axes[0].set_ylabel('Salary', fontsize=13)

sns.boxplot(data=df,x='employment_type',y='salary',ax=axes[1])
axes[1].set_title('Employment Type vs Salary', fontsize=16)
axes[1].set_xlabel('Employment Type', fontsize=13)

plt.tight_layout()
plt.show()

# Salary vs Company Size
salary_by_company_size = (df.groupby("company_size")["salary"].mean().sort_values())
print('Average salary by company size\n',salary_by_company_size)

# plt.figure()
# plt.bar(salary_by_company_size.index, salary_by_company_size.values)
# plt.title("Average Salary by Company Size")
# plt.xlabel("Company Size")
# plt.ylabel("Average Salary")
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

sns.barplot(x=salary_by_company_size.index, y=salary_by_company_size,ax=axes[0])
axes[0].set_title('Mean Salary VS Company Size', fontsize=16)
axes[0].set_xlabel('Company Size', fontsize=13)
axes[0].set_ylabel('Salary', fontsize=13)

sns.boxenplot(data=df,x='company_size', y='salary',ax=axes[1])
axes[1].set_title('Company Size VS Salary', fontsize=16)
axes[1].set_xlabel('Company Size', fontsize=13)

plt.tight_layout()
plt.show()

# Salary vs Job Type (Remote / Hybrid / Onsite)

# Salary by remote ratio
salary_by_job_type = (df.groupby("job_type")["salary"].mean().sort_values())

print("\nAverage Salary by Work Location:")
print(salary_by_job_type.round(0))

# Remote work distribution
remote_dist = df['job_type'].value_counts()
print("\nRemote Work Distribution:")
print(remote_dist)

# Remote work percentage
remote_pct = (remote_dist / total_jobs * 100).round(2)
print("\nRemote Work Percentage:")
print(remote_pct)

# Remote work by experience level
remote_by_exp = df.groupby(['experience_level', 'job_type']).size().unstack(fill_value=0)
print("\nRemote Work by Experience Level:")
print(remote_by_exp)

# plt.figure()
# plt.bar(salary_by_job_type.index, salary_by_job_type.values)
# plt.title("Average Salary by Job Type")
# plt.xlabel("Job Type")
# plt.ylabel("Average Salary")
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

sns.barplot(x=salary_by_job_type.index, y=salary_by_job_type,ax=axes[0])
axes[0].set_title('Mean Salary VS Job Type', fontsize=16)
axes[0].set_xlabel('Job Type', fontsize=13)
axes[0].set_ylabel('Salary', fontsize=13)

sns.violinplot(data=df, x='job_type', y='salary',ax=axes[1])
axes[1].set_title('Job Type VS Salary', fontsize=16)
axes[1].set_xlabel('Job Type', fontsize=13)

plt.tight_layout()
plt.show()

# job type and company size VS salary

plt.figure(figsize=(14, 7))
sns.set_palette('Set2')
ax = sns.boxenplot(data=df, x='job_type', y='salary', hue='company_size')
ax.set_xlabel('Job Type', fontsize=13)
ax.set_ylabel('Salary', fontsize=13)
ax.legend(title='Company Size')
ax.set_title('Job Type & Company Size VS Salary', fontsize=16)
plt.tight_layout()
plt.show()

#Job Types and Experience Level distributions (Pie)

plt.figure(figsize=(12, 5))
sns.set_palette('Set2')
plt.subplot(1,2,1)
ax = df['job_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
ax.set_title('Job Type', fontsize=16)
plt.subplot(1,2,2)
ax = df['experience_level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
ax.set_title('Experience Level', fontsize=16)
plt.tight_layout()
plt.show()

#top 10 ds salaries

# top 10 data science roles according to mean salary
top_roles =df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
top_roles_filtered = top_roles[df["job_title"].value_counts()[top_roles.index] > 1].head(10)
print('top 10 data science roles according to mean salary\n',top_roles_filtered.round(2))


# plt.figure()
# plt.barh(top_roles_filtered.index, top_roles_filtered.values)
# plt.title("Top 10 Highest Paying Data Science Roles")
# plt.xlabel("Average Salary")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()
 
plt.figure(figsize=(18, 5))
# top 10 data science roles according to mean salary
plt.subplot(1, 2, 1)
top_roles = top_roles_filtered
ax = sns.barplot(y=top_roles.index, x=top_roles)
ax.set_ylabel('Job Title', fontsize=13)
ax.set_xlabel('Mean Salary', fontsize=13)
ax.set_title('Top DS roles according to mean salary', fontsize=16)
# top 10 data science roles with highest number of openings
plt.subplot(1, 2,2,)
top_dr = df['job_title'].value_counts().head(10)
ax = sns.barplot(x=top_dr, y=top_dr.index)
ax.set_ylabel('')
ax.set_xlabel('Job Openings', fontsize=13)
ax.set_title('Top 10 data science roles with highest number of openings', fontsize=16)
plt.tight_layout()
plt.show()

# top 10 company-locations according to mean salary
top_cmp_locations = df.groupby('company_location')['salary'].mean().sort_values(ascending=False).head(10).round(2)
print('top 10 company-locations according to mean salary\n',top_cmp_locations)

plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
ax = sns.barplot(y=top_cmp_locations.index, x=top_cmp_locations)
ax.set_ylabel('Company Location', fontsize=13)
ax.set_xlabel('Mean Salary', fontsize=13)
ax.set_title('Top 10 countries according to DS mean salaries', fontsize=16)

top_cl = df['company_location'].value_counts().head(10)
plt.subplot(1, 2, 2)
ax = sns.barplot(x=top_cl, y=top_cl.index)
ax.set_ylabel('')
ax.set_xlabel('Number of Job Opportunities', fontsize=13)
ax.set_title('Top 10 countries having most DS job opportunities', fontsize=16)
plt.tight_layout()
plt.show()

# top 10 employee-residence according to mean salary
top_emp_residence = df.groupby('employee_residence')['salary'].mean().sort_values(ascending=False).head(10).round(2)
print('top 10 employee-residence according to mean salary\n',top_emp_residence)

plt.figure(figsize=(16, 5))
plt.subplot(1,2,1)
ax = sns.barplot(y=top_emp_residence.index, x=top_emp_residence)
ax.set_ylabel('Employee residence', fontsize=13)
ax.set_xlabel('Mean Salary', fontsize=13)
ax.set_title('Top 10 employee-residence according to mean DS salary', fontsize=16)

plt.subplot(1,2,2)
top_er = df['employee_residence'].value_counts().head(10)
ax = sns.barplot(x=top_er, y=top_er.index)
ax.set_title('Top 10 countries having most DS employees', fontsize=16)
ax.set_ylabel('')
ax.set_xlabel('Job Openings', fontsize=13)
plt.tight_layout()
plt.show()

#Company Size VS Job Types Counts

plt.figure(figsize=(10, 5))
sns.set_palette('Set2')
ax = sns.countplot(data=df, x='company_size', hue='job_type')
ax.set_title('Company Size VS Job Types Counts', fontsize=16)
ax.set_ylabel('Counts', fontsize=13)
ax.set_xlabel('Company Size', fontsize=13)
ax.legend(title= 'Job Type')
plt.tight_layout()
plt.show()

# Job count by year
yearly_jobs = df['work_year'].value_counts().sort_index()
print("\nJob Count by Year:")
print(yearly_jobs)

# Work Year Salary Trend
salary_by_year = (df.groupby("work_year")["salary"].mean().sort_index())
print("\nAverage Salary by Year:\n", salary_by_year)

plt.figure()
plt.plot(salary_by_year.index, salary_by_year.values, marker="o")
plt.title("Average Data Science Salary Trend Over Time", fontsize=16)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Average Salary", fontsize=13)
plt.tight_layout()
plt.show()
