# Data Science Career Trends and Salary Prediction: Roles, Skills, and Market Growth  
---

## Table of Contents  
1. [Introduction](#introduction)  
   - [Motivation](#motivation)  
   - [Data Collection](#data-collection)  
   - [Dataset Description](#dataset-description)  
2. [Salary Estimation](#salary-estimation)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Derived Metrics](#derived-metrics)  
   - [Key Observations](#key-observations)  
3. [Location Analysis](#location-analysis)  
4. [Job Title and Description Analysis](#job-title-and-description-analysis)  
5. [Skills Analysis and Industry Insights](#skills-analysis-and-industry-insights)  
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)  
7. [Conclusion and Future Directions](#conclusion-and-future-directions)  

---

## Introduction  
### Motivation  
Data science is revolutionizing industries globally. The motivation for this project is to analyze the rapidly growing data science job market by studying roles, skills, salary trends, and market growth using the "Data Science Jobs 2024" dataset.  

### Data Collection  
Data was collected from Glassdoor using Selenium to scrape job listings, capturing critical details like job titles, salaries, and company information.  

### Dataset Description  
The dataset contains 5,745 job listings, including detailed attributes such as salaries, company size, founding year, and industry. Preprocessing addressed missing values and ensured robust analysis.  

---

## Salary Estimation  
### Data Preprocessing  
- Standardized salary data by removing non-numeric elements and converting hourly rates to annual equivalents.  
- Imputed missing values using strategies based on context, such as grouping by job title.  

### Derived Metrics  
Created new columns: `Min Salary`, `Max Salary`, `Average Salary`, and annualized hourly rates.  

### Key Observations  
- Competitive hourly roles with annualized salaries.  
- Senior positions reaching up to $160,000 annually.  

---

## Location Analysis  
### Data Preprocessing  
Standardized city and state names; categorized remote jobs.  

### Geographic Distribution  
- Roles span 33 states, with high demand in California, Texas, and New York.  
- Remote jobs offer flexible opportunities.  

---

## Job Title and Description Analysis  
### Refinement of Job Titles  
Mapped job titles to categories (e.g., Data Scientist, Data Engineer) and seniority levels (Junior, Mid-Level, etc.).  

### Job Description Preprocessing  
Cleaned descriptions to extract key skills such as Python, machine learning, and data analysis.  

---

## Skills Analysis and Industry Insights  
### Skills Analysis  
Applied TF-IDF to rank skills, highlighting Python, machine learning, and SQL as the most in-demand.  

### Company Insights  
- Majority of roles are in IT and private sectors.  
- Companies with revenues between $5M and $25M dominate.  

---

## Exploratory Data Analysis (EDA)  
### Salary Comparisons  
- AI roles have the highest average and maximum salaries, exceeding $200,000.  
- Data Scientist and Data Engineer roles show strong salary growth with experience.  

### Geographic Salary Trends  
- Rhode Island offers the highest average salary (> $200,000), followed by California and Washington.  

### Experience vs. Salary  
- Clear correlation: salaries increase significantly with experience, peaking after 10 years.  

### Company Age and Salary  
- Older companies (> 25 years) offer the highest average salaries.  
- Startups show more variability due to funding constraints.  

---

## Conclusion and Future Directions  
This project provides insights into trends in data science careers, emphasizing the importance of key skills like Python, machine learning, and SQL. Future research could focus on regional salary trends and emerging skills shaping the industry.  

---

## Repository Structure  
- `data/`: Contains the dataset used for analysis.  
- `notebooks/`: Jupyter notebooks for data preprocessing, analysis, and modeling.  
- `scripts/`: Python scripts for Selenium scraping and data cleaning.  
- `visualizations/`: Charts and graphs from exploratory analysis.  

---

## Getting Started  
1. Clone the repository:  
   ```bash
   git clone https://github.com/username/data-science-career-trends.git
   cd data-science-career-trends
