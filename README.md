# Selenium Job Scraper

## Overview
A web scraping tool designed to extract comprehensive job listings from Glassdoor using Selenium.

## Description
Selenium Job Scraper is an automated web scraping tool that extracts detailed job information from Glassdoor based on user-defined keywords. The scraper navigates through job listings and collects essential details including job titles, company names, locations, salary estimates, job descriptions, required skills, company ratings, and direct URLs to the job postings. This tool enables users to efficiently gather and analyze job market data for research, job searching, or market analysis.

## Getting Started
### Dependencies
* Python 3.x
* Selenium
* WebDriver (e.g., ChromeDriver)
* pandas
* Time

## Installing
1. Clone the repository from GitHub:
bash
```git clone https://github.com/yourusername/selenium-job-scraper.git```
2. Install the required packages
3. Download and install the appropriate WebDriver for your browser and ensure it is in your system's PATH.
## Executing program
1. Open the Selenium.ipynb 
2. Run the code (or copy the entire code and paste it into new file with .py extension and run)
3. Output will be saved with name  `jobs.csv`

## Help
If you encounter common issues such as:

* **WebDriver not found:** Ensure the WebDriver is correctly installed and in your system's PATH.
* **Element not found:** Verify that the page structure on Glassdoor hasn't changed. Update the selectors in the script if necessary.

## Authors
* **@moiz-punisher**
## Version History
* **1.0**
    * Initial Release
* **1.1**
    * Various bug fixes and optimizations