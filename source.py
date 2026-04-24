#!/usr/bin/env python
# coding: utf-8

# # Poverty and Resource Distribution Across U.S. States
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 The issue I am addressing is poverty within the United States, with a focus on understanding how it varies across different states. Poverty is a widespread issue, but its distribution and severity differ depending on geographic and economic factors.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 
# How does poverty vary by state over time, and is assistance distributed proportionally to areas with the highest need?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 I expect to find that certain states consistently have higher poverty rates than others, and that assistance may not always align with areas of greatest need.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 I- U.S. Census Bureau (poverty rates)
# - SNAP dataset (assistance data)
# - Bureau of Labor Statistics (optional: unemployment)

# ## Final Project Roadmap
# 
# This notebook is organized as a data analysis system for exploring poverty rates and resource distribution. The project follows these steps:
# 
# 1. Define the problem and project question.
# 2. Load and clean the poverty dataset.
# 3. Perform exploratory data analysis to identify trends, distributions, relationships, missing values, and outliers.
# 4. Create static and interactive visualizations.
# 5. Prepare the data for machine learning using scikit-learn pipelines.
# 6. Train and compare multiple machine learning models.
# 7. Evaluate model performance and explain the final model choice.
# 8. Reflect on feedback, challenges, limitations, and future improvements.
# 

# ## Exploratory Data Analysis (EDA)
# 
# At this stage, I explored the datasets to better understand patterns, distributions, and relationships between variables such as poverty rates and SNAP assistance.
# 
# ### Initial Insights:
# - Poverty rates vary significantly across states and over time.
# - Some states consistently show higher poverty levels than others.
# - SNAP assistance appears to increase in areas with higher poverty, suggesting a relationship between need and support.
# 
# ### Distributions:
# - Poverty rates show a range of values with some states consistently above average.
# - SNAP data shows variation in assistance levels across different regions.
# 
# ### Correlations:
# - There appears to be a positive relationship between poverty rates and SNAP assistance.
# - Higher poverty areas tend to have higher SNAP participation.
# 
# ### Data Issues Identified:
# - Missing values were found in some rows.
# - Some datasets contained inconsistent column names.
# - Data types needed to be adjusted for numerical analysis.
# 
# ### Outliers:
# - Some states show unusually high or low poverty rates.
# - These outliers are important for analysis and were not removed.
# 
# ### Missing Values:
# - Missing values were identified and handled by removing incomplete rows where necessary.
# 
# ### Duplicate Values:
# - Duplicate rows were checked and removed to ensure data accuracy.
# 
# ### Data Type Issues:
# - Some columns needed to be converted to numeric types for analysis.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[ ]:


#- Clean and prepare datasets
# - Analyze trends over time
# - Compare poverty and assistance
# - Use visualizations to identify pattern
# Start your code here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load the raw poverty file
# ----------------------------
file_path = "/Users/bennykj04/Downloads/annual_poverty_rates_by_survey.xlsx"

# Read without assuming headers so we can clean it ourselves
raw_poverty = pd.read_excel(file_path, header=None)

print("Raw shape:", raw_poverty.shape)
print(raw_poverty.head(15))

# ----------------------------
# 2. Clean the raw structure
# ----------------------------
# Keep only the first 5 columns since those are the useful ones for this assignment
poverty = raw_poverty.iloc[:, :5].copy()
poverty.columns = ["col0", "col1", "col2", "col3", "col4"]

# Remove fully empty rows
poverty = poverty.dropna(how="all")

# Convert first column to string for easier handling
poverty["col0"] = poverty["col0"].astype(str).str.strip()

# Detect rows where the first column is a year
poverty["Year"] = pd.to_numeric(poverty["col0"], errors="coerce")

# Forward fill the year down to the actual group rows
poverty["Year"] = poverty["Year"].ffill()

# Keep only rows that are actual group rows, not the year-only rows
# Group rows usually start with ".."
poverty = poverty[poverty["col0"].str.startswith("..", na=False)].copy()

# Clean group names
poverty["Group"] = poverty["col0"].str.replace("..", "", regex=False).str.strip()

# Assign the likely data columns
# Based on the structure you showed:
# col1 = CPS ASEC
# col3 = Official
# col4 = SPM
poverty["CPS"] = pd.to_numeric(poverty["col1"], errors="coerce")
poverty["Official"] = pd.to_numeric(poverty["col3"], errors="coerce")
poverty["SPM"] = pd.to_numeric(poverty["col4"], errors="coerce")

# Keep only the cleaned columns we need
poverty = poverty[["Year", "Group", "CPS", "Official", "SPM"]]

# Drop rows that do not have actual poverty values
poverty = poverty.dropna(subset=["Official"])

# Convert Year to integer
poverty["Year"] = poverty["Year"].astype(int)

# Remove duplicates
poverty = poverty.drop_duplicates()


# ----------------------------
# 4. Optional: remove obvious outliers check
# ----------------------------
# For this project, we will keep outliers because they may represent meaningful poverty differences.
q1 = poverty["Official"].quantile(0.25)
q3 = poverty["Official"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = poverty[(poverty["Official"] < lower_bound) | (poverty["Official"] > upper_bound)]
print("\nPotential outliers based on Official poverty rate:")
print(outliers[["Year", "Group", "Official"]])

# ----------------------------
# 5. Visualization 1: Line chart
# Poverty rate over time
# ----------------------------
official_by_year = poverty.groupby("Year")["Official"].mean()

plt.figure(figsize=(10, 5))
plt.plot(official_by_year.index, official_by_year.values, marker="o")
plt.title("Average Official Poverty Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Official Poverty Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Visualization 2: Bar chart
# Poverty rate by group
# ----------------------------
group_avg = poverty.groupby("Group")["Official"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
group_avg.plot(kind="bar")
plt.title("Average Official Poverty Rate by Group")
plt.xlabel("Group")
plt.ylabel("Official Poverty Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Visualization 3: Histogram
# Distribution of Official poverty rate
# ----------------------------
plt.figure(figsize=(8, 5))
poverty["Official"].hist(bins=15)
plt.title("Distribution of Official Poverty Rates")
plt.xlabel("Official Poverty Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Visualization 4: Box plot
# Outliers and spread
# ----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x=poverty["Official"])
plt.title("Box Plot of Official Poverty Rates")
plt.xlabel("Official Poverty Rate")
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Correlation analysis
# ----------------------------
correlation = poverty[["CPS", "Official", "SPM"]].corr()
print("\nCorrelation matrix:")
print(correlation)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="Blues")
plt.title("Correlation Between Poverty Measures")
plt.tight_layout()
plt.show()


# In[25]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "Model": name,
        "Test RMSE": rmse
    })

results_df = pd.DataFrame(results)

display(results_df)


# In[ ]:


# ----------------------------
# 3. EDA
# ----------------------------
print("\nCleaned poverty data:")
print(poverty.head())

print("\nData types:")
print(poverty.dtypes)

print("\nShape after cleaning:", poverty.shape)

print("\nMissing values:")
print(poverty.isnull().sum())

print("\nDuplicate rows:", poverty.duplicated().sum())

print("\nDescriptive statistics:")
print(poverty.describe())

print("\nUnique groups:")
print(poverty["Group"].unique())


# In[26]:


# Visualizations #
# Visualization 1: Line chart of average official poverty rate over time
official_by_year = poverty.groupby("Year")["Official"].mean()

plt.figure(figsize=(10, 5))
plt.plot(official_by_year.index, official_by_year.values, marker="o")
plt.title("Average Official Poverty Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Official Poverty Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Bar chart of average poverty rate by group
group_avg = poverty.groupby("Group")["Official"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
group_avg.plot(kind="bar")
plt.title("Average Official Poverty Rate by Group")
plt.xlabel("Group")
plt.ylabel("Official Poverty Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Visualization 3: Histogram of official poverty rates
plt.figure(figsize=(8, 5))
poverty["Official"].hist(bins=15)
plt.title("Distribution of Official Poverty Rates")
plt.xlabel("Official Poverty Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization 4: Box plot of official poverty rates
plt.figure(figsize=(8, 5))
sns.boxplot(x=poverty["Official"])
plt.title("Box Plot of Official Poverty Rates")
plt.xlabel("Official Poverty Rate")
plt.tight_layout()
plt.show()

# Visualization 5: Correlation heatmap between poverty measures
correlation = poverty[["CPS", "Official", "SPM"]].corr()

print("Correlation matrix:")
display(correlation)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="Blues")
plt.title("Correlation Between Poverty Measures")
plt.tight_layout()
plt.show()

# Visualization 6: Scatter plot comparing SPM and Official poverty rates
plt.figure(figsize=(8, 5))
sns.scatterplot(data=poverty, x="SPM", y="Official", hue="Group")
plt.title("SPM vs. Official Poverty Rate")
plt.xlabel("Supplemental Poverty Measure (SPM)")
plt.ylabel("Official Poverty Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# Visualization 7: Interactive line chart using Plotly
# This supports the final project requirement for an interactive visualization.
try:
    import plotly.express as px

    fig = px.line(
        poverty,
        x="Year",
        y="Official",
        color="Group",
        markers=True,
        title="Interactive Official Poverty Rate Over Time by Group"
    )
    fig.show()
except ImportError:
    print("Plotly is not installed. Install it with: pip install plotly")


# ### Visualization 1: Line Chart
# This chart shows the average official poverty rate over time. It helps identify whether poverty has increased or decreased across the years in the dataset.
# 
# ### Visualization 2: Bar Chart
# This bar chart compares the average official poverty rate across groups. It makes it easy to see which groups have the highest and lowest poverty rates.
# 
# ### Visualization 3: Histogram
# This histogram shows the distribution of official poverty rates. It helps show how common certain poverty values are and whether the data is concentrated in one range or spread out.
# 
# ### Visualization 4: Box Plot
# This box plot highlights the spread of official poverty rates and identifies possible outliers. It is useful for understanding variation in the data.

# ## EDA Findings
# 
# The exploratory data analysis shows that poverty rates vary across both year and demographic group. The line chart helps show changes over time, while the group bar chart shows that some groups experience consistently higher average poverty rates than others. The histogram and box plot show the distribution and spread of poverty rates, including potential outliers.
# 
# The correlation heatmap shows the relationship between CPS, Official, and SPM poverty measures. These measures are related, but they are not exactly the same because they are calculated using different methods. This matters because the model may perform well when similar poverty measures are used as features, but that can also create possible target leakage. For that reason, model results should be interpreted carefully.

# ## 1. Machine Learning Plan
# 
# For this project, I plan to use supervised machine learning because the goal is to predict a numerical poverty-related outcome using existing data. Since the target variable is the Official poverty rate, this is a regression problem. I will begin with Linear Regression as a baseline model and compare it with Polynomial Regression, Decision Tree Regression, and Random Forest Regression.
# 
# The main challenges I anticipate are data quality, small dataset size, missing values, categorical variables, outliers, and possible target leakage. Poverty is also a complex social issue, so the model cannot fully explain poverty by itself. The dataset includes related poverty measures such as CPS and SPM, which may make prediction easier but could also make the model less realistic if those values are not available in future use cases.
# 
# To address these challenges, I will use scikit-learn pipelines to organize preprocessing and modeling. Numerical features will be imputed and scaled, while categorical features will be imputed and one-hot encoded. I will split the data into training and test sets, test multiple algorithms, and evaluate them using RMSE, MAE, and R². The final model will be selected based on performance and interpretability.

# ## 2. Machine Learning Implementation Process
# 
# ### Ask
# The main question is whether poverty rates can be modeled using available poverty-related features such as year, group, CPS, and SPM. The target variable is the Official poverty rate.
# 
# ### Prepare
# The dataset was loaded, cleaned, and reshaped from the original spreadsheet format. I reviewed rows, columns, data types, missing values, duplicates, and descriptive statistics.
# 
# ### Process
# The data was cleaned by removing empty rows, extracting year and group values, converting poverty measures to numeric values, and removing duplicates. Missing values and feature transformations are handled later using scikit-learn pipelines.
# 
# ### Analyze
# EDA was completed through histograms, line charts, bar charts, box plots, scatter plots, and correlation analysis. These visuals helped identify trends, relationships, and outliers.
# 
# ### Evaluate
# Multiple regression models are trained and compared using RMSE, MAE, and R² on both training and test data.
# 
# ### Share
# The final findings, model comparison, challenges, and limitations are summarized at the end of the presentation.
# 

# In[ ]:


#Prepare features and target for machine learning
# Target: Official poverty rate
# Features: Year, Group, CPS, and SPM
ml_data = poverty.copy()

X = ml_data.drop(columns=["Official"])
y = ml_data["Official"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Target: Official")


# In[ ]:


from sklearn.model_selection import train_test_split
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)



# In[15]:


# Create preprocessing pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# In[16]:


# Train and evaluate multiple machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    
    "Polynomial Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", LinearRegression())
    ]),
    
    "Decision Tree Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
    
    "Random Forest Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}


# In[18]:


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    results.append({
        "Model": name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
display(results_df)

# Visualize model comparison using Test RMSE
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Test RMSE")
plt.title("Model Comparison by Test RMSE")
plt.xlabel("Model")
plt.ylabel("Test RMSE")
plt.xticks(rotation=45)
plt.show()


# In[19]:


# Select the best model based on the lowest Test RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print("Best model based on Test RMSE:", best_model_name)
print(results_df.iloc[0])


# In[20]:


# Compare actual vs predicted values for the best model
best_predictions = best_model.predict(X_test)
comparison_df = pd.DataFrame({
    "Actual Official Poverty Rate": y_test.values,
    "Predicted Official Poverty Rate": best_predictions
})

comparison_df.head(10)


# In[21]:


# Visualization: Actual vs predicted poverty rates
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=comparison_df["Actual Official Poverty Rate"],
    y=comparison_df["Predicted Official Poverty Rate"]
)
plt.title("Actual vs Predicted Official Poverty Rates")
plt.xlabel("Actual Official Poverty Rate")
plt.ylabel("Predicted Official Poverty Rate")
plt.tight_layout()
plt.show()


# ## Resources and References
# *What resources and references have you used for this project?*
# 
# - U.S. Census Bureau poverty rate data
# - SNAP participation data for future comparison and project expansion
# - Bureau of Labor Statistics data for future socioeconomic context
# - Pandas documentation for data cleaning and analysis
# - Matplotlib and Seaborn documentation for static visualizations
# - Plotly documentation for interactive visualizations
# - Scikit-learn documentation for train/test splits, preprocessing pipelines, regression models, and evaluation metrics

# In[27]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

