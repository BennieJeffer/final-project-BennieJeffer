# Poverty Trends and Assistance Analysis
<!-- Edit the title above with your project title -->

## Project Overview
<!--This project analyzes poverty trends over time and explores whether assistance aligns with changes in poverty rates. The goal is to use data-driven methods to better understand patterns in poverty and evaluate how well predictive models can estimate poverty rates.

The project begins with data collection and cleaning, where raw datasets were restructured and missing values were handled. Exploratory Data Analysis (EDA) was then performed to identify trends and relationships using visualizations such as histograms, line charts, and bar graphs.

To extend the analysis, machine learning models were implemented, including Linear Regression, Polynomial Regression, Decision Tree Regression, and Random Forest Regression. Data preprocessing was handled using pipelines that included imputation, scaling, and encoding of categorical variables. The dataset was split into training and testing sets, and models were evaluated using Root Mean Squared Error (RMSE).

The results showed that Linear Regression performed best for this dataset, indicating that the relationship between features and poverty rate is relatively linear. This project demonstrates how data analysis and machine learning can be applied to real-world social issues. -->
## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | 9/ 10 |
| **Execution**     | 18 / 20 |
| **Documentation** | 9/ 10 |
| **Presentation**  | 27 / 30 |
| **Total**         | 63/ 70 |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
The data cleaning and preprocessing process went well, especially using pipelines to organize transformations. The machine learning models were successfully implemented and evaluated, and the visualizations helped clearly show trends in the data. Additionally, the project successfully met the requirement of combining EDA with machine learning techniques.
#### What did not go well?
One of the main challenges was dealing with raw datasets that were not well-structured. This required additional time to clean and reshape the data before analysis could begin. There were also several errors encountered during the machine learning implementation, particularly related to missing imports and undefined variables, which required debugging.
#### What did you learn?
I learned how to apply the full data analysis workflow, including data cleaning, exploratory data analysis, and machine learning. I also gained a better understanding of pipelines in scikit-learn and how they streamline preprocessing steps. Additionally, I improved my debugging skills and learned how to interpret model evaluation metrics such as RMSE.
#### What would you do differently next time?
Next time, I would spend more time selecting and preparing datasets earlier in the process to reduce cleaning complexity later on. I would also explore additional models or tuning techniques to improve performance. Finally, I would plan my workflow more thoroughly to reduce the number of errors encountered during development.
---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)