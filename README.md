# SwimCloud Score Prediction

This Jupyter Notebook predicts swim scores for the next two seasons using polynomial regression. It uses historical swim score data to train a model and makes predictions based on that model.

## Prerequisites

Make sure you have the following installed on your machine:

- Python 3.x
- Jupyter Notebook
- Required Python packages: numpy, pandas, scikit-learn, matplotlib

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Required Packages:**

   Install the required Python packages using pip:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

## Running the Notebook

1. **Launch Jupyter Notebook:**

   In your terminal, navigate to the directory containing the notebook and run:

   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**

   In the Jupyter Notebook interface, open the notebook file (`swim_score_prediction.ipynb`).

3. **Run the Notebook:**

   Execute the cells in the notebook to train the polynomial regression model and predict future swim scores.

## Notebook Explanation

### Data Points

The historical swim score data is provided in the notebook, **Replace these with your values from your SwimCloud page (Profile--> Rankings)**:

```python
seasons = ["2016-2017", "2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024"]
scores = [384.2, 273.69, 388.55, 384.53, 404.4, 458.3, 528.65]
```

### Polynomial Regression

The notebook uses polynomial regression to fit the data and make predictions. Here’s the key part of the code:

```python
# Polynomial Regression
poly = PolynomialFeatures(degree=2)  # Adjust the degree if needed
X_poly = poly.fit_transform(years)
model = LinearRegression()
model.fit(X_poly, scores)

# Predict the next two seasons' scores (2025, 2026)
future_years = np.array([2025, 2026]).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predictions = model.predict(future_years_poly)
```

### Plotting

The notebook plots the actual swim scores, the polynomial regression line, and the predicted future scores:

```python
# Plot the data and the polynomial regression line
plt.scatter(data['Year'], data['Score'], color='blue', label='Actual scores')

# Generate a smooth curve for the polynomial fit
year_range = np.linspace(2017, 2026, 300).reshape(-1, 1)
year_range_poly = poly.transform(year_range)
predicted_range = model.predict(year_range_poly)

plt.plot(year_range.flatten(), predicted_range, color='red', label='Polynomial fit')
plt.scatter(future_years.flatten(), predictions, color='green', label='Predicted scores')

# Adjust x-axis labels to show full season
season_labels = ["2016-2017", "2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
plt.xticks(np.append(years.flatten(), future_years.flatten()), season_labels, rotation=45)
plt.xlabel('Season')
plt.ylabel('Score')
plt.title('Swim Scores Over Seasons with Polynomial Predictions')
plt.legend()
plt.show()
```

### Predictions

The notebook prints the predicted scores for the next two seasons:

```python
# Print the predictions
for season, score in zip(["2024-2025", "2025-2026"], predictions):
    print(f"Predicted score for season {season}: {score:.2f}")
```
