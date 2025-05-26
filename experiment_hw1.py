import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


# Load the dataset
df = pd.read_csv('homework_1.1.csv')

# Define features and target
X = df[['X1', 'X2', 'X3']]
y = df['Y']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict using the model
y_pred = model.predict(X)

# Print model coefficients and evaluation metrics
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R^2 Score:", r2_score(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))

#Finding the differences in the code:

differences = {}

for xi in ['X1', 'X2', 'X3']:
    # Simple regression: Y ~ Xi
    model_simple = LinearRegression().fit(df[[xi]], df['Y'])
    beta_simple = model_simple.coef_[0]
    
    # Multiple regression: Y ~ X1 + X2 + X3
    model_multi = LinearRegression().fit(df[['X1', 'X2', 'X3']], df['Y'])
    beta_multi = model_multi.coef_[['X1', 'X2', 'X3'].index(xi)]
    
    # Difference
    diff = abs(beta_simple - beta_multi)
    differences[xi] = diff
    print(f"{xi}: Simple = {beta_simple:.4f}, Multiple = {beta_multi:.4f}, Difference = {diff:.4f}")

# Find the Xi with the greatest difference
max_xi = max(differences, key=differences.get)
print(f"\n {max_xi} has the greatest difference between simple and multiple regression coefficients.")

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print t-statistics
print(model.summary())