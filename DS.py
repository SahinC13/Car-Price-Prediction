import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

car_sales_path = r"file_path"
df = pd.read_csv(car_sales_path)

# Adding Age
df['age'] = 2026 - df['year']

# fuel and transmission
df['fuel_trans'] = df['fuel'] + '_' + df['transmission']

# Seller and owner
df['seller_owner'] = df['seller_type'] + '_' + df['owner']

# Age/KM
df['age_per_km'] = df['km_driven'] / df['age']

# Remove year
df.drop('year', axis=1, inplace=True)

# Adjusting brand column
df["brand"] = df['name'].str.split().str[0]
brand_counts = df['brand'].value_counts()
rare_brands = brand_counts[brand_counts < 10].index
df['brand'] = df['brand'].replace(rare_brands, 'Other')

# Adding counts(counts the amount sold for each brand)
counts = df['brand'].value_counts()
df['counts'] = df['brand'].map(counts)

# Removing outliers
q1, median, q3 = df['selling_price'].quantile([0.25,0.5,0.75])
lower_limit = q1 - 1.5*(q3-q1)
upper_limit = q3 + 1.5*(q3-q1)
df = df[(df['selling_price'] <= upper_limit) & (df['selling_price'] >= lower_limit)]
for col in ['selling_price','age_per_km','counts']:
  upper_limit = df[col].quantile(0.99)
  df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

# _________________________________________________________________________________________________________
# THIS IS LINEAR REGRESSION SECTION, COMMENT OUT FOR XGBoost
"""
# Feature Selection
continuous_feature_df  = ['selling_price','age_per_km','counts']
calc_vif(df[[i for i in continuous_feature_df]])

df_encoded = pd.get_dummies(df, drop_first=True).squeeze()
"""

# _________________________________________________________________________________________________________
# THIS IS XGBoost SECTION, COMMENT OUT FOR LINEAR REGRESSION
"""
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name']) 
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'name_encoded'] 
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X_xgb = df.drop(['selling_price', 'name'], axis=1) # Drop raw text 'name'
y_xgb = np.log1p(df['selling_price']) # Log transform helps XGBoost too

x_train, x_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
"""

# This XGBoost had score of 0.8223832370384574
"""
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(x_train, y_train)

predictions = np.expm1(model.predict(x_test))
actuals = np.expm1(y_test)
print(f"XGBoost Optimized Score: {r2_score(actuals, predictions)}")
"""

# This is grid_search model, had score of 0.8227354290338529
"""
param_grid = {
    'n_estimators': [100, 500, 1000],   
    'max_depth': [3, 5, 7],             
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8]
} 
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
log_predictions = best_model.predict(x_test)
real_predictions = np.expm1(log_predictions)
real_actuals = np.expm1(y_test)
print(r2_score(real_actuals, real_predictions))
"""

# _________________________________________________________________________________________________________
# SHAP SECTION (This generates the "Beeswarm" plot)
"""
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, x_test, show=False)
plt.title("What actually drives Car Price? (SHAP Values)")
plt.tight_layout()
plt.show()
"""

# MODELS:
"""
lr_X = df_encoded.drop('selling_price', axis=1)
lr_y = df_encoded['selling_price']

lr_x_train, lr_x_test, lr_y_train, lr_y_test = train_test_split(lr_X, lr_y, test_size=0.2, random_state=42)
"""

# Using RidgeCV for cross-calidation we get 0.8024434856409493
"""
ridge_cv = RidgeCV().fit(lr_x_train, lr_y_train)
y_pred_ridge = ridge_cv.predict(lr_x_test)
print(r2_score(lr_y_test, y_pred_ridge))
"""

# This Linear Regression Model had score of 0.7991112723554374
"""
lr_model = LinearRegression().fit(lr_x_train, lr_y_train)
print(lr_model.score(lr_x_test, lr_y_test))
"""

# Log transformation of Linear Regression had score of 0.8055361176373788
"""
lr_model_log = LinearRegression().fit(lr_x_train, np.log1p(lr_y_train))
y_pred_log = lr_model_log.predict(lr_x_test)
y_pred_real = np.expm1(y_pred_log)
print(f"Log-Transformed Score: ", r2_score(lr_y_test, y_pred_real))
"""

# This is Random Forest Model had score of 0.7601892154785663
"""
rf_X = df_encoded.drop('selling_price', axis=1)
rf_y = df_encoded['selling_price']

rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(rf_X, rf_y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100).fit(rf_x_train, rf_y_train)
print(rf_model.score(rf_x_test, rf_y_test))
"""

# Original lr model
"""
lr_X = df_encoded.drop('selling_price', axis=1)
lr_y = df_encoded['selling_price']
lr_x_train, lr_x_test, lr_y_train, lr_y_test = train_test_split(lr_X, lr_y, test_size=0.2, random_state=42)

lr_model = LinearRegression().fit(lr_x_train, lr_y_train)
print(lr_model.score(lr_x_test, lr_y_test))

lr_model_log = LinearRegression().fit(lr_x_train, np.log1p(lr_y_train))

y_pred_log = lr_model_log.predict(lr_x_test)
y_pred_real = np.expm1(y_pred_log)
print(f"Log-Transformed Score: ", r2_score(lr_y_test, y_pred_real))
"""

# Visuals of log making the lr better.
"""
residuals = lr_y_test - y_pred_real

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=lr_y_test, y=y_pred_real, alpha=0.6)
plt.plot([lr_y_test.min(), lr_y_test.max()], [lr_y_test.min(), lr_y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted (The tighter the better)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_real, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot (Look for patterns/curves)')

plt.tight_layout()
plt.show()
"""
