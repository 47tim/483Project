import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics       import mean_absolute_error, r2_score

# Timothy Hyd3 2025
# 483 Project, Measuring Energy Efficiency based on building characteristics
# using Heating and Cooling Load values.

# IMPORT CODE FROM UCI
# -----------------------------------------------
# fetch dataset 
energy_efficiency = fetch_ucirepo(id=242) 
  
# data (as pandas dataframes) 
X = energy_efficiency.data.features 
y = energy_efficiency.data.targets 
  
# metadata 
#print(energy_efficiency.metadata) 
  
# variable information 
#print(energy_efficiency.variables) 
# -----------------------------------------------


df = pd.concat([X, y], axis=1)
y.columns = ['Heating_Load', 'Cooling_Load']

y_hl = y['Heating_Load']
y_cl = y['Cooling_Load']

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix – Inputs vs Targets")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

#print(X.columns.tolist())

# Renaming the columns for the graphs
col_map = {
    "X1": "RelCompactness",
    "X2": "SurfaceArea",
    "X3": "WallArea",
    "X4": "RoofArea",
    "X5": "OverallHeight",
    "X6": "Orientation",
    "X7": "GlazingArea",
    "X8": "GlazingAreaDistribution",
}

X = X.rename(columns=col_map)

# one hot encoding
cat_cols = ["Orientation", "GlazingAreaDistribution"]
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)


# Heating
X_train, X_test, y_train_hl, y_test_hl = train_test_split(
    X_encoded, y_hl, test_size=0.2, random_state=42)

# Cooling
_, _, y_train_cl, y_test_cl = train_test_split(
    X_encoded, y_cl, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# Four models

models = {
    "LinearRegression":    LinearRegression(),
    "Ridge(alpha=1.0)":    Ridge(alpha=1.0),
    "RandomForest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting":    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

# ------------------------------------------------------------------------------------

# prints the MAE and R squared values for each of the 4 models to determine the best one

def evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE":   mean_absolute_error(y_test, preds),
            "R2":    r2_score(y_test, preds)
        })
    return pd.DataFrame(results)


print("\nHeating Load: ")
hl_results = evaluate_models(X_train_scaled, X_test_scaled, y_train_hl, y_test_hl)
print(hl_results)

print("\nCooling Load: ")
cl_results = evaluate_models(X_train_scaled, X_test_scaled, y_train_cl, y_test_cl)
print(cl_results)

print("\n\n\n")

# Results of the 4 models
def plot_results(df, title):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["MAE"])
    plt.ylabel("MAE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"mae_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

plot_results(hl_results, "Heating Load")
plot_results(cl_results, "Cooling Load")

# ------------------------------------------------------------------------------------------
# COLORS TO KEEP PIE CHART FEATURE COLORS CONSISTENT BETWEEN THE TWO CHARTS: 

color_map = {
    'RelCompactness': '#1f77b4',
    'SurfaceArea': '#ff7f0e',
    'WallArea': '#2ca02c',
    'RoofArea': '#d62728',
    'OverallHeight': '#9467bd',
    'GlazingArea': '#8c564b'
}

# ------------------------------------------------------------------------------------------
# HEATING LOAD PIE CHART

best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train_hl)

# threshold to remove the unnecesary features cluttering up the pie charts

THRESH = 0.01 

importances = best_model.feature_importances_
features    = X_train.columns

mask = importances >= THRESH
features_filtered    = features[mask]
importances_filtered = importances[mask]

sorted_idx = np.argsort(importances_filtered)[::-1]

sorted_idx = importances_filtered.argsort()[::-1]
labels     = features_filtered[sorted_idx]
sizes      = importances_filtered[sorted_idx]


colors = [color_map.get(label, '#cccccc') for label in labels]

# pie chart plotting
plt.figure(figsize=(8, 8))  
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Heating Load Feature Importance")
plt.axis("equal")  
plt.tight_layout()
plt.savefig("feature_importance_pie.png", dpi=300)
plt.show()


# ------------------------------------------------------------------------------------------
# COOLING LOAD PIE CHART

best_model_cl = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model_cl.fit(X_train_scaled, y_train_cl)

importances_cl = best_model_cl.feature_importances_
features       = X_train.columns                       

# threshold to remove the unnecesary features cluttering up the pie charts

THRESH = 0.01
mask_cl = importances_cl >= THRESH
features_cl_filtered     = features[mask_cl]
importances_cl_filtered  = importances_cl[mask_cl]

sorted_idx_cl  = importances_cl_filtered.argsort()[::-1]
labels_cl      = features_cl_filtered[sorted_idx_cl]
sizes_cl       = importances_cl_filtered[sorted_idx_cl]


colors_cl = [color_map.get(label, '#cccccc') for label in labels_cl]

# pie chart plotting

plt.figure(figsize=(8, 8))
plt.pie(sizes_cl, labels=labels_cl, colors=colors_cl, autopct='%1.1f%%', startangle=140)

plt.title("Cooling Load Feature Importance")
plt.axis("equal")
plt.tight_layout()
plt.savefig("feature_importance_cooling_pie.png", dpi=300)
plt.show()
