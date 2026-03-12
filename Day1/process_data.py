# %%
import os, sys, re, time, datetime
import pandas as pd
import numpy as np
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# %% uncompress and read the data, and select the file Raw Data.csv from the zip file
def read_data(file_path):
    # uncompress the zip file
#    import zipfile
#    with zipfile.ZipFile(file_path, 'r') as zip_ref:
#        zip_ref.extractall('Data')
    # read the csv file
    df = pd.read_csv(file_path)
    return df
# %% clean the data
def clean_data(df):
    # drop rows with missing values
    df = df.dropna()
    # convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    return df   
# %% Load the data
walk_df = read_data('Data/walk/Raw Data.csv')
jump_df = read_data('Data/jump/Raw Data.csv')

# %% Visualize the data
walk_df.head()

# %%
jump_df.head()

# %% Add a column to indicate the activity type
walk_df['activity'] = 'walk'
jump_df['activity'] = 'jump'
# Concat the two dataframes
movement = pd.concat([walk_df, jump_df], ignore_index=True)

# %% Explore the data
print(movement.shape)
print(movement['activity'].value_counts())
movement.describe()
movement.columns
# %% Plot the scatter plot of the data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(movement['Linear Acceleration x (m/s^2)'], movement['Linear Acceleration y (m/s^2)'], c=movement['activity'].map({'walk': 'blue', 'jump': 'red'}), alpha=0.21)
plt.xlabel('Acceleration X (m/s^2)')    
plt.ylabel('Acceleration Y (m/s^2)')
plt.title('Scatter plot of Acceleration X vs Y')
plt.legend(['Walk', 'Jump'])
plt.show()  
# %% Plot the histogram of the data, change the contours of the histogram to be more visible and add a legend to the histogram
plt.figure(figsize=(10, 6))
plt.hist(movement[movement['activity'] == 'walk']['Linear Acceleration x (m/s^2)'], bins=30, alpha=0.5, label='Walk', color='blue')
plt.hist(movement[movement['activity'] == 'jump']['Linear Acceleration x (m/s^2)'], bins=30, alpha=0.5, label='Jump', color='red')
plt.xlabel('Acceleration X (m/s^2)')
plt.ylabel('Frequency')
plt.title('Histogram of Acceleration X')
plt.legend()
plt.show()

# %% Separate the data into train and test sets
from sklearn.model_selection import train_test_split
X = movement.drop('activity', axis=1)
y = movement['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select random forest classifier to train the model and evaluate the model using accuracy score and classification report using the activity column as the target variable and the rest of the columns as features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
rf = RandomForestClassifier(n_estimators=100,
                            random_state=42,
                            max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# %% Print the feature importance of the model using gini and permutation importance
importances = rf.feature_importances_
feature_names = X.columns   
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df)

# %% Plot the feature importance using a bar plot
plt.figure(figsize=(10, 6)) 
plt.bar(feature_importance_df['feature'], feature_importance_df['importance'], color='blue', alpha=0.7)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.5)
plt.show()


# %% Use a logistic regression model to classify the data and evaluate the model using accuracy score and classification report using the activity column as the target variable and the rest of the columns as features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")
print(classification_report(y_test, y_pred_logreg))

# %% Print the coefficients of the logistic regression model and plot the coefficients using a bar plot
coefficients = logreg.coef_[0]
feature_names = X.columns
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
coef_df = coef_df.sort_values(by='coefficient', ascending=False)
print(coef_df)
plt.figure(figsize=(10, 6))
plt.bar(coef_df['feature'], coef_df['coefficient'], color='blue', alpha=0.7)
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Logistic Regression Coefficients')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.5)
plt.show()

# %% Plot the data using a pairplot to visualize the relationships between the features and the target variable
import seaborn as sns
sns.pairplot(movement, hue='activity', palette={'walk': 'blue', 'jump': 'red'}, diag_kind='kde', plot_kws={'alpha': 0.2})
plt.suptitle('Pairplot of Movement Data', y=1.02)
plt.show()

# %% Plot the data using scatter and the regression line to visualize the relationships between the features and the target variable
sns.lmplot(x='Linear Acceleration x (m/s^2)',
           y='Linear Acceleration y (m/s^2)',
           hue='activity', data=movement,
           palette={'walk': 'blue', 'jump': 'red'},
           scatter_kws={'alpha': 0.2})
plt.title('Scatter plot with Regression Line of Acceleration X vs Y')
plt.xlabel('Acceleration X (m/s^2)')
plt.ylabel('Acceleration Y (m/s^2)')
plt.show()


# %%
print(f"Am I overfitting? {accuracy > accuracy_logreg} (Random Forest accuracy: {accuracy}, Logistic Regression accuracy: {accuracy_logreg})")
# %% Workflow schematic to keep in mind: use the data and specify the model we want to use such as logreg, random forest, nn and we can evaluate then the results and the hyperparameters of the model and then we can extract the features and visualize the data and the results, print a schematic of the workflow using a flowchart
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

nodes = [
    (0.5, 0.90, 'Load Data'),
    (0.5, 0.72, 'Preprocess Data'),
    (0.5, 0.54, 'Train Model\n(Random Forest, Logistic Regression, …)'),
    (0.5, 0.36, 'Evaluate Model'),
    (0.5, 0.18, 'Decide if need to use another method \nor\n continue \nand\n extract features'),
]

fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

box_w, box_h = 0.52, 0.10
for x, y, label in nodes:
    box = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         edgecolor='steelblue', facecolor='#ddeeff')
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, wrap=True)

for i in range(len(nodes) - 1):
    x0, y0, _ = nodes[i]
    x1, y1, _ = nodes[i + 1]
    ax.annotate('', xy=(x1, y1 + box_h / 2), xytext=(x0, y0 - box_h / 2),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8))

plt.title('ML Workflow', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('workflow.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved workflow.png")


# %% Extract features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# %% Plot the PCA components
plt.figure(figsize=(10, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=movement['activity'].map({'walk': 'blue', 'jump': 'red'}), alpha=0.2)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Movement Data')
plt.legend(['Walk', 'Jump'])
plt.show()

# %%
