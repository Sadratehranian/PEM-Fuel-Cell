# %%
# @@@@@   start with: @@@@@@
# 1.remove extra empty rows and units row.
# 2.change the format of the datset from CSV to xlsx.
# 3.also there are 2 Load column in the dataset, edit the second Load to Loaddensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.spatial.distance import cdist

# %% 
# Load the new dataset
data = pd.read_excel('data3040b.xlsx')

# Feature Engineering
## 1. Voltage Variability Features
# Calculate the standard deviation of each cell voltage across the dataset
for i in range(1, 97):                       # Assuming there are 96 cells
    data[f'Cell{i}_std'] = data[f'Cell{i}'].rolling(window=5).std()

## 2. Voltage Delta Features
# Calculate the difference in voltage between consecutive readings for each cell
for i in range(1, 97):  # Assuming there are 96 cells
    data[f'Cell{i}_delta'] = data[f'Cell{i}'].diff()

## 3. Voltage Anomaly Indicator
# Identify anomalies in voltage (e.g., a large change between consecutive readings)
threshold = 0.05  # Define a threshold for significant voltage change
for i in range(1, 97):  # Assuming there are 96 cells
    data[f'Cell{i}_anomaly'] = data[f'Cell{i}_delta'].apply(lambda x: 1 if abs(x) > threshold else 0)

## 4. Aggregated Voltage Features
# Calculate the mean and standard deviation of all cell voltages at each time point
data['mean_voltage'] = data[[f'Cell{i}' for i in range(1, 97)]].mean(axis=1)
data['std_voltage'] = data[[f'Cell{i}' for i in range(1, 97)]].std(axis=1)

# 5. CVM Board Related Features 
cvm_features = ['CVM_Temp [HeliumFlow]', 'CVM_Error [O2Flow]', 'CVM_Relais [PremixFlow]']
for feature in cvm_features:
    data[f'{feature}_std'] = data[feature].rolling(window=5).std()
    data[f'{feature}_delta'] = data[feature].diff()
# Export the enhanced dataset with new features
###### enhanced_data_path = 'enhanced_dataset.xlsx'  # Define the path for the enhanced dataset
###### data.to_excel(enhanced_data_path, index=False)
# %%
# EDA

## Basic Statistical Summary
print("Basic Statistical Summary:")
print(data.describe())

## Distribution of Key Features
key_features = ['Load', 'StackVolt', 'mean_voltage', 'std_voltage'] + [f'Cell{i}_std' for i in range(94, 97)] + cvm_features
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# %% [Adjusted Correlation Analysis Code]

plt.figure(figsize=(20, 18))  # Increase figure size
sns.set(font_scale=0.7)  # Increase font scale for readability

# List of features to include in the correlation matrix
features = [
    'Load', 'Loaddensity', 'StackVolt', 'AnodeElectrodeVolt', 'CathodeElectrodeVolt',
    'AnodeStoichio', 'CathodeStoichio', 'AirFlow', 'N2Flow', 'H2Flow',
    'TstackIn', 'TstackOut', 'TcathodeScrubber', 'TanodeIn', 'TcathodeIn',
    'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 'CoolantPressureDiff',
    'AnodeBPressure', 'CathodeBPressure', 'Relative', 'mean_voltage', 'std_voltage'
] + [f'Cell{i}_std' for i in range(93, 97)] + cvm_features

# Creating the correlation matrix
corr_matrix = data[features].corr()

# Creating the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
plt.yticks(rotation=0)  # Rotate y labels for better visibility
plt.title('Correlation Matrix')
plt.show()
# %%

## Boxplots for Key Features
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

## Time Series Plots
time_series_features = ['StackVolt', 'mean_voltage', 'std_voltage']
for feature in time_series_features:
    plt.figure(figsize=(15, 5))
    plt.plot(data['Date_Time'], data[feature])
    plt.title(f'Time Series Plot of {feature}')
    plt.xlabel('Date_Time')
    plt.ylabel(feature)
    plt.show()

## Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='StackVolt', y='mean_voltage')
plt.title('Relationship between Stack Voltage and Mean Cell Voltage')
plt.xlabel('Stack Voltage')
plt.ylabel('Mean Cell Voltage')
plt.show()

## Boxplots for CVM Board Features
cvm_features = ['CVM_Temp [HeliumFlow]', 'CVM_Error [O2Flow]', 'CVM_Relais [PremixFlow]']
for feature in cvm_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

# %%
## Anomaly Detection in Cell Voltages
for i in range(1, 3):  
    plt.figure(figsize=(15, 5))
    plt.plot(data['Date_Time'], data[f'Cell{i}'], label='Voltage')
    plt.scatter(data[data[f'Cell{i}_anomaly'] == 1]['Date_Time'], data[data[f'Cell{i}_anomaly'] == 1][f'Cell{i}'], color='red', label='Anomaly')
    plt.title(f'Voltage and Anomalies for Cell{i}')
    plt.xlabel('Date_Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.show()

# %% Anomaly Detection with Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05)
data['anomaly_iso_forest'] = iso_forest.fit_predict(data[[f'Cell{i}' for i in range(1, 97)]])
iso_forest_anomalies = data['anomaly_iso_forest'].value_counts()[-1]  # Count anomalies
plt.figure(figsize=(15, 6))
for i in range(1, 97):
    anomalies = data[data['anomaly_iso_forest'] == -1][f'Cell{i}']
    plt.scatter(anomalies.index, anomalies, color='red', label='Anomaly' if i==1 else "")
plt.plot(data[f'Cell{i}'], label='Voltage', color='blue')
plt.legend()
plt.title("Anomalies detected by Isolation Forest")
plt.xlabel('Index')
plt.ylabel('Voltage')
plt.show()
# %% K-Means Clustering for Anomaly Detection
features_for_clustering = [f'Cell{i}' for i in range(1, 97)]
# Standardize the data before applying K-Means
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features_for_clustering])
# Calculate SSE for different k values
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot SSE vs. k
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()
# %% #####
kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['cluster'] = clusters
# Calculate distance of each point from its cluster centroid
centroids = kmeans.cluster_centers_
distances = cdist(data_scaled, centroids, 'euclidean')
data['distance_from_cluster_center'] = np.min(distances, axis=1)
# Determine a threshold for anomaly detection
threshold = np.percentile(data['distance_from_cluster_center'], 96)
data['anomaly_kmeans'] = data['distance_from_cluster_center'] > threshold

# Visualize the anomalies by K-Means
plt.figure(figsize=(15, 6))
# Plot all data points, color by cluster membership (normal data)
plt.scatter(data.index, data['mean_voltage'], c=data['cluster'], cmap='viridis', alpha=0.5, label='Clustered Data')

# Highlight the anomalies in red
anomalies_kmeans = data[data['anomaly_kmeans']]
plt.scatter(anomalies_kmeans.index, anomalies_kmeans['mean_voltage'], color='red', label='Anomaly')

plt.title("Anomalies detected by K-Means Clustering")
plt.xlabel('Index')
plt.ylabel('Mean Voltage')
plt.legend()
plt.show()

# %%
# One-Class SVM for Anomaly Detection
one_class_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
data['anomaly_svm'] = one_class_svm.fit_predict(data_scaled)
svm_anomalies = (data['anomaly_svm'] == -1).sum()  # Count anomalies

# Visualize the anomalies by One-Class SVM
plt.figure(figsize=(15, 6))
# Normal data points are plotted with blue color
plt.scatter(data.index, data['mean_voltage'], c=np.where(data['anomaly_svm'] == 1, 'blue', 'red'), label='Normal Data')
# Anomalies are plotted with red color
anomalies = data[data['anomaly_svm'] == -1]
plt.scatter(anomalies.index, anomalies['mean_voltage'], color='red', label='Anomaly')
plt.title("Anomalies detected by One-Class SVM")
plt.xlabel('Index')
plt.ylabel('Mean Voltage')
plt.legend()
plt.show()

# %% Assuming anomaly indicators are already calculated in the dataset

# Counting the number of anomalies for each cell
anomaly_counts = {f'Cell{i}': data[data[f'Cell{i}_anomaly'] == 1].shape[0] for i in range(1, 97)}

# Sorting to find top 3 cells with the most anomalies
top_3_cells = sorted(anomaly_counts, key=anomaly_counts.get, reverse=True)[:3]
print("Top 3 cells with the most anomalies:", top_3_cells)

# Determine the time with the most anomalies
# This creates a Series with the sum of anomalies for each time point
time_anomalies = data.set_index('Date_Time').loc[:, [f'Cell{i}_anomaly' for i in range(1, 97)]].sum(axis=1)
time_with_most_anomalies = time_anomalies.idxmax()
print(f"Time with the most anomalies: {time_with_most_anomalies}")

# Plotting voltage graphs for these cells with anomalies highlighted
for cell in top_3_cells:
    plt.figure(figsize=(15, 5))
    plt.plot(data['Date_Time'], data[cell], label=f'Voltage of {cell}')
    anomalies = data[data[f'{cell}_anomaly'] == 1]
    plt.scatter(anomalies['Date_Time'], anomalies[cell], color='red', label='Anomaly')
    plt.title(f'Voltage and Anomalies for {cell}')
    plt.xlabel('Date_Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.show()
# Time-Series Analysis
decomposition = sm.tsa.seasonal_decompose(data['StackVolt'], model='additive', period=30)
fig = decomposition.plot()
fig.set_size_inches(15, 12)
plt.show()
# %%
# Standardizing the data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[[f'Cell{i}' for i in range(1, 97)]])

# Applying DBSCAN
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: The number of samples in a neighborhood for a point to be considered as a core point
dbscan = DBSCAN(eps=0.4, min_samples=10)
clusters = dbscan.fit_predict(data_scaled)

# Adding cluster labels to the dataset
data['cluster'] = clusters

# Identifying outliers (anomalies are points classified in the '-1' cluster)
data['anomaly_dbscan'] = (data['cluster'] == -1).astype(int)

# Counting the number of anomalies
anomaly_count = data['anomaly_dbscan'].sum()
print(f"Number of anomalies detected by DBSCAN: {anomaly_count}")

# Optionally, visualize the anomalies
plt.figure(figsize=(12, 6))
plt.scatter(data.index, data['mean_voltage'], c=data['anomaly_dbscan'], cmap='coolwarm', label='Data Point')
plt.scatter(data[data['anomaly_dbscan'] == 1].index, data[data['anomaly_dbscan'] == 1]['mean_voltage'], color='red', label='Anomaly')
plt.title("Anomalies detected by DBSCAN")
plt.xlabel('Index')
plt.ylabel('Mean Voltage')
plt.legend()
plt.show()

# %% determining eps and min-samples
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Assuming data_scaled is your dataset scaled appropriately
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
# %%
from sklearn.metrics import silhouette_score

# Define a range of possible eps values
eps_values = np.linspace(start=0.1, stop=1.0, num=10)
min_samples_values = range(2, 10)  # or any other range based on your domain knowledge

best_score = -1
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # Exception handling in case DBSCAN finds only one cluster or assigns all points as noise
        try:
            clusters = dbscan.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, clusters)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
        except ValueError:
            continue

print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best silhouette score: {best_score}")


#So
# %%
# Standardizing the data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[[f'Cell{i}' for i in range(1, 97)]])

# Applying DBSCAN
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: The number of samples in a neighborhood for a point to be considered as a core point
dbscan = DBSCAN(eps=0.4, min_samples=9)
clusters = dbscan.fit_predict(data_scaled)

# Adding cluster labels to the dataset
data['cluster'] = clusters

# Identifying outliers (anomalies are points classified in the '-1' cluster)
data['anomaly_dbscan'] = (data['cluster'] == -1).astype(int)

# Counting the number of anomalies
anomaly_count = data['anomaly_dbscan'].sum()
print(f"Number of anomalies detected by DBSCAN: {anomaly_count}")

# Optionally, visualize the anomalies
plt.figure(figsize=(12, 6))
plt.scatter(data.index, data['mean_voltage'], c=data['anomaly_dbscan'], cmap='coolwarm', label='Data Point')
plt.scatter(data[data['anomaly_dbscan'] == 1].index, data[data['anomaly_dbscan'] == 1]['mean_voltage'], color='red', label='Anomaly')
plt.title("Anomalies detected by DBSCAN")
plt.xlabel('Index')
plt.ylabel('Mean Voltage')
plt.legend()
plt.show()

# %%
# Assuming the variable `data_scaled` contains the scaled features for all cells
# and `data` is the original DataFrame with the cells' data
# and DBSCAN has already been applied and the 'cluster' and 'anomaly_dbscan' columns are set

# Extract anomalies for cell number 96 based on DBSCAN results
anomalies_cell_96 = data.loc[data['anomaly_dbscan'] == 1, 'Cell96']

# Visualize the anomalies for cell number 96
plt.figure(figsize=(12, 6))
plt.scatter(data.index, data['Cell96'], label='Cell 96 Voltage', color='blue', alpha=0.6)
plt.scatter(anomalies_cell_96.index, anomalies_cell_96, color='red', label='Anomaly')
plt.title("Anomalies in Cell 96 Detected by DBSCAN")
plt.xlabel('Index')
plt.ylabel('Voltage')
plt.legend()
plt.show()
# %%
