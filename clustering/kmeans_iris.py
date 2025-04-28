


### **2. Code Examples**
#### **A. K-Means Clustering (Iris Dataset)**

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Visualize clusters (using first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering (Iris Dataset)")
plt.show()
