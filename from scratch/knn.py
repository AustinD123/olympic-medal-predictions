import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- KNN Class ---
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, X, Y, k):
        self.k = k
        self.X = X
        self.Y = Y

    def predict(self, X_test):
        return [self._predict_one(x) for x in X_test]

    def _predict_one(self, x):
        distances = [euclidean_distance(x, train_x) for train_x in self.X]
        k_in = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y[i] for i in k_in]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- Main Execution ---
if __name__ == "__main__":
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    # Visualization of petal features
    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Iris Dataset - Petal Features")
    plt.show()

    # Train and predict
    clf = KNN(X_train, y_train, k=5)
    predictions = clf.predict(X_test)

    print("Predictions:", predictions)
    print("True Labels:", list(y_test))

    # Calculate accuracy
    acc = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuracy: {acc:.2f}")
