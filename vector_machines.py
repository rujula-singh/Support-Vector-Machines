
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

data=pd.read_csv("./breast-cancer.csv")
#print(data.head(3))
#print(data.info())
#print(data.columns.tolist())
#data_values=data['diagnosis'].unique()
#print(data_values)

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

x=data.drop(['diagnosis'],axis=1).values
y=data['diagnosis'].values

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
x_train_2d = pca.fit_transform(x_train)
x_test_2d = pca.transform(x_test)

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # step size in the mesh
    
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()

# 1. Linear SVM
print("\n=== Linear SVM ===")
linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
linear_svm.fit(x_train_2d, y_train)

# Plot decision boundary for linear SVM
plot_decision_boundary(x_train_2d, y_train, linear_svm, "Linear SVM Decision Boundary")

# Evaluate on test set
y_pred_linear = linear_svm.predict(x_test_2d)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
print("Classification Report:\n", classification_report(y_test, y_pred_linear))

# 2. RBF Kernel SVM
print("\n=== RBF Kernel SVM ===")
rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
rbf_svm.fit(x_train_2d, y_train)

# Plot decision boundary for RBF SVM
plot_decision_boundary(x_train_2d, y_train, rbf_svm, "RBF SVM Decision Boundary")

# Evaluate on test set
y_pred_rbf = rbf_svm.predict(x_test_2d)
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print("Classification Report:\n", classification_report(y_test, y_pred_rbf))

# 3. Hyperparameter Tuning with Grid Search
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate best model on test set
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(x_test)
print("Test set accuracy with best model: {:.2f}".format(accuracy_score(y_test, y_pred_best)))

# 4. Cross-validation evaluation
print("\n=== Cross-Validation Evaluation ===")
cv_scores = cross_val_score(best_svm, x_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy: {:.2f}".format(np.mean(cv_scores)))
print("Standard deviation: {:.2f}".format(np.std(cv_scores)))

# Full-dimensional evaluation (without PCA)
print("\n=== Full-Dimensional Evaluation ===")
full_linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
full_linear_svm.fit(x_train, y_train)
y_pred_full_linear = full_linear_svm.predict(x_test)
print("Full-dimensional Linear SVM Accuracy:", accuracy_score(y_test, y_pred_full_linear))

full_rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
full_rbf_svm.fit(x_train, y_train)
y_pred_full_rbf = full_rbf_svm.predict(x_test)
print("Full-dimensional RBF SVM Accuracy:", accuracy_score(y_test, y_pred_full_rbf))