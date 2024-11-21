















































#Practical 3 : Gradient Descent for Linear Regression


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
X,y = make_regression(n_samples=1000,n_features=1,noise=15,random_state=10)
plt.scatter(X,y)



nrow = X.shape[0]
np.random.seed(10)
all_indices = np.random.permutation(nrow)
print(all_indices[0])


test_size = float(input("Enter the test size (between 0 and 1): "))
if not 0 <= test_size <= 1:
    raise ValueError("Test size must be between 0 and 1")

train_size = math.floor((1 - test_size) * nrow)
X_train = X[all_indices[:train_size]]
y_train = y[all_indices[:train_size]]
X_test = X[all_indices[train_size:]]
y_test = y[all_indices[train_size:]]

X_train.shape,X_test.shape,y_train.shape,y_test.shape

alpha = 0.001
epochs = 2000
t0 = np.random.rand(1)[0]
t1 = np.random.rand(1)[0]


for i in range(epochs):
    y_pred = t0+(t1*X_train)
    y_train=y_train.reshape(-1,1)
    dt0 = (-1/len(X_train))*((np.sum(y_train-y_pred)))
    dt1 = (-1/len(X_train))*((np.sum((y_train-y_pred)*X_train)))
    t0=t0-alpha*dt0
    t1=t1-alpha*dt1
    loss = (1/2*(len(X_train)))*(np.sum((y_train-y_pred)**2))
    if loss<1e-3:
        print("Error is less so breaking the loop")
        break
    if i%100==0:
        print(f"Epoch {i}, Loss: {loss}")
        plt.scatter(X_train,y_train)
        plt.plot(X_train,y_pred,c='r')
        plt.show()
print(f"The thetha 0 is {t0}")
print(f"The thetha 1 is {t1}")



plt.plot(X_train,t0+(t1*X_train),c='r')
plt.scatter(X_train,y_train)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


X, y = make_regression(n_samples=1000, n_features=1, noise=15, random_state=10)


X_intercept = np.c_[np.ones(X.shape[0]), X]


theta = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y


y_pred = X_intercept @ theta


plt.scatter(X, y, label='Actual Values')
plt.plot(X, y_pred, color='red', label='Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Normal Equation')
plt.legend()
plt.show()

print("Intercept (theta_0):", theta[0])
print("Slope (theta_1):", theta[1])



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=10)


X_intercept = np.c_[np.ones(X.shape[0]), X]


m, n = X.shape
theta = np.random.randn(n + 1)
alpha = 0.01
num_iterations = 1000
tolerance = 1e-3
previous_error = float('inf')


for iteration in range(1, num_iterations + 1):

    y_pred = X_intercept @ theta


    gradient = (1/m) * X_intercept.T @ (y_pred - y)


    theta -= alpha * gradient


    error = (1/m) * np.sum((y_pred - y) ** 2)


    if iteration % 100 == 0:
        print(f"Iteration: {iteration}, Error: {error:.6f}")


    if abs(previous_error - error) < tolerance:
        print(f"Convergence reached at iteration {iteration}")
        break

    previous_error = error

print("Final Theta:", theta)




import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=10)


X_intercept = np.c_[np.ones(X.shape[0]), X]


theta = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y


y_pred = X_intercept @ theta

print("Final Theta:", theta)



















# Practical 4 : Linear regression with regularization  without sklearn 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

# Load and prepare data
data = pd.read_csv('Housing DB.csv')
print(data.shape)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X.shape, y.shape)

# Normalize features
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data
nrows = X_norm.shape[0]
print(nrows)
per = np.random.permutation(nrows)
train_rows = math.floor(0.8 * nrows)
X_train = X_norm[per[:train_rows]]
y_train = y[per[:train_rows]]
X_test = X_norm[per[train_rows:]]
y_test = y[per[train_rows:]]
print("Shape of Training Set: ", X_train.shape)
print("Shape of Testing Set: ", X_test.shape)

# Initialize weights and bias
initial_w = np.random.uniform(size=X_train.shape[1])
initial_b = np.random.rand()

def compute_cost(X, y, w, b, lam=0.01, regression_type='simple'):
    m = X.shape[0]
    total_cost = 0
    if regression_type == 'ridge':
        reg = np.sum(np.square(w))  # L2 regularization
    elif regression_type == 'lasso':
        reg = np.sum(np.abs(w))  # L1 regularization
    else:
        reg = 0  # No regularization for simple linear regression
    for i in range(m):
        f_x = np.dot(w, X[i]) + b
        total_cost += (f_x - y[i]) ** 2  # Squared error
    total_cost = (total_cost / (2 * m)) + (lam / (2 * m)) * reg
    return total_cost

def compute_gradient(X, y, w, b, lam=0.01, regression_type='simple'):
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_x = np.dot(w, X[i]) + b
        err = f_x - y[i]
        for j in range(n):
            if regression_type == 'ridge':
                dj_dw[j] += (err * X[i, j]) + lam * w[j]  # Ridge gradient
            elif regression_type == 'lasso':
                dj_dw[j] += (err * X[i, j]) + lam * np.sign(w[j])  # Lasso gradient
            else:
                dj_dw[j] += err * X[i, j]  # Simple linear regression gradient
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha=0.03, iter=1000, lam=0.01, regression_type='simple'):
    J_his = []
    w_his = []
    b_his = []
    w_tmp = copy.deepcopy(w)
    b_tmp = copy.deepcopy(b)
    for i in range(iter + 1):
        cost = compute_cost(X, y, w_tmp, b_tmp, lam, regression_type)
        J_his.append(cost)
        w_his.append(copy.deepcopy(w_tmp))
        b_his.append(b_tmp)
        dj_dw, dj_db = compute_gradient(X, y, w_tmp, b_tmp, lam, regression_type)
        w_tmp = w_tmp - alpha * dj_dw
        b_tmp = b_tmp - alpha * dj_db
        if i % (iter // 10) == 0:
            print(f'Iteration {i}: Cost = {cost}')
    return w_tmp, b_tmp, J_his, w_his, b_his

# Hyperparameters
alpha = 0.03
num_iter = 1000
lambda_param = 0.01

# Perform gradient descent for Simple Linear regression
print("Simple Linear Regression:")
w_simple, b_simple, J_his_simple, w_his_simple, b_his_simple = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, num_iter, 0, 'simple')

# Perform gradient descent for Ridge regression
print("\nRidge Regression:")
w_ridge, b_ridge, J_his_ridge, w_his_ridge, b_his_ridge = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, num_iter, lambda_param, 'ridge')

# Perform gradient descent for Lasso regression
print("\nLasso Regression:")
w_lasso, b_lasso, J_his_lasso, w_his_lasso, b_his_lasso = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, num_iter, lambda_param, 'lasso')

# Predict function
def predict(X, w, b):
    return np.dot(X, w) + b

# Make predictions
y_pred_simple_train = predict(X_train, w_simple, b_simple)
y_pred_simple_test = predict(X_test, w_simple, b_simple)

y_pred_ridge_train = predict(X_train, w_ridge, b_ridge)
y_pred_ridge_test = predict(X_test, w_ridge, b_ridge)

y_pred_lasso_train = predict(X_train, w_lasso, b_lasso)
y_pred_lasso_test = predict(X_test, w_lasso, b_lasso)

# Calculate MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print(f"\nSimple Linear Regression Training MSE: {mse(y_train, y_pred_simple_train)}")
print(f"Simple Linear Regression Testing MSE: {mse(y_test, y_pred_simple_test)}")
print(f"Ridge Training MSE: {mse(y_train, y_pred_ridge_train)}")
print(f"Ridge Testing MSE: {mse(y_test, y_pred_ridge_test)}")
print(f"Lasso Training MSE: {mse(y_train, y_pred_lasso_train)}")
print(f"Lasso Testing MSE: {mse(y_test, y_pred_lasso_test)}")

# Plotting
plt.figure(figsize=(15, 5))

# Simple Linear Regression Plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_simple_test, color='red', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Simple Linear Regression: Actual vs Predicted')

# Ridge Regression Plot
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ridge_test, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Actual vs Predicted')

# Lasso Regression Plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_lasso_test, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted')

plt.tight_layout()
plt.show()








# inbuilt library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv('Housing DB.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize features
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

print("Shape of Training Set: ", X_train.shape)
print("Shape of Testing Set: ", X_test.shape)

# Simple Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear_train = linear_reg.predict(X_train)
y_pred_linear_test = linear_reg.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge_train = ridge.predict(X_train)
y_pred_ridge_test = ridge.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso_train = lasso.predict(X_train)
y_pred_lasso_test = lasso.predict(X_test)

# Plotting
plt.figure(figsize=(15, 5))

# Simple Linear Regression Plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_linear_test, color='red', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Simple Linear Regression: Actual vs Predicted')

# Ridge Regression Plot
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ridge_test, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Actual vs Predicted')

# Lasso Regression Plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_lasso_test, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted')

plt.tight_layout()
plt.show()

# Calculate and print MSE for all methods on training and test sets
print("Simple Linear Regression:")
print(f"Training MSE: {mean_squared_error(y_train, y_pred_linear_train)}")
print(f"Testing MSE: {mean_squared_error(y_test, y_pred_linear_test)}")

print("\nRidge Regression:")
print(f"Training MSE: {mean_squared_error(y_train, y_pred_ridge_train)}")
print(f"Testing MSE: {mean_squared_error(y_test, y_pred_ridge_test)}")

print("\nLasso Regression:")
print(f"Training MSE: {mean_squared_error(y_train, y_pred_lasso_train)}")
print(f"Testing MSE: {mean_squared_error(y_test, y_pred_lasso_test)}")







# Practical 5: Bernoulli Naive Bayes Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris



iris = load_iris()
X_iris = iris.data
y_iris = iris.target




spam_url = 'https://raw.githubusercontent.com/OmkarPathak/Playing-with-datasets/refs/heads/master/Email%20Spam%20Filtering/emails.csv'

spam_df = pd.read_csv(spam_url)


print(spam_df.head())

# 'spam' column has binary labels (1 for spam, 0 for not spam)
spam_df['label'] = spam_df['spam']
spam_df['label'] = spam_df['label'].map({1: 1, 0: 0})

# Vectorizing the text using binary term-document matrix
vectorizer = CountVectorizer(binary=True)
X_spam = vectorizer.fit_transform(spam_df['text']).toarray()
y_spam = spam_df['label']


X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train_iris, y_train_iris)


y_pred_iris_gnb = gnb.predict(X_test_iris)
print("Gaussian Naive Bayes (Iris Dataset) Accuracy:", accuracy_score(y_test_iris, y_pred_iris_gnb))
print(classification_report(y_test_iris, y_pred_iris_gnb))



X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(X_spam, y_spam, test_size=0.3, random_state=42)

# Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train_spam, y_train_spam)


y_pred_spam_mnb = mnb.predict(X_test_spam)
print("Multinomial Naive Bayes (Spam Dataset) Accuracy:", accuracy_score(y_test_spam, y_pred_spam_mnb))
print(classification_report(y_test_spam, y_pred_spam_mnb))


# Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(X_train_spam, y_train_spam)


y_pred_spam_bnb = bnb.predict(X_test_spam)
print("Bernoulli Naive Bayes (Spam Dataset) Accuracy:", accuracy_score(y_test_spam, y_pred_spam_bnb))
print(classification_report(y_test_spam, y_pred_spam_bnb))










#Practical 6 : Decision Trees

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


iris = load_iris()
x,y = iris.data , iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)



clf = DecisionTreeClassifier(criterion = "entropy").fit(x_train, y_train)
plot_tree(clf, filled=True)
accuracy = accuracy_score(clf.predict(x_test),y_test)
print(accuracy)
plt.title("Decision tree trained on all the iris features using entropy")
plt.show()


clf = DecisionTreeClassifier(criterion = "gini").fit(x_train, y_train)
plot_tree(clf, filled=True)
accuracy = accuracy_score(clf.predict(x_test),y_test)
print(accuracy)
plt.title("Decision tree trained on all the iris features using entropy")
plt.show()
















# Practical 7 : Support Vector Machines

# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],        # Regularization parameter
    'gamma': ['scale', 'auto'],     # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': [2, 3, 4]             # Degree of the polynomial kernel function (only relevant for 'poly')
}

# Define the kernels to evaluate
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Dictionary to store accuracy for each kernel
accuracy_dict = {}

# Loop over each kernel
for kernel in kernels:
    param_grid['kernel'] = [kernel]
    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict[kernel] = accuracy
    print(f"Best parameters for kernel '{kernel}': {grid_search.best_params_}")

print("\nAccuracy for each kernel:")
for kernel, accuracy in accuracy_dict.items():
    print(f"Kernel: {kernel}, Accuracy: {accuracy:.2f}")






# Practical 8 : AND Gate , OR Gate using Perceptron Learning

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate random points
np.random.seed(0)
points_pos = np.random.rand(50, 2)  # First quadrant points (positive class)
points_neg = -points_pos             # Third quadrant points (negative class)

# Combine the points and assign labels
X = np.vstack((points_pos, points_neg))
y = np.hstack((np.ones(50), -1 * np.ones(50)))

# Step 2: Implement Perceptron Learning
def perceptron_train(X, y, lr=0.01, epochs=1000):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    # Training loop
    for _ in range(epochs):
        for i, x in enumerate(X):
            if y[i] * (np.dot(x, weights) + bias) <= 0:
                weights += lr * y[i] * x
                bias += lr * y[i]
    return weights, bias

weights, bias = perceptron_train(X, y)

# Step 3: Plot the points and the decision boundary
def plot_decision_boundary(weights, bias):
    # Define a range for x-axis
    x_values = np.linspace(-1, 1, 100)
    # Calculate the corresponding y values
    y_values = -(weights[0] * x_values + bias) / weights[1]
    
    # Plot data points
    plt.scatter(points_pos[:, 0], points_pos[:, 1], color='blue', label='First Quadrant')
    plt.scatter(points_neg[:, 0], points_neg[:, 1], color='red', label='Third Quadrant')
    
    # Plot decision boundary
    plt.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc="lower right")  # Move legend to the bottom right
    plt.title("Perceptron for Linearly Separable Points")
    plt.grid(True)
    plt.show()

plot_decision_boundary(weights, bias)





import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Define the AND gate input data and labels
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Visualize the input data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title("AND Gate Inputs")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

# Initialize weights and bias
n_features = x.shape[1]
w = np.random.uniform(0, 1, size=n_features)
b = np.random.uniform(0, 1)
lr = 0.1

# Train the perceptron
n_epochs = 10  # A few epochs are sufficient for this simple problem
for epoch in range(n_epochs):
    for i in range(x.shape[0]):
        net = np.dot(x[i], w) + b
        if net >= 0:
            output = 1
        else:
            output = 0
        error = y[i] - output
        w += lr * error * x[i]
        b += lr * error

# Decision boundary for visualization
def plot_decision_boundary(X, Y, w, b):
    x1 = np.linspace(-0.5, 1.5, 10)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, 'k-', label="Decision Boundary")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary for AND Gate")
    plt.show()

plot_decision_boundary(x, y, w, b)

# Testing the perceptron on the same data (since it's a small dataset)
predictions = []
for i in range(x.shape[0]):
    net = np.dot(x[i], w) + b
    output = 1 if net >= 0 else 0
    predictions.append(output)

# Evaluate the model
print("Classification Report:")
print(classification_report(y, predictions))
print("Accuracy:", accuracy_score(y, predictions))
print("Predictions:", predictions)


#or gate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Define the AND gate input data and labels
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Visualize the input data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title("OR Gate Inputs")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

# Initialize weights and bias
n_features = x.shape[1]
w = np.random.uniform(0, 1, size=n_features)
b = np.random.uniform(0, 1)
lr = 0.1

# Train the perceptron
n_epochs = 10  # A few epochs are sufficient for this simple problem
for epoch in range(n_epochs):
    for i in range(x.shape[0]):
        net = np.dot(x[i], w) + b
        if net >= 0:
            output = 1
        else:
            output = 0
        error = y[i] - output
        w += lr * error * x[i]
        b += lr * error

# Decision boundary for visualization
def plot_decision_boundary(X, Y, w, b):
    x1 = np.linspace(-0.5, 1.5, 10)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, 'k-', label="Decision Boundary")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary for OR Gate")
    plt.show()

plot_decision_boundary(x, y, w, b)

# Testing the perceptron on the same data (since it's a small dataset)
predictions = []
for i in range(x.shape[0]):
    net = np.dot(x[i], w) + b
    output = 1 if net >= 0 else 0
    predictions.append(output)

# Evaluate the model
print("Classification Report:")
print(classification_report(y, predictions))
print("Accuracy:", accuracy_score(y, predictions))
print("Predictions:", predictions)






# Practical 9 : Implement Ex-OR Gate using Backpropagation Neural Networks

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR labels (reshaped for matrix operations)

# Parameters
input_neurons = X.shape[1]
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# Training process
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Compute error (MSE)
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_output) * learning_rate
    bias_hidden += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the network
print("\nTesting the XOR gate:")
for i in range(len(X)):
    hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)
    print(f"Input: {X[i]}, Predicted Output: {np.round(predicted_output[0])}, Expected Output: {y[i][0]}")







# Practical 10 : Implement Backpropagation Neural Network and K-means using sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids)

# Plot the clustered data
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100, alpha=0.7)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# Print results
print("Cluster centers (in PCA-transformed space):\n", centroids_pca)
print("Inertia (Sum of Squared Distances):", kmeans.inertia_)



import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR output

# Define the neural network model with more neurons and 'tanh' activation
mlp = MLPClassifier(
    hidden_layer_sizes=(8, 8),     # Two hidden layers with 8 neurons each
    activation='tanh',             # Tanh activation function
    solver='lbfgs',                # Stable solver for small datasets
    learning_rate_init=0.001,      # Lower learning rate for stability
    max_iter=10000,
    random_state=43                # Trying a different random seed
)

# Train the model
mlp.fit(X, y)

# Predict and evaluate
y_pred = mlp.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Predictions for XOR problem:", y_pred)
print("Accuracy:", accuracy)