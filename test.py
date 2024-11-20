




















































# 2nd practical :
# Use the pytesseract library in Python for optical character recognition
# from (i) an image file and (ii) a multi-page PDF file. 


# from local image
import pytesseract
from PIL import Image
image = Image.open('IvV2y.png')
extracted_text=pytesseract.image_to_string(image)
print(extracted_text)


from pdf2image import convert_from_path
import pytesseract
from PIL import Image
pdf_path = '/content/21583473018.pdf'
pages = convert_from_path(pdf_path, 300)
for page_number, page_image in enumerate(pages):
    print(f"Extracting text from page {page_number + 1}...")
    text = pytesseract.image_to_string(page_image)
    print(f"Text from page {page_number + 1}:")
    print(text)
    print()



!apt-get install tesseract-ocr
!pip install pytesseract pillow
# from url fetching the text
import pytesseract
from PIL import Image
import requests
from io import BytesIO
url='https://developer.mozilla.org/en-US/docs/Learn/HTML/Tables/Basics/numbers-table.png'
response1=requests.get(url)
img=Image.open(BytesIO(response1.content))



# Extrating text from multipage file
from pypdf import PdfReader
render=PdfReader('21583473018.pdf')
le1=len(render.pages)
for i in range(le1):
  page=render.pages[i]
  print(page.extract_text())



!pip install pypdf
!pip install tabula-py
!pip install jpype1
import tabula
pdf_path = 'table.pdf'
dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
for i, df in enumerate(dfs):
    print(f"Table {i + 1}")
    print(df)
    print()



!pip install camelot-py[cv]
import camelot
# File path for the uploaded file
file_path = 'table.pdf'
# Extract tables from the PDF
tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
# Print each table
for i, table in enumerate(tables):
    print(f"Table {i + 1}")
    print(table.df)
    print()












# 3rd practical 
# Simple and Multiple Linear Regression using Gradient 
# Descent & Normal Equation Method (without using 
# sklearn or equivalent library 
# for both) 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('housing.csv')

print(df.head())
print(df.columns)

df = pd.read_csv('housing.csv')

missing=df.isnull().sum()

print(missing)

# importing modules and packages

df.fillna(method="ffill",inplace=True)

X = df.iloc[:,0:-2]
y = df.iloc[:,-2]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=2)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

#ploting the graph

plt.scatter(y_test, predictions, color='black')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

plt.show()

#multiple linear regression

df = pd.read_csv('housing.csv')

missing=df.isnull().sum()

print(missing)

df.fillna(method="ffill",inplace=True)

X = df.iloc[:,0:-2]
y = df.iloc[:,-2]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=2)

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

#simple linear ligression

df = pd.read_csv('housing.csv')

missing=df.isnull().sum()

print(missing)

df.fillna(method="ffill",inplace=True)

X = df['median_income'].values.reshape(-1, 1)
y = df['median_house_value']


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=2)

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

#ploting the graph

plt.scatter(y_test, predictions, color='black')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=3)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

plt.show()




# 4th practical :
# Linear Regression with Regularization (without using
# sklearn or equivalent library) and Simple and Multiple
# Linear Regression with and without regularisation using
# Sklearn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge , ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df=pd.read_csv('housing.csv')
missing=df.isnull().sum()
print(missing)


df['total_bedrooms'].fillna(0,inplace=True)

# Gradient descent for simple linear regression

class Linear_Regression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.b = [0, 0]

    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        Y = self.Y
        m = len(Y)
        self.b[0] = self.b[0] - (learning_rate * ((1/m) * np.sum(Y_pred - Y)))
        self.b[1] = self.b[1] - (learning_rate * ((1/m) * np.sum((Y_pred - Y) * self.X)))

    def predict(self, X=None):
        if X is None:  # Better way to check if X is not provided
            X = self.X
        Y_pred = np.array([self.b[0] + self.b[1] * x for x in X])  # Ensure numeric multiplication
        return Y_pred

    def get_current_accuracy(self, Y_pred):
        p, e = Y_pred, self.Y
        n = len(Y_pred)
        return 1 - sum(
            [
                abs(p[i] - e[i]) / e[i]
                for i in range(n)
                if e[i] != 0]
        ) / n

    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1 / (2 * m)) * np.sum((Y_pred - self.Y) ** 2)
        return J

    def plot_best_fit(self, Y_pred, fig):
        f = plt.figure(fig)
        plt.scatter(self.X, self.Y, color='b')
        plt.plot(self.X, Y_pred, color='g')
        plt.title(fig)
        plt.show()
        print()


def main():

    X = df['median_income']
    Y = df['median_house_value']

    regressor = Linear_Regression(X, Y)

    iterations = 0
    steps = 100
    learning_rate = float(input("Enter learning rate: "))
    costs = []


    Y_pred = regressor.predict()
    regressor.plot_best_fit(Y_pred, 'Initial Best Fit Line')

    while iterations != 100:
        Y_pred = regressor.predict()
        cost = regressor.compute_cost(Y_pred)
        costs.append(cost)
        regressor.update_coeffs(learning_rate)
        iterations += 1

    regressor.plot_best_fit(Y_pred, 'Final Best Fit Line')

    # Plot to verify cost function decreases
    plt.figure('Verification')
    plt.plot(range(iterations), costs, color='b')
    plt.title("Cost Function Decrease")
    plt.show()
    print()

if __name__ == '__main__':
    main()


#L1 and L2 regualization using sklerarn

x= df.iloc[:,0:-2]
y = df.iloc[:,-2]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=2)

model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_linear=model.predict(x_test)

for i, col in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col, model.coef_[i]))


print()
print()
print()
#L2 Ridge regression
print("Ridge regualization")
ridge=Ridge(alpha=100)
ridge.fit(x_train,y_train)

for i, col in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col, ridge.coef_[i]))

y_pred_ridge = ridge.predict(x_test)

print()
print()
print()

print("Lasso regualization")
#lasso regualization
lasso=Lasso(alpha=100)
lasso.fit(x_train,y_train)

for i, col in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col, lasso.coef_[i]))

y_pred_lasso=lasso.predict(x_test)

print()
print()
print("Simple linear regression r2 score : " ,(r2_score(y_test,y_pred_linear)))
print("Ridge score : ",r2_score(y_test,y_pred_ridge))
print("lasso score : " ,r2_score(y_test,y_pred_lasso))


# Ridge regualization for simple linear regression

def compute_cost(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * (np.sum((predictions - y) ** 2) + alpha * np.sum(theta[1:] ** 2))
    return cost

def compute_gradient(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    gradient = (1 / m) * (X.T.dot(predictions - y))
    gradient[1:] += (alpha / m) * theta[1:]
    return gradient

def gradient_descent(X, y, theta, alpha, learning_rate, num_iterations):
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta, alpha)
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta, alpha)
    return theta, cost_history

def ridge_regression(X_train, y_train, X_test, y_test, alpha, learning_rate, num_iterations):
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    theta = np.zeros(X_train.shape[1])

    theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, learning_rate, num_iterations)

    y_pred = X_test.dot(theta)

    return theta, y_pred, cost_history



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

alpha = 100
learning_rate = 0.01
num_iterations = 10000

theta, y_pred, cost_history = ridge_regression(x_train_scaled, y_train, x_test_scaled, y_test, alpha, learning_rate, num_iterations)

feature_names = df.columns[:-1]
for i, col in enumerate(feature_names):
    print("The coefficient for {} is {}".format(col, theta[i]))


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Lasso regualization for simple linear regression

def compute_cost_lasso(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * (np.sum((predictions - y) ** 2) + alpha * np.sum(np.abs(theta[1:])))
    return cost

def compute_gradient_lasso(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    gradient = (1 / m) * (X.T.dot(predictions - y))

    gradient[1:] += (alpha / m) * np.sign(theta[1:])

    return gradient

def gradient_descent_lasso(X, y, theta, alpha, learning_rate, num_iterations):
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        gradient = compute_gradient_lasso(X, y, theta, alpha)
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost_lasso(X, y, theta, alpha)
    return theta, cost_history

def lasso_regression(X_train, y_train, X_test, y_test, alpha, learning_rate, num_iterations):
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    theta = np.zeros(X_train.shape[1])

    theta, cost_history = gradient_descent_lasso(X_train, y_train, theta, alpha, learning_rate, num_iterations)

    y_pred = X_test.dot(theta)


    return theta, y_pred, cost_history


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

alpha = 100
learning_rate = 0.01
num_iterations = 100

theta, y_pred, cost_history = lasso_regression(x_train_scaled, y_train, x_test_scaled, y_test, alpha, learning_rate, num_iterations)

feature_names = df.columns[:-1]

for i, col in enumerate(feature_names):
    print("The coefficient for {} is {}".format(col, theta[i]))

print("The intercept term is {}".format(theta[0]))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)




#simple linear regression with L1 and L2 regualization

class LinearRegressionWithRegularization:
    def __init__(self, X, Y, reg_type, alpha=0.1, learning_rate=0.01, iterations=1000):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_type = reg_type
        self.theta = np.zeros(2)

    def predict(self, X):
        return self.theta[0] + self.theta[1] * X

    def compute_cost(self):
        """ Compute cost (MSE + regularization term) """
        m = len(self.Y)
        predictions = self.predict(self.X)
        error = predictions - self.Y
        cost = (1 / (2 * m)) * np.sum(error ** 2)


        if self.reg_type == 'L2':
            cost += (self.alpha / (2 * m)) * np.sum(self.theta[1:] ** 2)
        elif self.reg_type == 'L1':
            cost += (self.alpha / (2 * m)) * np.sum(np.abs(self.theta[1:]))
        return cost

    def update_coeffs(self):

        m = len(self.Y)
        predictions = self.predict(self.X)
        error = predictions - self.Y


        intercept_gradient = (1 / m) * np.sum(error)


        slope_gradient = (1 / m) * np.sum(error * self.X)


        if self.reg_type == 'L2':
            slope_gradient += (self.alpha / m) * self.theta[1]


        elif self.reg_type == 'L1':
            slope_gradient += (self.alpha / m) * np.sign(self.theta[1])


        self.theta[0] -= self.learning_rate * intercept_gradient
        self.theta[1] -= self.learning_rate * slope_gradient

    def fit(self):

        cost_history = []

        for _ in range(self.iterations):
            self.update_coeffs()
            cost_history.append(self.compute_cost())
            print(cost_history[cost_history.size()-1])
        print("Theta 0 : ",self.theta[0])
        print("Theta 1 : ",self.theta[1])
        return cost_history

    def plot_best_fit(self):

        plt.scatter(self.X, self.Y, color='blue', label='Data points')
        plt.plot(self.X, self.predict(self.X), color='red', label='Best fit line')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title(f"Linear Regression with {self.reg_type} Regularization")
        plt.show()


X=df['median_income']
Y=df['median_house_value']

model_ridge = LinearRegressionWithRegularization(X, Y, reg_type='L2', alpha=100, learning_rate=0.1, iterations=10000)
cost_history_ridge = model_ridge.fit()
model_ridge.plot_best_fit()

model_lasso = LinearRegressionWithRegularization(X, Y, reg_type='L1', alpha=100, learning_rate=0.1, iterations=10000)
cost_history_lasso = model_lasso.fit()
model_lasso.plot_best_fit()














# 5th practical :
# Implement Naïve-Bayes models (Multivariate Bernoulli,
# Multinomial and Gaussian using sklearn)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv("Titanic-Dataset.csv")

missing=df.isnull().sum()
print(missing)

# Multivariate Bernoulli naive baye classification
q_em=[]
s_em=[]
for embarked in df["Embarked"]:
  if embarked == "S":
    s_em.append(1)
    q_em.append(0)
  elif embarked == "Q":
    q_em.append(1)
    s_em.append(0)
  else:
    q_em.append(0)
    s_em.append(0)


df2 = pd.DataFrame()
df3 = pd.DataFrame()
df2["S_Embarked"]=s_em
df3["Q_Embarked"]=q_em

df_new = pd.concat([df["Sex"], df2["S_Embarked"], df3["Q_Embarked"], df["Survived"]], axis=1)

print(df_new)

#train and test
x=df_new[['Sex','S_Embarked', 'Q_Embarked']]
y = df_new['Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = BernoulliNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


conf_matrix = confusion_matrix(y_test, y_pred)


precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')


print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Multinomial Naive bayes classifier
# Using Spam or not Spam databaset


dfspam=pd.read_csv('spam_or_not_spam.csv')

dfspam=dfspam.dropna(subset=["email"])


vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dfspam['email']).toarray()
y = dfspam['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)


conf_matrix = confusion_matrix(y_test, y_pred)


precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')


print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:' )
print()
print(conf_matrix)
print()
print(f'Precision: {precision:.2f}')
print()
print(f'Recall: {recall:.2f}')

# Naïve Bayes Classifier- Gaussian
# Rainfall dataset


rainfall=pd.read_csv('Rainfall.csv')
rainfall.dropna(subset=["winddirection","windspeed"],inplace=True)
rain_new=rainfall.drop('day',axis=1)
rain_new.drop('rainfall',axis=1,inplace=True)

x=rain_new
y=rainfall["rainfall"]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:' )
print()
print(conf_matrix)
print()
print(f'Precision: {precision:.2f}')
print()
print(f'Recall: {recall:.2f}')













# 6th practical :
# Implement Decision Trees (ID3, C4.5) using sklearn (2
# Hrs). Compare the parameters such as criterion,
# min_samples_split, min_samples_leaf, max_depth, and
# their effects on accuracy.


!pip install -U scikit-learn
!pip install -U seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load the iris dataset
df = sns.load_dataset('iris')
df.head()

# Check dataset information
df.info()
df.shape
df.isnull().any()

# Plot the data to visualize
sns.pairplot(data=df, hue='species')

# Define features and target
X = df.drop('species', axis=1)
y = LabelEncoder().fit_transform(df['species'])

# Split the dataset (40% test, 60% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

# Create and train the decision tree model
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
print("Decision Tree Classifier Created")

# Predict and evaluate the model
y_pred = dtree.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap='Blues', square=True)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title(f'Accuracy Score: {dtree.score(X_test, y_test):.2f}')
plt.show()

# Plot the decision tree
plt.figure(figsize=(12,12))
plot_tree(dtree, feature_names=X.columns, class_names=['setosa', 'versicolor', 'virginica'], filled=True)
plt.show()


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Features
y = iris.target  # Target

# Split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Decision Tree Classifier with Gini index
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot the Confusion Matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()





# 7th practical : 
# Implement Support Vector Classification and Regression
# with Grid Search for Hyper-parameter tuning using
# sklearn. Plot the hyperplane. (3D plot)

import numpy as np
from sklearn import datasets,metrics,neighbors
from sklearn.model_selection import GridSearchCV,train_test_split as tts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris = datasets.load_iris()
iris1 = pd.DataFrame(iris.data)
iris1.columns = iris.feature_names
iris1['Species'] = iris.target
iris1.head()


x,y=datasets.load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=42)
X=iris.data
y=iris.target
features = iris.feature_names

for i in range(4):
  sns.boxplot(x=y,y=X[:,i])
  plt.ylabel(features[i])
  plt.show()

for i in range(4):
  plt.hist(X[:,i],edgecolor='black')
  plt.title(features[i])
  plt.show()

from sklearn.svm import SVC
kernel = ['linear','rbf','poly','sigmoid']
for i in kernel:
  model=SVC(kernel=i, C=1.0)
  model.fit(X_train,y_train)
  print('Kernel method type:',i)
  print('Accuracy for method type :',i,'is :',model.score(X_test,y_test))

#svm new example 

acc_val=[]
acc_val2=[]
for i in range(1,16):
  model=SVC(kernel='poly', C=100,degree=i)
  model.fit(X_train,y_train)
  print('For Degree',i,'Accuracy on test data is:',model.score(X_test,y_test))
  acc_val.append(model.score(X_test,y_test))
  acc_val2.append(model.score(X_train,y_train))
degree=[i for i in range(1,16)]
plt.plot(degree,acc_val,color='blue')
plt.plot(degree,acc_val2,color='black')
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.legend(['test','train'])
plt.show()

from sklearn import svm
parameters = {'kernel':('linear','poly','rbf','sigmoid'), 'C':[0.5,1,5,10,50]}
svm=svm.SVC()
clf = GridSearchCV(svm, parameters,cv=5)
clf.fit(X_train,y_train)
print(clf.best_params_)
pred=clf.predict(X_test)


print("accuracy : {}% ".format(metrics.accuracy_score(y_test,pred,normalize=True)*100))
print(metrics.confusion_matrix(y_test,pred))

from sklearn.datasets import load_wine
wine = load_wine()
print(wine.data.shape)
print(wine)
print(wine.DESCR)

X = wine.data
y = wine.target

features = wine.feature_names
for i in range(13):
  sns.boxplot(x=y, y=X[:,i])
  plt.ylabel(features[i])
  plt.show()
for i in range(13):
  plt.hist(X[:,i], edgecolor='red')
  plt.title(features[i])
  plt.show()

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=32)
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
for i in kernel:
  model = SVC(kernel=i, C=1.0)
  model.fit(X_train, y_train)
  print('For Kernel : ', i)
  print('Accuracy is : ', model.score(X_test, y_test))
model = SVC()
model.fit(X_train, y_train)
print('Accuracy on test data is : ', model.score(X_test, y_test))
print('Accuracy on train data is : ', model.score(X_train, y_train))

param_grid = {'C':[0.1,1,100,100], 'kernel':['rbf','poly','sigmoid','linear'], 'degree':[1,10]}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))



# or 

import itertools
 import numpy as np
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 from sklearn import datasets
 from sklearn.model_selection import train_test_split
 from sklearn.svm import SVC
 from sklearn.metrics import classification_report, accuracy_score
 data = datasets.load_iris()
 X = data.data
 y = data.target
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)
 param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5, 1.0],
 }
 keys, values = zip(*param_grid.items())
 combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
 results = []
 for params in combinations:
    svm = SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'],
              degree=params['degree'], coef0=params['coef0'], random_state=16)
    
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'C': params['C'],
        'kernel': params['kernel'],
        'gamma': params['gamma'],
 'degree': params['degree'],
})
 deg ee : pa a s[ deg ee ],
 'coef0': params['coef0'],
 'accuracy': accuracy
 df_results = pd.DataFrame(results)
 df_results








# 8th pratical : 
# Implement AND gate using Perceptron Learning (self-
# implementation)

#importing required libraries
import numpy as np
import pandas as pd
import matplotlib as plt

# Define the AND gate data
data = pd.DataFrame({
    'X': [0, 0, 1, 1],
    'Y': [0, 1, 0, 1],
    'X_AND_Y': [0, 0, 0, 1]
})

# Convert DataFrame to numpy array
data = data.values

# Define perceptron parameters
n_datapoints = data.shape[0]
n_dimensions = data.shape[1] - 1
W = 2 * np.random.random_sample((n_dimensions)) - 1
b = np.random.random()
lr = 0.1
n_epochs = 50

# Perceptron learning algorithm
for epoch in range(n_epochs):  # outer loop for epochs
    for i in range(n_datapoints):  # inner loop for each input-output pair
        net_input = np.dot(W, data[i, 0:n_dimensions]) + b  # net_input = W * p + b
        a = net_input >= 0  # apply hardlim activation function
        e = data[i, n_dimensions] - a  # error = target - actual

        # Update weights and bias using the perceptron rule
        W = W + lr * e * data[i, 0:n_dimensions].T
        b = b + lr * e
        print(f"Epoch {epoch}, Data point {i}, Weights: {W}, Bias: {b}")


print(W)
print(b)




#OR GATE 
pred=(np.dot(data[:,0:n_dimensions],W)+b)>=0
print("Prediction: ",pred)
arr=np.array(data[:,2],dtype=bool)
print("Actual Values: ",arr)


import numpy as np
import pandas as pd

# Define the OR gate data
data = pd.DataFrame({
    'X': [0, 0, 1, 1],
    'Y': [0, 1, 0, 1],
    'X_OR_Y': [0, 1, 1, 1]
})

# Convert DataFrame to numpy array
data = data.values

# Define perceptron parameters
n_datapoints = data.shape[0]
n_dimensions = data.shape[1] - 1
W = 2 * np.random.random_sample((n_dimensions)) - 1
b = np.random.random()
lr = 0.1
n_epochs = 50

# Perceptron learning algorithm
for epoch in range(n_epochs):  # outer loop for epochs
    for i in range(n_datapoints):  # inner loop for each input-output pair
        net_input = np.dot(W, data[i, 0:n_dimensions]) + b  # net_input = W * p + b
        a = net_input >= 0  # apply hardlim activation function
        e = data[i, n_dimensions] - a  # error = target - actual

        # Update weights and bias using the perceptron rule
        W = W + lr * e * data[i, 0:n_dimensions].T
        b = b + lr * e
        print(f"Epoch {epoch}, Data point {i}, Weights: {W}, Bias: {b}")

# Final weights and bias
print("Trained Weights:", W)
print("Trained Bias:", b)

# Testing the perceptron on the OR gate inputs
print("\nTesting OR Gate Perceptron:")
for i in range(n_datapoints):
    net_input = np.dot(W, data[i, 0:n_dimensions]) + b
    prediction = net_input >= 0
    print(f"Input: {data[i, 0:n_dimensions]}, Prediction: {int(prediction)}, Actual: {data[i, n_dimensions]}")



# practical -8 

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











# 9th practical : 
# Implement Ex-OR Gate using Backpropagation Neural
# Networks (self-implementation)
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







# 10th practical : 
# Implement Backpropagation Neural Network and K-
# means using sklearn.
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