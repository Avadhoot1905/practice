Linear Regression:

```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv('Salary_Data.csv')
X = df[['YearsExperience']]
y = df['Salary']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience')
plt.show()
print('Coef:', model.coef_, 'Intercept:', model.intercept_)
```

Logistic regression:

```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv('loan.csv')
X = df[['age']]
y = df['loan']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

Preprocessing

```
import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('pima-indians-diabetes.csv')
X = df.values
# 1. Scaling (MinMax)
print(preprocessing.MinMaxScaler().fit_transform(X)[:3])
# 2. Normalization
print(preprocessing.Normalizer().fit_transform(X)[:3])
# 3. Binarization
print(preprocessing.Binarizer(threshold=0.5).fit_transform(X)[:3])
# 4. Standardization
print(preprocessing.StandardScaler().fit_transform(X)[:3])
```

KNN & SVM

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print('KNN Accuracy:', accuracy_score(y_test, knn.predict(X_test)))
# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print('SVM Accuracy:', accuracy_score(y_test, svm.predict(X_test)))
```

Naive Bayes

```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

Ada Boost

```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X, y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = AdaBoostClassifier(
estimator=DecisionTreeClassifier(max_depth=1),
n_estimators=50, random_state=42
)
model.fit(X_train, y_train)
print('AdaBoost Accuracy:', accuracy_score(y_test, model.predict(X_test)))
```

Kmeans

```
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
# Elbow method
wcss = [KMeans(n_clusters=k,random_state=42).fit(X).inertia_ for k in range(1,8)]
plt.plot(range(1,8), wcss, marker='o')
plt.title('Elbow Method')
plt.show()
# Fit with optimal K=3
km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
s=200, c='red', marker='X')
plt.title('K-Means Clusters')
plt.show()
```