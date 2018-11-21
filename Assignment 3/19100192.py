import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print("\n\t---Data Description---\n")
iris=load_iris()
print("Feature Names: ", iris.feature_names)
iris.data
print("Target Names: ", iris.target_names)
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
print("Total no. of rows:        ", X.shape[0])
print("No. of rows for training: ", X_train.shape[0])
print("No. of rows for testing:  ", X_test.shape[0])
print("\n\t---KNeighbours Classifier---\n")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
accuracy = knn.score(X,y)
print("Confusion Matrix: ", conf)
precision_a = conf[0,0]/sum(conf[0,:])
recall_a = conf[0,0]/sum(conf[:,0])
precision_b = conf[1,1]/sum(conf[1,:])
recall_b = conf[1,1]/sum(conf[:,1])
precision_c = conf[2,2]/sum(conf[2,:])
recall_c = conf[2,2]/sum(conf[:,2])
print("Accuracy:          ", accuracy)
print("Precision Class 1: ", precision_a)
print("Precision Class 2: ", precision_b)
print("Precision Class 3: ", precision_c)
print("Recall Class 1:    ", recall_a)
print("Recall Class 2:    ", recall_b)
print("Recall Class 3:    ", recall_c)
print("\n\t---Decision Tree Classifier---\n")
DTC=DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
DTC.fit(X_train,y_train)
y_pred=DTC.predict(X_test)

conf = confusion_matrix(y_test, y_pred)
accuracy = DTC.score(X,y)
precision_a = conf[0,0]/sum(conf[0,:])
recall_a = conf[0,0]/sum(conf[:,0])
precision_b = conf[1,1]/sum(conf[1,:])
recall_b = conf[1,1]/sum(conf[:,1])
precision_c = conf[2,2]/sum(conf[2,:])
recall_c = conf[2,2]/sum(conf[:,2])
print("Confusion: ", conf)
print("Accuracy:          ", accuracy)
print("Precision Class 1: ", precision_a)
print("Precision Class 2: ", precision_b)
print("Precision Class 3: ", precision_c)
print("Recall Class 1:    ", recall_a)
print("Recall Class 2:    ", recall_b)
print("Recall Class 3:    ", recall_c)

print("\n\t---Cross Validation Scores---\n")
scores = cross_val_score(knn, iris.data, iris.target, cv=10)
mean = scores.mean()
scores_knn = scores
scores = list(map(lambda x: int(x*100), scores))
scores = list(map(lambda x: str(x)+"%", scores))
print("KNeighbours Cross-Validation scores: ", scores)
print("KNeighbours Cross-Validation mean: ", str(int(mean*100))+"%")

scores = cross_val_score(DTC, iris.data, iris.target, cv=10)
scores_DTC = scores
mean = scores.mean()
scores = list(map(lambda x: int(x*100), scores))
scores = list(map(lambda x: str(x)+"%", scores))
print("\nDecision Tree Cross-Validation scores: ", scores)
print("Decision Tree Cross-Validation mean: ", str(int(mean*100))+"%")

print("\n\t---Task 3---\n")
print("How can DM Tecniques be applied in a Internet Search Engine Company?")
print("Since the data on the internet is huge and growing at exponential rates, search engine companies try to classify all that data into groups based on different parameters to provide quick retrieval of information.")
print("Google, for example, uses multiple n-ary trees with the root of every tree known as the index.")
print("This allows the query to quickly go through the indeces and find out which tree the information lies in")
print("Similarly, association rule mining can be used to find which other information can be relevant to a query made by a user based on information from previous user queries")

plt.plot(range(0,10), scores_knn)
plt.xlabel("Fold No.")
plt.ylabel("Accuracy")
plt.axis([0, 10, 0.5, 1.1])
plt.title("KNeighbours Accuracy Graph")
plt.show()

plt.plot(range(0,10), scores_DTC)
plt.xlabel("Fold No.")
plt.ylabel("Accuracy")
plt.axis([0, 10, 0.5, 1.1])
plt.title("Decision Tree Accuracy Graph")
plt.show()