import bentoml
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X,y = iris.data, iris.target
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

# Save model to the bentoml local model store
saved_model = bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model saved: {saved_model}")

# Model(tag="iris_clf:2pdwhif55smkayay")