# This project is a classifier for a dataset of breast cancer data using sklearn's dataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# bcd represents the dataset of breast cancers
bcd = load_breast_cancer()


# Train test split the data
training_data, validation_data, training_labels, validation_labels = train_test_split(bcd.data, bcd.target, train_size=0.8, test_size=0.2, random_state = 95)

k_scores = []
k_list = range(1,101)

for i in k_list:
  bcd_knn = KNeighborsClassifier(n_neighbors = i)
  bcd_knn.fit(training_data, training_labels)
  score = bcd_knn.score(validation_data, validation_labels)
  k_scores.append(score)

optimal_score = max(k_list)

print("The score of that is: " + str(optimal_score))

plt.plot(k_list, k_scores)
plt.xlabel('K Values')
plt.ylabel('Accuracies')
plt.show()
