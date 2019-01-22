from sklearn import tree
from sklearn import neighbors

# Training Data - [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

# labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

print("How tall are you? (cm)")
height = int(input("> "))
print("How much do you weight? (kg)")
weight = int(input("> "))
print("What is your shoe size? (European)")
shoeSize = int(input("> "))

# Create Decision tree model and train it
classify = tree.DecisionTreeClassifier()
classify = classify.fit(X, Y)
prediction = classify.predict([[height, weight, shoeSize]]) #input 2D array
prediction = str(prediction[0])
decisionScore = round(classify.score(X, Y) * 100)

#Create K Nearest Neightbors model and train it
clf = neighbors.KNeighborsClassifier()
clf = clf.fit(X, Y)
predict = clf.predict([[height, weight, shoeSize]])
predict = str(predict[0])
kScore = round(clf.score(X,Y) * 100)

print(f"The tree model predicted that you are {prediction} with {decisionScore}% accuracy")
print(f"The KNN model predicted that you are {predict} with {kScore}% accuracy")
