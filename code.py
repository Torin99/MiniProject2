print(40*'=')
print("Testing IMDb Dataset")
print(40*'=')

train_data, train_labels = read_train_imdb()
test_data, test_labels = read_test_imdb()

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 5000)
vectorizer.fit(train_data)
processed_train_data =  vectorizer.transform(train_data)
processed_test_data =  vectorizer.transform(test_data)
"""
for estimators in [3,10,50,100,300,500,1000,2000]:
    boost = AdaBoostClassifier(n_estimators = estimators, learning_rate = 0.5)
    model = boost.fit(processed_train_data, train_labels)

    y_pred = model.predict(processed_test_data)

    print("Accuracy with number of estimators", estimators, "is:", metrics.accuracy_score(y_pred, test_labels))

for alpha in [0.1,0.5,0.8,1,1.5,2,3]: #note that from 2 onwards produces the exact same accuracy
    boost = AdaBoostClassifier(n_estimators = 300, learning_rate = alpha)
    model = boost.fit(processed_train_data, train_labels)

    y_pred = model.predict(processed_test_data)

    print("Accuracy with learning rate:", alpha, "is:", metrics.accuracy_score(y_pred, test_labels))
"""
boost = AdaBoostClassifier(n_estimators = 1000, learning_rate = 0.5)
model = boost.fit(processed_train_data, train_labels)

y_pred = model.predict(processed_test_data)

print("Accuracy:", metrics.accuracy_score(y_pred, test_labels))
print("Precision:", metrics.precision_score(y_pred, test_labels))
print("Recall:", metrics.recall_score(y_pred, test_labels))
print("F1 Score:", metrics.f1_score(y_pred, test_labels))

print("\n")
print(40*'=')
print("Testing 20newsgroups Dataset")
print(40*'=')

twenty_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

train_data = twenty_train["data"]
train_labels = twenty_train["target"]

twenty_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

test_data = twenty_test["data"]
test_labels = twenty_test["target"]

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 5000)
processed_train_data = vectorizer.fit_transform(train_data)
processed_test_data = vectorizer.fit_transform(test_data)
"""
for estimators in [3,10,50,100,300,500,1000,2000]:
    boost = AdaBoostClassifier(n_estimators = estimators, learning_rate = 0.8)
    model = boost.fit(processed_train_data, train_labels)

    y_pred = model.predict(processed_test_data)

    print("Accuracy with number of estimators", estimators, "is:", metrics.accuracy_score(y_pred, test_labels))

for alpha in [0.1,0.5,0.8,1,1.5,2,3]: #note that from 2 onwards produces the exact same accuracy
    boost = AdaBoostClassifier(n_estimators = 300, learning_rate = alpha)
    model = boost.fit(processed_train_data, train_labels)

    y_pred = model.predict(processed_test_data)

    print("Accuracy with learning rate:", alpha, "is:", metrics.accuracy_score(y_pred, test_labels))
"""
boost = AdaBoostClassifier(n_estimators = 300, learning_rate = 0.8)
model = boost.fit(processed_train_data, train_labels)
y_pred = model.predict(processed_test_data)

print("Accuracy:", metrics.accuracy_score(y_pred, test_labels))
print("Precision:", metrics.precision_score(y_pred, test_labels, average='weighted'))
print("Recall:", metrics.recall_score(y_pred, test_labels, average='weighted'))
print("F1 Score:", metrics.f1_score(y_pred, test_labels, average='weighted'))
