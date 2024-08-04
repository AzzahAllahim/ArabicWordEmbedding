#model = Word2Vec.load(model_name)
model = FastText.load_fasttext_format(model_name)
    def average_word_embedding(text, label):
        embeddings = []
        for word in text:
            if word in model.wv.vocab:
                embeddings.append(model.wv[word])
        if embeddings:
            average_embedding = np.mean(embeddings, axis=0)
            return average_embedding, label
        else:
            # Handle case when no embeddings are found
            return None, None
    # Split dataset into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    # Convert training texts to average word embeddings
    train_embeddings = []
    trained_labels = []
    index = 0
    for text in train_texts:
        embedding, final_train_label = average_word_embedding(text, train_labels[index])
        if embedding is not None:
            train_embeddings.append(embedding)
            trained_labels.append(final_train_label)
        index += 1
    # train_embeddings = np.array(train_embeddings).reshape(-1, 1)
    # Train SVM classifier
    svm_classifier = svm.SVC()
    svm_classifier.fit(train_embeddings, trained_labels)
    joblib.dump(svm_classifier, 'svm_classifier_model.pkl')
    # Convert test texts to average word embeddings
    test_embeddings = []
    tested_labels = []
    index = 0
    for text in test_texts:
        embedding, final_test_label = average_word_embedding(text, test_labels[index])
        if embedding is not None:
            test_embeddings.append(embedding)
            tested_labels.append(final_test_label)
        index += 1
    # Predict using the trained SVM classifier
    predictions = svm_classifier.predict(test_embeddings)
    # Calculate accuracy
    accuracy = accuracy_score(tested_labels, predictions)
    print("Accuracy:", accuracy)
