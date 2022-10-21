# This README contains briefly implementation details for Assignment3


************************************************************************************
## 1.) Define functions

### Read data:
- read_file(filename) : reading data from file and storing as a pandas dataframe

### Evaluations: 
- supervised_learning_evaluate(pred, true_label) : evaluating prediction performance for supervised learning model regarding classification_report and accuracy_score
- unsupervised_learning_evaluate(pred, true_label) : evaluating prediction performance for supervised learning model regarding purity of clusters

### Preprocess data:
- pred_transfer(pred) : 
- tfidf_list_to_df(data, feature_names) :
- fs_KBest(X_train, y, X_test, score_func=fs.chi2, k=20) :
- fs_Fpr(X_train, y, X_test, score_func=fs.chi2, alpha=0.01) :
- auto_fs(X_train, y, X_test, y_test, classifier) :

### Machine learning training and predicting: all return predicted sentiment
supervised learning and semi-supervised learning:
- zero_rule_algorithm_classification(train, test)
- logistic(train_X, train_y, test_X)
- bnb_train_predict(train_X, train_y, test_X)
- mnb_train_predict(train_X, train_y, test_X)
- light_gbm(train_X,train_y,test_X,test_y, eval_X)
- ada_boost(train_X, train_y,test_X)
- grad_boost(train_X, train_y,test_X)
- random_forest(train_X, train_y,test_X)
- support_vector_machine(train_X, train_y,test_X)
- self_training(train_x, y, test_x)
- label_spreading(train_x, y, test_x)
unsupervised learning:
- k_means(train_x, test_x)
- mean_shift(train_x, test_x)

************************************************************************************
## 2.) Predicting on raw data

- reading raw data
- processing raw data
- spliting train set and test set
- training models and evaluating performances on full feature set
- selecting subset of features which performs best, then training models and evaluating


************************************************************************************
## 3.) Predicting on tfidf data

- reading tfidf data
- processing tfidf data
- selecting train instances which are similar to instances in test_tfidf.pkl (not used in report)
    See if training the model with instances similar to the test set improves prediction accuracy.
- spliting train set and test set
- training models and evaluating performances on full feature set
- selecting subset of features which performs best, then training models and evaluating


************************************************************************************
## 4.) Predicting on embedding data

- reading embedding data
- processing embedding data
- spliting train set and test set
- training models and evaluating performances on full feature set
- selecting train instances which are similar to instances in test_tfidf.pkl (not used in report)
    See if training the model with instances similar to the test set improves prediction accuracy.
