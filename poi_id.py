#!/usr/bin/python
from __future__ import print_function

# 1. Project Overview
# In this project, you will play detective, and put your machine learning skills to use by building an algorithm
# to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.
import matplotlib

'''Ignore deprecation and future warnings.'''


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


'''Import required modules.'''
import sys  # For system-specific parameters and functions
import pickle  # For object serialization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import confusion_matrix, classification_report

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform data from dictionary to the Pandas DataFrame For data analysis
data_frame = pd.DataFrame.from_dict(data_dict, orient='index')
print('PREVIEW OF FINANCIAL DATA:')
print(data_frame.head(2))
print("")

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
print('LIST OF ALL FEATURES:')
all_features = ['poi - Indicates if person is POI - Starting Feature',
                'salary - Annual salary of person - Starting Feature',
                'to_messages - Number of email that the person has received',
                'deferral_payments - The temporary postponement of a payment',
                'total_payments - Sum of all payments to the person',
                'exercised_stock_options - Dollar value of stocks sold (at a \"strike price\")',
                'bonus -  Monetary payment made to an employee over and above their standard salary',
                'restricted_stock - Shares issued to employees which cannot be fully transferred to them '
                'until certain conditions have been met',
                'shared_receipt_with_poi - Number of email received with known POI/s',
                'restricted_stock_deferred - Restricted stock for which the payment is temporary postponement',
                'total_stock_value - Sum of all stock options',
                'expenses - Dollar value of expenses incurred while employed',
                'loan_advances - Dollar value of loans to person',
                'from_messages - Number of email that the person has sent',
                'other - Dollar value of other payments made to person',
                'from_this_person_to_poi - Number of email from the person to a know POI',
                'director_fees - Dollar value of payments to person for services as a director',
                'deferred_income - Dollar value of payments received in advance for services '
                'which have not yet been performed',
                'long_term_incentive - Dollar value of the incentive owed to person for for staying at the company',
                'email_address - The person\'s email address',
                'from_poi_to_this_person - Number of email from a know POI to the person']
for feature in all_features:
    print("\t", feature)
print("")

# The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'other',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_messages',
                 'to_messages',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'shared_receipt_with_poi']
data_frame = data_frame[features_list]
data_frame = data_frame.replace("NaN", np.nan)
print(data_frame.info())

# Task 2: Remove outliers
# Using the interquartile range (IQR) as a means to find outliers:
quartile1 = data_frame.quantile(0.25)
quartile3 = data_frame.quantile(0.75)
IQR = quartile3 - quartile1
outliers = data_frame[(data_frame > (quartile3 + 1.5 * IQR)) | (data_frame < (quartile1 - 1.5 * IQR))].count(axis=1)

print("")
print("PREVIEW OF THE (interquartile range [IQR]) OUTLIERS")
outliers.sort_values(ascending=False, inplace=True)
print(outliers.head(5))
# Using bonus as the first additional feature, plot the data to identify any obvious outliers
plt.xlabel("salary")
plt.ylabel("bonus")
plt.scatter(data_frame['salary'], data_frame['bonus'])
plt.show()

# Remove the TOTAL column from data_frame as this contains
# values which are not related to a single person
# Removing all of the person with outlying figures between 3 and 9, according to the IQF results.
data_frame.drop(axis=0, labels=["TOTAL", "FREVERT MARK A", "BAXTER JOHN C", "MARTIN AMANDA K", "PICKERING MARK R",
                                "CAUSEY RICHARD A", "WHALLEY LAWRENCE G", "DERRICK JR. JAMES V", "PAI LOU L",
                                "BHATNAGAR SANJAY", "LAVORATO JOHN J", "ALLEN PHILLIP K", "WHITE JR THOMAS E",
                                "LAVORATO JOHN J"], inplace=True)
# # Plot the data again
plt.xlabel("salary")
plt.ylabel("bonus")
plt.scatter(data_frame['salary'], data_frame['bonus'])
plt.show()
print("")

# Task 3: Create new feature(s)

email_features = ['from_messages', 'to_messages', 'from_this_person_to_poi', 'from_poi_to_this_person',
                  'shared_receipt_with_poi']
for feature in email_features:
    print('Max', feature, ': ', data_frame.loc[:, feature].idxmax(), "-",
          data_frame.loc[:, feature].max())
    print('Min', feature, ': ', data_frame.loc[:, feature].idxmin(), "-",
          data_frame.loc[:, feature].min())
# Investigating the email features alone, it is noticeable that the persons in the POI list sent and received much
# fewer email than those who were not in the original POI list. In contrast, the numbers of email to or shared
# with a POI are greater for those in the POI list. Interestingly, the number of email from POIs to other POIs are
# less than half of those than from non-POIs to POIs. From this discovery, we can identify two new features.
data_frame["fraction_poi_email_from_person_total"] = data_frame["from_this_person_to_poi"] \
    .divide(data_frame["from_messages"], fill_value=0)
data_frame["fraction_email_shared_with_poi_total"] = (data_frame["from_poi_to_this_person"]
                                                      + data_frame["shared_receipt_with_poi"]) \
    .divide(data_frame["to_messages"], fill_value=0)
# As these new features did not improve the performance of the classifier, they were excluded
# from the final features_list

# Updating the features_list with the new list
features_list = ['poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'other',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 ]
data_frame = data_frame.fillna(0)

# Store to my_dataset for easy export below.
my_dataset = data_frame.to_dict('index')
print("")
print("New data_dict:\n", my_dataset)

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=50)
selector.fit(features, labels)
features_train = selector.transform(features)
features_test = selector.transform(features)

SelectPercentile_features = zip(selector.get_support(), features_list[1:], selector.scores_)
SelectPercentile_features = sorted(SelectPercentile_features, key=lambda x: x[2], reverse=True)
print ("(Features marked with 'True' are used in the final algorithm.):")
for feature in SelectPercentile_features:
    print(feature)

# Task 4: Try a variety of classifiers
# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script.
clf = GradientBoostingClassifier(init=None,
                                 learning_rate=0.1, max_depth=3,
                                 max_features=None, max_leaf_nodes=None,
                                 min_impurity_split=1e-07, min_samples_leaf=1,
                                 min_samples_split=3, min_weight_fraction_leaf=0.0,
                                 n_estimators=100, presort='auto', random_state=42,
                                 subsample=1.0, verbose=0, warm_start=False)
#  Accuracy: 0.90050 Precision: 0.73509	Recall: 0.47450	F1: 0.57672	F2: 0.51071
# clf = RandomForestClassifier()
# Accuracy: 0.89100	Precision: 0.85586	Recall: 0.28500	F1: 0.42761	F2: 0.32887
# clf = GaussianNB()
#  Accuracy: 0.83536	Precision: 0.43308	Recall: 0.49350	F1: 0.46132	F2: 0.48011

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# Sklearn's f1_score is used as this takes into account both the precision and the recall of the algorithm.
# The upper limited value for f1_score is 1.0 and the lowest is 0.0.
print("")
print("Classification Report:")
print(classification_report(labels_test, pred))

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
