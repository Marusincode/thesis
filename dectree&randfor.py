###DECISION TREE IMPLEMENTATION ON "DELAY"(THE IMPLEMENTATION IS THE SAME FOR ALL DATASETS)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

attackdata = pd.read_csv("C:/Users/User/Desktop/THESIS/Machine learning/Decision tree/Delay tree/delay300merged.csv")

feat = ['AV_x', 'AV_y', 'AV_steer', 'AV_vel', 'AV_yaw', 'npc_x', 'npc_y','Rollout_num']
tar = ['is_attack']

X = attackdata.drop('is_attack', axis=1)
y = attackdata['is_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=1234)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

from matplotlib import pyplot as plt
print("Class Names:", tar)

unique_labels = set(clf.classes_)
print("Unique Labels in Model:", unique_labels)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   filled=True)

results = pd.DataFrame({
    'Real data': y_test,
    'Predictions': y_pred
})

X_test_with_columns = X_test.assign(**{'Real data': y_test, 'Predictions': y_pred})

X_test_with_columns.to_csv('C:/Users/User/Desktop/THESIS/Machine learning/Decision tree/Delay tree/300runsDelaytree.csv', index=False)

###RANDOM FOREST ON DELAY

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

attackdata = pd.read_csv("C:/Users/User/Desktop/THESIS/Machine learning/Random Forest/Delay forest/delay300merged.csv")

X = attackdata.drop('is_attack', axis=1)
y = attackdata['is_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_resampled, y_resampled)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

X_test_with_columns = X_test.assign(**{'Real data': y_test, 'Predictions': y_pred})
X_test_with_columns.to_csv('C:/Users/User/Desktop/THESIS/Machine learning/Random Forest/Delay forest/300runsDelayforest.csv', index=False)
