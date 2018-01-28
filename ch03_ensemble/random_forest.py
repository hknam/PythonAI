
# coding: utf-8

# In[1]:


import argparse
import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[4]:


from utilities import visualize_classifier


# In[5]:


def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Classification data using Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest = 'classifier_type',
    required = True, choices = ['rf', 'erf'], help = '''Type of Classifier to use; can be either 'rf ' or ''erf' ''')
    return parser


# In[6]:


if __name__ == '__main__':
    
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    
    input_file = 'data_random_forests.txt'
    data = np.loadtxt('./data/' + input_file, delimiter = ',')
    X, y = data[:, :-1], data[:, -1]
    
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])
    
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = 's')
    plt.scatter(class_1[:, 0], class_1[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = 'o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s = 75, facecolors = 'white',
                edgecolors = 'black', linewidth = 1, marker = '^')
    plt.title('Input data')
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25, random_state =5)
    
    params = {'n_estimators' : 100, 'max_depth' : 4, 'random_state' : 0}
    
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)
        
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')
    
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')
    
    class_names = 'Class-0', 'Class-1', 'Class-2'
    print('#' * 40)
    print('Classifier performance on training dataset')
    print(classification_report(y_train, classifier.predict(X_train), target_names = class_names))
    print('#' * 40 + '\n')
    
    print('# 40')
    print('Classifier performance on test dataset')
    print(classification_report(y_test, y_test_pred, target_names = class_names))
    print('#' * 40)

