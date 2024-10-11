# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:36:52 2024

@author: kenneyke
"""
#%% Import libraries
from PyQt5.QtWidgets import QApplication, QFileDialog

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from scipy.stats import randint
import seaborn as sns

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
#import graphviz
import matplotlib.pyplot as plt

#%% Function definitions
app = QApplication([])  # Create a PyQt application
def select_file(root_dir, title="Select a file", file_filter="CSV files (*.csv)"):
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle(title)
    file_dialog.setFileMode(QFileDialog.ExistingFile) 
    file_dialog.setNameFilter(file_filter)
    file_dialog.setDirectory(root_dir)  # Set the root directory
    if file_dialog.exec_():
        file_paths = file_dialog.selectedFiles()
        return file_paths[0]
    return None

def calculate_normalized_class_weights(y_train, class_label_weights):
    # Map the predefined weights to the classes found in y_train
    class_weights = {}
    total_weight = 0
    
    # Create a mapping from class labels to weights
    weight_mapping = dict(zip(class_label_weights.iloc[:, 0], class_label_weights.iloc[:, 1]))
    
    # Assign weights based on y_train distribution and the predefined weights
    for class_label in y_train.unique():
        class_weights[class_label] = weight_mapping.get(class_label, 1)
        total_weight += class_weights[class_label]
    
    # Normalize the weights so they sum to 1
    class_weights = {k: v / total_weight for k, v in class_weights.items()}
    
    return class_weights

def flexible_sample_split(csv_path, test_size=0.2, random_state=None, train_random_state=None):
    """
        random_state (int or None): Controls the overall random state for reproducibility.
        train_random_state (int or None): Controls the random state for training data.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Determine random states based on the parameters
    final_test_random_state = random_state
    final_train_random_state = train_random_state if train_random_state is not None else random_state

    # Split the data into training and testing sets
    train_data, Test20 = train_test_split(data, test_size=test_size, random_state=final_test_random_state)

    # Shuffle the training data to introduce randomness
    np.random.seed(final_train_random_state)  # Set the random seed
    np.random.shuffle(train_data.values)

    # Define the proportions for training splits
    train_splits = [0.2, 0.4, 0.6, 0.8]
    training_sets = []

    # Calculate and store each training split
    for split in train_splits:
        subset_size = int(len(train_data) * split/0.8)
        train_subset = train_data.iloc[:subset_size]  # Select the first subset_size rows
        training_sets.append(train_subset)

    # Unpack the list to individual variables
    Train20, Train40, Train60, Train80 = training_sets

    # Return all data splits as separate variables
    return Train20, Train40, Train60, Train80, Test20


def random_sample_split(csv_path, test_size=0.2, random_state=None):
    """
    random_state (int or None): Controls the random state for reproducibility. 
    Use the same integer for the same sample across multiple runs or None for different samples each time.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Calculate the number of test samples
    test_count = int(len(data) * test_size)

    # Randomly sample test data from the original dataset
    test_data = data.sample(n=test_count, random_state=random_state)

    # Drop the test data samples from the original dataset to create the training data
    train_data = data.drop(test_data.index)

    return train_data, test_data

def stratify_TrainTest_Split(csv_path, test_size=0.2, random_state=None):
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Split the data into training and testing sets, stratified by the "Ext_class" column
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['Ext_class'], random_state=random_state)

    return train_data, test_data

# def reclassify_labels(X_train, y_train, X_test, y_test):
#     # Specify the reclassification label
#     reclassification_label = int(input("Enter the reclassification label: "))  # Convert input to integer

#     # Ensure y_train and y_test are pandas Series
#     y_train = pd.Series(y_train)
#     y_test = pd.Series(y_test)

#     unique_labels_train = set(y_train)
#     unique_labels_test = set(y_test)

#     common_labels = unique_labels_train.intersection(unique_labels_test)
#     removed_values_train = unique_labels_train - common_labels
#     removed_values_test = unique_labels_test - common_labels
   
#     print("Reclassified non-common values from y_train:")
#     for label in removed_values_train:
#         count = sum(y_train == label)
#         print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

#     print("\nReclassified non-common values from y_test:")
#     for label in removed_values_test:
#         count = sum(y_test == label)
#         print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

#     # Reclassify labels not present in both y_train and y_test
#     y_train[~y_train.isin(common_labels)] = reclassification_label
#     y_test[~y_test.isin(common_labels)] = reclassification_label
#     return X_train, y_train, X_test, y_test

def reclassify_labels(X_train, y_train, X_test, y_test):
    # Specify the reclassification label
    reclassification_label = int(input("Enter the reclassification label: "))  # Convert input to integer

    # Ensure y_train and y_test are pandas Series
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)
    removed_values_train = unique_labels_train - common_labels
    removed_values_test = unique_labels_test - common_labels

    print("Reclassified non-common values from y_train:")
    for label in removed_values_train:
        count = sum(y_train == label)
        print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

    print("\nReclassified non-common values from y_test:")
    for label in removed_values_test:
        count = sum(y_test == label)
        print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

    # Reclassify labels not present in both y_train and y_test
    y_train = y_train.replace(list(removed_values_train), reclassification_label)
    y_test = y_test.replace(list(removed_values_test), reclassification_label)
    
    # Update common_labels after reclassification
    common_labels = common_labels.union({reclassification_label})  # Ensure new label is included
    removed_values_train = unique_labels_train - common_labels
    removed_values_test = unique_labels_test - common_labels
    
    y_train[y_train.isin(removed_values_train)] = reclassification_label
    y_test[y_test.isin(removed_values_test)] = reclassification_label

    return X_train, y_train, X_test, y_test

def synchronize_labels(X_train, y_train, X_test, y_test, excel_file_path):
    # Ensure labels in y_test are present in y_train, and vice versa
    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)

    # Find and print removed values
    removed_values_train = set(y_train) - common_labels
    removed_values_test = set(y_test) - common_labels
    
    # Load Excel file
    excel_data = pd.read_excel(excel_file_path)
    ext_class_labels_mapping = dict(zip(excel_data['Ext_Class'], excel_data['Labels']))
    
    print("Removed values from y_train:")
    for label in removed_values_train:
        count = sum(y_train == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    print("\nRemoved values from y_test:")
    for label in removed_values_test:
        count = sum(y_test == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    # Remove rows with labels not present in both y_train and y_test
    mask_train = y_train.isin(common_labels)
    mask_test = y_test.isin(common_labels)

    X_train_synchronized = X_train[mask_train]
    y_train_synchronized = y_train[mask_train]

    X_test_synchronized = X_test[mask_test]
    y_test_synchronized = y_test[mask_test]

    return X_train_synchronized, y_train_synchronized, X_test_synchronized, y_test_synchronized

#%% Load data and perform Classification

# # Load CSV data
# data = pd.read_csv(r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\ind_objects_data.csv')

# # Specify the X data & target Data Y
# cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_Class %', 'Total Points', 'Ext_Class', 'Root_Class', 'Sub_Class']   #Columns to exclude from the X data
# X = data.drop(columns=cols_to_remove, axis=1)    
# y = data['Root_Class']

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # 20% for testing. Random state for split reproducibility

# # Create and fit the Random Forest model
# #RF = RandomForestClassifier()      # Uncomment for hyperparameter tunning
# RF = RandomForestClassifier(n_estimators=100, random_state=42)  #Comment for hyperparam tunning
# RF.fit(X_train, y_train)    #Comment for hyperparam tunning

# # Make predictions on the test set
# predictions = RF.predict(X_test)                #Comment for hyperparam tunning

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy}')

# # Additional evaluation metrics
# print('\nClassification Report:')
# print(classification_report(y_test, predictions))

# # Additional evaluation metrics
# print('\nConfusion Matrix:')
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
# disp.plot()
# plt.show()

#%%  Visuallizing the Results

# #Export the first three decision trees from the forest
# for i in range(3):
#     tree = RF.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                 feature_names=X_train.columns,  
#                                 filled=True,  
#                                 max_depth=2, 
#                                 impurity=False, 
#                                 proportion=True)
#     graph = graphviz.Source(dot_data)
    
#     # Save the decision tree as an image file
#     image_path = f'tree_{i+1}.png'
#     graph.render(filename=image_path, format='png', cleanup=True)
    
#     # Open the saved image file
#     Image(filename=image_path)

#%%  Hyperparameter Tuniing
# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}

# # Create a random forest classifier
# RF = RandomForestClassifier()

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(RF, 
#                                   param_distributions = param_dist, 
#                                   n_iter=5, 
#                                   cv=5)

# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)

# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)

# # Generate predictions with the best model
# y_pred = best_rf.predict(X_test)

# # Evaluate the best model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# # Additional evaluation metrics
# print('\nClassification Report:')
# print(classification_report(y_test, y_pred))

# # Additional evaluation metrics
# print('\nConfusion Matrix:')
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
# disp.plot()
# plt.show()

#%% Assign train and text data 

# # Labels file path
labels_file_path = r'D:\ODOT_SPR866\My Label Data Work\Sample Label data for testing\Ext_Class_labels.xlsx'

# Parent file path for datasets
root_dir = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\5_ind_objects'

# # Select the training dataset
# training_file = select_file(root_dir,"Select Training Data")
# print("Selected training dataset:", training_file)

# # Select the testing dataset
# testing_file = select_file(root_dir, "Select Testing Data")
# print("Selected testing dataset:", testing_file)

# # Check selected files selected and load CSV
# if training_file and testing_file:
#     # Load the selected CSV files into DataFrames
#     Train_data = pd.read_csv(training_file)
#     Test_data = pd.read_csv(testing_file)

#     print("Training Data Loaded. Shape:", Train_data.shape)
#     print("Testing Data Loaded. Shape:", Test_data.shape)
# else:
#     print("No file selected. Please select a valid CSV file.")

label_Weight_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\Label_Class_Weights.xlsx'
class_label_weights_path = pd.read_excel(label_Weight_path)

csv_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\3_All_data_Combined\All_data_Combined_TargetFeaturesOnly.csv'


# Train_data, Test_data = stratify_TrainTest_Split(csv_path, test_size=0.2, random_state=42)

# Train_data, Test_data = random_sample_split(csv_path, test_size=0.2, random_state=42)

Train20, Train40, Train60, Train80, Test20 = flexible_sample_split(csv_path, random_state=42, train_random_state=42)

# Training
Train_cols_to_remove = ['Sub_class', 'In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 
                        'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   # Columns to exclude from the X data
X_train = Train80.drop(columns=Train_cols_to_remove, axis=1)  
y_train = Train80['Ext_class']
 
# Testing
Test_cols_to_remove = ['Sub_class','In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 
                       'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   # Columns to exclude from the X data
X_test = Test20.drop(columns=Test_cols_to_remove, axis=1)    
y_test = Test20['Ext_class']

# Synchronize or reclassify labels
print("Select an option:\n1. Reclassify non-common labels\n2. Synchronize labels")
action = input("Enter 1 or 2: ").strip()
if action == '1':
        X_train, y_train, X_test, y_test = reclassify_labels(X_train, y_train, X_test, y_test)
elif action == '2':
    X_train, y_train, X_test, y_test = synchronize_labels(X_train, y_train, X_test, y_test, labels_file_path)
else:
    print("Invalid option entered. No changes made.")

#%% Create and Train the RF Model       
###### Create and fit the Random Forest model

unique_labels = np.unique(np.concatenate([y_train, y_test]))    #Create the unique label class

class_weights = calculate_normalized_class_weights(y_train, class_label_weights_path)
# # RF = RandomForestClassifier(n_estimators=100)
# RF = RandomForestClassifier(n_estimators=100, class_weight=class_weights )  #Comment for hyperparam tunning
# RF.fit(X_train, y_train)    #Comment for hyperparam tunning

#%% ######### Create a best classifier
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
RF = RandomForestClassifier(class_weight=class_weights)

# # Create a StratifiedKFold object with a specific random state
# kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(RF, 
                                  param_distributions = param_dist, 
                                  n_iter=10,
                                  random_state=42,
                                  cv=3)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

#%% Predict and Evaluate

# Make predictions on the test set
predictions = best_rf.predict(X_test)                #Comment for hyperparam tunning

class_weights_list = [class_weights[label] for label in y_test]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions, sample_weight=class_weights_list)
print(f'\n\nAccuracy: {accuracy}')

# # Additional evaluation metrics
# target_names = [str(label) for label in unique_labels]
# print('\nClassification Report:')
# print(classification_report(y_test, predictions, labels=unique_labels, target_names=target_names))

# # Additional evaluation metrics
# print('\nConfusion Matrix:')
# cm = confusion_matrix(y_test, predictions, labels=unique_labels)

# # Plot confusion matrix using Seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()

