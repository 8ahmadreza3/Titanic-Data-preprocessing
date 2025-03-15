import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Reading the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Filling missing 'Age' and 'Fare' with median values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Filling missing 'Embarked' with mode (most frequent value)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# Filling missing 'Cabin' with 'Unknown'
train_df['Cabin'].fillna('Unknown', inplace=True)
test_df['Cabin'].fillna('Unknown', inplace=True)

# 'Sex' column: male -> 0, female -> 1
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding 'Embarked' column
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')
test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Embarked')

# Aligning columns in test set to match train set
test_df['Embarked_Q'] = 0 if 'Embarked_Q' not in test_df.columns else test_df['Embarked_Q']
test_df['Embarked_S'] = 0 if 'Embarked_S' not in test_df.columns else test_df['Embarked_S']

# Extracting titles from 'Name' column
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Replacing rare titles with 'Rare' and mapping titles to numerical values
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
train_df['Title'] = train_df['Title'].replace(rare_titles, 'Rare').map(title_mapping).fillna(0)
test_df['Title'] = test_df['Title'].replace(rare_titles, 'Rare').map(title_mapping).fillna(0)

# Extracting first letter of 'Cabin' as a new feature
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'X')
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'X')

# Encoding 'Cabin' feature
cabin_mapping = {cabin: idx for idx, cabin in enumerate(train_df['Cabin'].unique())}
train_df['Cabin'] = train_df['Cabin'].map(cabin_mapping)
test_df['Cabin'] = test_df['Cabin'].map(cabin_mapping)

# Creating 'FamilySize' feature
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Dropping 'PassengerId', 'Name', 'Ticket' columns
train_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
test_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Using StandardScaler to standardize 'Age' and 'Fare' columns
scaler = StandardScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])

# Saving the processed datasets to new CSV files
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
