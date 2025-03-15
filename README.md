```markdown
# Data Preprocessing for Titanic Dataset

In this project, we perform data preprocessing on the Titanic dataset to prepare it for machine learning modeling. The steps include handling missing values, encoding categorical variables, feature extraction, and data normalization.

## 1. Loading the Data

We start by loading the training and test datasets using pandas.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Reading the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

## 2. Handling Missing Values

We check for missing values and handle them accordingly:

- **Numerical Columns:** Fill missing 'Age' and 'Fare' values with the median of each column.
- **Categorical Columns:** Fill missing 'Embarked' values with the mode (most frequent value). Fill missing 'Cabin' values with 'Unknown'.

```python
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
```

## 3. Encoding Categorical Variables

Convert categorical variables into numerical formats:

- **'Sex' Column:** Map 'male' to 0 and 'female' to 1.
- **'Embarked' Column:** Apply One-Hot Encoding to convert 'Embarked' into numerical values.

```python
# 'Sex' column: male -> 0, female -> 1
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding 'Embarked' column
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')
test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Embarked')

# Aligning columns in test set to match train set
for col in train_df.columns:
    if 'Embarked_' in col and col not in test_df.columns:
        test_df[col] = 0
```

## 4. Feature Extraction

Create new features from existing ones:

- **Extract Titles from 'Name':** Extract titles like Mr, Mrs, Miss, etc., and map rare titles to 'Rare'.
- **Extract First Letter of 'Cabin':** Use the first letter of 'Cabin' as a new feature.
- **Create 'FamilySize' Feature:** Combine 'SibSp' and 'Parch' to create a 'FamilySize' feature.

```python
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
```

## 5. Dropping Unnecessary Columns

Remove columns that are not useful for modeling:

```python
# Dropping 'PassengerId', 'Name', 'Ticket' columns
train_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
test_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
```

## 6. Normalization

Standardize numerical columns to have a mean of 0 and a standard deviation of 1:

```python
# Using StandardScaler to standardize 'Age' and 'Fare' columns
scaler = StandardScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])
```

## 7. Saving the Processed Datasets

Save the processed datasets to new CSV files for further modeling:

```python
# Saving the processed datasets to new CSV files
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
```
