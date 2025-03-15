import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training and test datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Fill missing values with appropriate statistical measures
# Median for numerical columns, mode for categorical, and 'Unknown' for Cabin
fill_values = {
    "Age": train_df["Age"].median(),
    "Embarked": train_df["Embarked"].mode()[0],
    "Cabin": "Unknown"
}
train_df.fillna(fill_values, inplace=True)
fill_values["Fare"] = test_df["Fare"].median()  # Only in test dataset

test_df.fillna(fill_values, inplace=True)

# Convert 'Sex' column to numerical representation (0: male, 1: female)
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})

# Apply one-hot encoding to 'Embarked' column (drop_first avoids redundancy)
train_df = pd.get_dummies(train_df, columns=["Embarked"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Embarked"], drop_first=True)

# Extract title from 'Name' column using regex
train_df["Title"] = train_df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
test_df["Title"] = test_df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Extract the first letter of 'Cabin' to determine deck; assign 'X' if missing
train_df["Cabin"] = train_df["Cabin"].apply(lambda x: x[0] if x != "Unknown" else "X")
test_df["Cabin"] = test_df["Cabin"].apply(lambda x: x[0] if x != "Unknown" else "X")

# Compute family size as sum of siblings/spouses and parents/children plus oneself
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Drop columns that are not needed for modeling
columns_to_drop = ["PassengerId", "Name", "Ticket"]
train_df.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=columns_to_drop, inplace=True)

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])

# Save the processed data to new CSV files
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
