import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/listings.csv")

# Exploration
print(df.head())
print(df.info())
print("Missing values in each column:")
print(df.isnull().sum())

# Drop irrelevant columns
columns_to_drop = ['license', 'id', 'host_id', 'name', 'host_name']
df = df.drop(columns=columns_to_drop)
print("Columns after dropping irrelevant ones:")
print(df.columns)

# Fill missing values in reviews_per_month
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Drop last_review column
df = df.drop(columns=['last_review'])
print('Remaining missing values:')
print(df.isnull().sum())

# Fill missing price values with group mean
df['price'] = df.groupby(['neighbourhood_group', 'room_type'])['price'].transform(lambda x: x.fillna(x.mean()))
print("Remaining missing values in 'price':", df['price'].isnull().sum())

# Drop duplicates
print("Number of duplicates: ", df.duplicated().sum())
df = df.drop_duplicates()
print("Number of duplicate rows after dropping: ", df.duplicated().sum())

#Summary Statistics
print("Summary Statistics:")
print(df.describe())

# One-hot encode categorical columns
categorical_cols = ['neighbourhood_group', 'neighbourhood', 'room_type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True).astype(int)


# Remove outliers and zero availability
df = df[df['minimum_nights'] <= 200]
df = df[df['availability_365'] > 0]

# Define price bins and labels
bins = [0, 50, 100, 200, 500, float('inf')]
labels = [0, 1, 2, 3, 4]

# Create a new column 'price_category' based on the 'price' column
df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)


# Save cleaned and encoded data
df.to_csv("data/cleaned_listings.csv", index=False)
