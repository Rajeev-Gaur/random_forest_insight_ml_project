import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)  # Automatically encodes all categorical variables
    
    # Separate features and target variable
    X = df.drop('target', axis=1)  # Assuming 'target' is the name of your target column
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
