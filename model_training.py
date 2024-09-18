import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Occupation'], drop_first=True)
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Resampling using SMOTE
    smote = SMOTE(sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scaling the features
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Training the model
    model = RandomForestClassifier(class_weight='balanced', random_state=42)

    param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
    }


    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, 
                                       cv=StratifiedKFold(n_splits=5), scoring='f1', 
                                       verbose=1, n_jobs=-1, random_state=42)
    random_search.fit(X_train_resampled, y_train_resampled)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Best Parameters: {random_search.best_params_}")
    print("Accuracy:", best_model.score(X_test, y_test))

    # Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r'C:\Users\DeLL\Desktop\Visual Studio Coding\random_forest_insight_ML_project\data\dataset.csv')  # Update with your data path
    X_train, X_test, y_train, y_test = preprocess_data(df)
    train_and_evaluate_model(X_train, y_train, X_test, y_test)

