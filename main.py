import pandas as pd
from data_preprocessing import preprocess_data  # Correct import for data preprocessing
from model_training import train_and_evaluate_model  # Updated import to match the correct filename

def main():
    # Load your dataset
    df = pd.read_csv(r'C:\Users\DeLL\Desktop\Visual Studio Coding\random_forest_insight_ML_project\data\dataset.csv')  # Update the path to your dataset

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()



