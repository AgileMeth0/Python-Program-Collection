import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    print("Original Data Shape:", df.shape)
    df = df.drop(columns=['Id'])
    
    # Drop missing values
    df = df.dropna()
    print("After Dropping Missing Values:", df.shape)

    if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("Dropped 'Unnamed: 0' column.")
    
    # Identify categorical features
    categorical_features = ['Student_Age','Sex', 'High_School_Type', 'Scholarship',
                            'Additional_Work', 'Sports_activity', 'Transportation', 'Weekly_Study_Hours',
                            'Attendance','Reading', 'Notes', 'Listening_in_Class', 'Project_work'
                        ]
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}")
    
    # Save label encoders
    model_dir = os.path.join('app', 'model')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    print("Saved label encoders.")

     # Encode the target variable 'grade'
    le_grade = LabelEncoder()
    df['Grade'] = le_grade.fit_transform(df['Grade'])
    joblib.dump(le_grade, os.path.join(model_dir, 'label_encoder_grade.pkl'))
    print("Encoded and saved label encoder for target variable 'grade'.")
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_csv = os.path.join('data', 'students_per.csv')
    output_csv = os.path.join('data', 'preprocessed_students_per.csv')
    preprocess_data(input_csv, output_csv)