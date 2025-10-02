# train_model.py - Run this once to train and save your model
def train_and_save_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from sklearn.datasets import fetch_openml
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    X, y = titanic.data, titanic.target
    
    # Simple preprocessing
    X = X[['pclass', 'sex', 'age', 'sibsp', 'parch']]
    X = pd.get_dummies(X, columns=['sex'], drop_first=True)
    X = X.fillna(X.mean())
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'titanic_model.pkl')
    print("✅ Model trained and saved as 'titanic_model.pkl'")
    
    # Print feature names for reference
    print("Feature names:", list(X.columns))
    return model

if __name__ == '__main__':
    train_and_save_model()