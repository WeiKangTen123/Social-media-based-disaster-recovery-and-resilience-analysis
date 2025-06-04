import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np

def train_random_forest(df):
    """
    Trains a Random Forest classifier on the given dataset with hyperparameter tuning.
    
    :param df: DataFrame containing 'cleaned_text' (features) and 'label' (target)
    :return: Best trained model, vectorizer, accuracy, classification report, top 10 important features
    """

    if isinstance(df, str):
        df = pd.read_csv(df)
    else:
        df = df


    if "Cleaned_Text" not in df.columns or "Label" not in df.columns:
        raise ValueError("DataFrame must contain 'Cleaned_Text' and 'Label' columns.")

    # Extract features using TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(df["Cleaned_Text"])
    y = df["Label"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Define Random Forest with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Use RandomizedSearchCV for hyperparameter tuning
    clf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Best model after tuning
    best_clf = random_search.best_estimator_

    # Predict on the test set
    y_pred = best_clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # # Get feature importance
    feature_importances = best_clf.feature_importances_
    important_features = sorted(zip(feature_importances, vectorizer.get_feature_names_out()), reverse=True)[:10]

    return best_clf, vectorizer, accuracy, classification_rep, df
