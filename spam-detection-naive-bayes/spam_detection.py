import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def load_data(filepath):
    """Load dataset and keep only required columns"""
    data = pd.read_csv(filepath, encoding='latin-1')

    # Keep only relevant columns
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    return data


def preprocess_data(data):
    """Clean labels and check dataset"""
    # Remove unwanted spaces in labels
    data['label'] = data['label'].str.strip()

    # Convert labels to binary
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Drop missing values if any
    data.dropna(inplace=True)

    return data


def explore_data(data):
    """Basic data exploration"""
    print("Dataset Shape:", data.shape)
    print("\nClass Distribution:\n", data['label'].value_counts())


def split_data(data):
    """Split dataset"""
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'],
        data['label'],
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """Convert text to numerical features using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words='english')

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


def train_model(X_train_vec, y_train):
    """Train Naive Bayes model"""
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_test_vec, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix - Spam Detection")
    plt.savefig("confusion_matrix.png")
    plt.show()


def main():
    data = load_data("spam.csv")
    data = preprocess_data(data)

    explore_data(data)

    X_train, X_test, y_train, y_test = split_data(data)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)


if __name__ == "__main__":
    main()