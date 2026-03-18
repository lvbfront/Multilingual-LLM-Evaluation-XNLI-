import json
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

def get_majority_class(data_path):
    """Calculate the majority class label from the dataset."""
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            labels.append(entry['label'])  # Extract the label (integer)
    # Find the most frequent label
    label_counts = Counter(labels)
    majority_class = label_counts.most_common(1)[0][0]
    return majority_class, labels

def create_baseline_predictions(majority_class, num_samples):
    """Generate baseline predictions based on the majority class."""
    return [majority_class] * num_samples

if __name__ == "__main__":
    # Path to the dataset
    data_path = r'C:\Users\abdu4\OneDrive\سطح المكتب\year3\NLP\homework\XNLI_Project\scripts\data\xnli_arabic_limited.jsonl'
    
    # Get the majority class and true labels
    majority_class, true_labels = get_majority_class(data_path)
    
    # Generate baseline predictions
    baseline_predictions = create_baseline_predictions(majority_class, len(true_labels))
    
    # Evaluate the baseline
    accuracy = accuracy_score(true_labels, baseline_predictions)
    report = classification_report(true_labels, baseline_predictions, digits=4)
    
    print(f"Majority Class Baseline Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
