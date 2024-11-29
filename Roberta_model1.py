import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import os

# Load the CSV file
df = pd.read_csv('all_amanzon.csv')  # Correct the filename if needed
df = df.dropna()
df = df.reset_index(drop=True)

print(df.head())
print(len(df))

# Split into train, validation, and test sets
train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # Convert to integer explicitly

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define hyperparameters
MAX_LENGTH = 64
BATCH_SIZE = 256
LEARNING_RATE = 2e-5
EPOCHS = 5

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Create DataLoader for train, validation, and test sets
train_dataset = CustomDataset(train_df['Text'], train_df['Label'].astype(int), tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = CustomDataset(val_df['Text'], val_df['Label'].astype(int), tokenizer, MAX_LENGTH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

test_dataset = CustomDataset(test_df['Text'], test_df['Label'].astype(int), tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Fine-tune the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
validation_accuracies = []

for epoch in range(EPOCHS):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation:'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    validation_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{EPOCHS}, Val Accuracy: {val_accuracy:.2%}')
    with open('validation_accuracies.txt', 'a') as f:
        f.write(f'Epoch {epoch + 1}: {val_accuracy:.2%}\n')

# Save the list of validation accuracies to a file
with open('validation_accuracies_list.txt', 'w') as f:
    for epoch, accuracy in enumerate(validation_accuracies, start=1):
        f.write(f'Epoch {epoch}: {accuracy:.2%}\n')

# Save the model and tokenizer
model.save_pretrained('fine_tuned_roberta_all_book')
tokenizer.save_pretrained('fine_tuned_roberta_all_book')

# List of test files
test_files = [
    'restaurant.csv','base.csv', 'coling.csv', 
    'kaggle.csv', 'medicine.csv', 
    'wiki.csv', 'applied.csv'
]

# Directory to save individual test results
results_dir = 'test_results_all_amazon'
os.makedirs(results_dir, exist_ok=True)

# Column mapping for test datasets
column_mapping = {
    'text_': 'Text',
    'Text': 'Text',
}

# Evaluate the model on each test set
for test_file in test_files:
    # Load the test dataset
    current_test_df = pd.read_csv(test_file)
    current_test_df = current_test_df.dropna()
    current_test_df = current_test_df.reset_index(drop=True)  # Reset index to avoid KeyErrors

    # Print the columns and length to verify
    print(f"Columns in {test_file}: {current_test_df.columns}")
    print(f"Number of entries in {test_file}: {len(current_test_df)}")

    # Check for required columns and adjust if necessary
    text_col = None
    for col in column_mapping.keys():
        if col in current_test_df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"Missing 'Text' or 'Label' column in {test_file}. Available columns: {current_test_df.columns}")

    if 'Label' not in current_test_df.columns:
        raise ValueError(f"Missing 'Label' column in {test_file}. Available columns: {current_test_df.columns}")

    # Create DataLoader for the current test set
    current_test_dataset = CustomDataset(current_test_df[text_col], current_test_df['Label'].astype(int), tokenizer, MAX_LENGTH)
    
    # Print the length of the dataset before loading
    print(f"Creating DataLoader for {test_file}, Dataset Size: {len(current_test_dataset)}")
    
    current_test_loader = DataLoader(current_test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Evaluate the model on the current test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(current_test_loader, desc=f'Testing on {test_file}', unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            # Get the predictions
            predictions = torch.argmax(outputs.logits, dim=1)
            predicted_labels.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    avg_test_loss = test_loss / len(current_test_loader)
    test_accuracy = correct / total

    # Calculate additional metrics: precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', pos_label=1)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Save detailed results to a CSV file
    base_filename = os.path.basename(test_file)
    detailed_results = pd.DataFrame({
        'True Labels': true_labels,
        'Predicted Labels': predicted_labels
    })
    detailed_results.to_csv(os.path.join(results_dir, f'detailed_results_{base_filename}'), index=False)

    # Save summary metrics to a text file
    with open(os.path.join(results_dir, 'summary_metrics.txt'), 'a') as f:
        f.write(f'Test ({base_filename}) - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2%}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n')

    print(f'Test ({base_filename}) - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2%}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
