from unsloth import FastLanguageModel, is_bfloat16_supported  # Ensure this import is correct
from transformers import AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

# Main execution block
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer with appropriate settings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto"  # Automatically allocate model parts to GPU/CPU
)

# Prepare the model for inference
model = FastLanguageModel.for_inference(model)

# Set model to evaluation mode for inference
model.eval()

# Define a function to generate fake reviews based on the existing review
def generate_fake_review_batch(reviews, model, tokenizer, device, max_length=250):
    inputs = tokenizer(
        [f"Generate a fake review based on this review: '{review}'" for review in reviews], 
        return_tensors="pt", max_length=max_length, padding="max_length", truncation=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to the specified device

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,  # You can adjust the length of the generated review here
            do_sample=True,  # Enable sampling for diverse outputs
            top_p=0.95,  # Nucleus sampling
            top_k=50,    # Limit the sampling to top 50 candidates
            early_stopping=True
        )

    generated_reviews = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_tokens]
    return generated_reviews

# Load the dataset (adjust to your actual dataset path and structure)
data = pd.read_csv("WomensClothingE-CommerceReviews.csv")  # Make sure your dataset has a 'review' column
data = data.dropna(subset=['Review Text'])  # Drop rows where 'review' is NaN
data =data[:10000]
# Create a new column to store the fake reviews
data["fake_review"] = ""

# Batch size for processing
batch_size = 16  # You can adjust the batch size according to your GPU memory

# Use DataLoader for batch-wise processing
review_list = data['Review Text'].tolist()
data_loader = DataLoader(review_list, batch_size=batch_size, shuffle=False)

# Generate and store fake reviews batch-wise with a progress bar
fake_reviews = []
for batch in tqdm(data_loader, desc="Generating fake reviews"):
    fake_review_batch = generate_fake_review_batch(batch, model, tokenizer, device)
    fake_reviews.extend(fake_review_batch)

# Assign generated fake reviews back to the dataframe
data["fake_review"] = fake_reviews

# Save the updated dataset with fake reviews to a new CSV file
data.to_csv("mistral_fake_reviews.csv", index=False)

print("Dataset saved successfully with fake reviews.")

