
!pip install transformers torch

# Step 1: Prepare the Dataset
def load_dataset(file_path):
    """
    Load the dataset from a text file.
    Each line in the file represents a single training example.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# Loading my custom dataset
dataset = load_dataset('dataset.txt')

# Step 2: Tokenize the Dataset
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize the dataset
tokenized_dataset = [tokenizer.encode(line, return_tensors='pt') for line in dataset]

# Step 3: Prepare DataLoader
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# Convert tokenized dataset to a TensorDataset
input_ids = torch.cat(tokenized_dataset, dim=1).view(-1, len(tokenizer.encode(dataset[0])))
dataset = TensorDataset(input_ids)

# Create a DataLoader
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=1)

# Step 4: Fine-Tune the GPT-2 Model
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * 4  # 4 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
epochs = 4
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        inputs = batch[0].to('cuda')
        model.zero_grad()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

# Save the fine-tuned model and tokenizer
model.save_pretrained('fine-tuned-gpt2')
tokenizer.save_pretrained('fine-tuned-gpt2')

# Step 5: Generate Text with the Fine-Tuned Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('fine-tuned-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('fine-tuned-gpt2')

def generate_text(prompt, max_length=50):
    """
    Generate text using the fine-tuned model based on a given prompt.
    """
    inputs = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Generate text based on a prompt
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
