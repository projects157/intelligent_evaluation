import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import os
import random
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Retrieval model
class RetrievalModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(RetrievalModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token as sentence representation
        embeddings = outputs.last_hidden_state[:, 0]
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# Dataset for Retrieval
class RetrievalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r') as f:
            data_list = json.load(f)
            for item in data_list:
                src = item["src"]
                positive_knowledge = item["positive_knowledge"]
                negative_knowledge = item["negative_knowledge"]
                
                # For each source text, create a sample with its positive and negative knowledge
                self.samples.append({
                    "src": src,
                    "positive_knowledge": positive_knowledge,
                    "negative_knowledge": negative_knowledge
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        src = item["src"]
        
        # Randomly select one positive knowledge
        pos_knowledge = random.choice(item["positive_knowledge"])
        
        # Randomly select one negative knowledge
        neg_knowledge = random.choice(item["negative_knowledge"])
        
        # Tokenize source text, positive and negative knowledge
        src_encoding = self.tokenizer(src, max_length=self.max_length, padding='max_length', 
                                  truncation=True, return_tensors='pt')
        pos_encoding = self.tokenizer(pos_knowledge, max_length=self.max_length, padding='max_length', 
                                  truncation=True, return_tensors='pt')
        neg_encoding = self.tokenizer(neg_knowledge, max_length=self.max_length, padding='max_length', 
                                  truncation=True, return_tensors='pt')
        
        return {
            'src_input_ids': src_encoding['input_ids'].squeeze(),
            'src_attention_mask': src_encoding['attention_mask'].squeeze(),
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
        }

# InfoNCE loss function
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, src_embeddings, pos_embeddings, neg_embeddings):
        batch_size = src_embeddings.size(0)
        # Compute similarity between source and positive knowledge
        pos_similarity = torch.sum(src_embeddings * pos_embeddings, dim=1) / self.temperature
        # Compute similarity between source and negative knowledge
        neg_similarity = torch.sum(src_embeddings * neg_embeddings, dim=1) / self.temperature
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)
        # Labels are the positions of the positive examples (always 0 in this case)
        labels = torch.zeros(batch_size, dtype=torch.long, device=src_embeddings.device)
        # InfoNCE loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss

# Training function
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Move batch to device
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        pos_input_ids = batch['pos_input_ids'].to(device)
        pos_attention_mask = batch['pos_attention_mask'].to(device)
        neg_input_ids = batch['neg_input_ids'].to(device)
        neg_attention_mask = batch['neg_attention_mask'].to(device)
        
        # Forward pass
        src_embeddings = model(src_input_ids, src_attention_mask)
        pos_embeddings = model(pos_input_ids, pos_attention_mask)
        neg_embeddings = model(neg_input_ids, neg_attention_mask)
        
        # Compute loss
        loss = criterion(src_embeddings, pos_embeddings, neg_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

# Retrieval function
def retrieve(query_text, knowledge_base, model, tokenizer, device, top_k=5, max_length=128):
    """
    Retrieve the most relevant knowledge from the knowledge base for the given query text.
    
    Args:
        query_text (str): The input query text
        knowledge_base (list): List of knowledge texts to retrieve from
        model (RetrievalModel): The trained retrieval model
        tokenizer: The tokenizer for the model
        device: The device to run the model on
        top_k (int): Number of top results to return
        max_length (int): Maximum sequence length for tokenization
        
    Returns:
        list: Top-k most relevant knowledge texts with their similarity scores
    """
    model.eval()
    
    # Tokenize query text
    query_encoding = tokenizer(query_text, max_length=max_length, padding='max_length', 
                              truncation=True, return_tensors='pt')
    query_input_ids = query_encoding['input_ids'].to(device)
    query_attention_mask = query_encoding['attention_mask'].to(device)
    
    # Get query embedding
    with torch.no_grad():
        query_embedding = model(query_input_ids, query_attention_mask)
    
    # Calculate similarity with each knowledge text
    similarities = []
    for knowledge_text in knowledge_base:
        # Tokenize knowledge text
        knowledge_encoding = tokenizer(knowledge_text, max_length=max_length, padding='max_length', 
                                     truncation=True, return_tensors='pt')
        knowledge_input_ids = knowledge_encoding['input_ids'].to(device)
        knowledge_attention_mask = knowledge_encoding['attention_mask'].to(device)
        
        # Get knowledge embedding
        with torch.no_grad():
            knowledge_embedding = model(knowledge_input_ids, knowledge_attention_mask)
        
        # Calculate cosine similarity
        similarity = torch.sum(query_embedding * knowledge_embedding, dim=1).item()
        similarities.append((knowledge_text, similarity))
    
    # Sort by similarity (descending) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Main training loop
def train_loop():
    # Configuration
    model_name = "/home/qhn/Codes/Models/Bert-base-uncased/"
    train_file = "./data/trainRetrieval/trainRetrieval.json"
    output_dir = "./data/trainRetrieval/retrieval_model/"
    batch_size = 16
    num_epochs = 5
    learning_rate = 3e-5
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RetrievalModel(model_name).to(device)
    
    # Create dataset and dataloader
    train_dataset = RetrievalDataset(train_file, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = InfoNCELoss()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    # Save final model
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
def retrieval():
    # Load the trained model for inference
    inference_model = RetrievalModel("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/3_trainRetrieval/retrieval_model").to("cuda")
    inference_tokenizer = AutoTokenizer.from_pretrained("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/3_trainRetrieval/retrieval_model")
    
    # Example knowledge base (could be loaded from a file)
    example_knowledge_base = [
        "7 is a number",
        "Month is related to January",
        "Tesla is a unit of flux density",
        "Colorado is a state",
        "Tesla is a flux density unit",
        "depart is a synonym of depart",
        "Colorado is a state",
        "Colorado is a place",
        "The dollar is related to Americans.",
        "An American is a citizen of America.",
        "The chief is related to Americans.",
        "Yankee-Doodle is an American.",
        "American English is a synonym of American.",
        "American is a synonym of American.",
        "Football is a game",
        "Football is a sport",
        "Pass is related to football",
        "Nogometaški is related to football",
        "Football is a synonym of football",
        "A conference is a type of discussion.",
    ]
    # Example query
    query = "<on 7 january 1900 , tesla left colorado springs .>"
    
    # Retrieve relevant knowledge
    results = retrieve(query, example_knowledge_base, inference_model, inference_tokenizer, "cuda")
    
    print(f"\nQuery: {query}")
    print("Top relevant knowledge:")
    for i, (text, score) in enumerate(results):
        print(f"{i+1}. {text} (Score: {score:.4f})")

if __name__ == "__main__":
    retrieval()
