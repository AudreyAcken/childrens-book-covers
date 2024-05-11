from torch import nn
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class BoundingBoxPredictor(nn.Module):
    def __init__(self, max_boxes=10):
        super().__init__()
        self.max_boxes = max_boxes
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, batch_first=True).to(device)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, 512)).to(device)
        self.key_embedding = nn.Embedding(2, 512).to(device)
        self.fc = nn.Linear(512, 8 * max_boxes).to(device)  # Output 8 values per box, for max_boxes boxes

    def forward(self, texts, max_len=50):
        inputs = self.processor(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        text_embeddings = self.clip_model.get_text_features(**inputs).unsqueeze(1)
        seq_len = text_embeddings.size(1)
        
        key_ids = torch.zeros((text_embeddings.size(0), seq_len), dtype=torch.long, device=device)
        key_ids[:, 0] = 1
        key_embeddings = self.key_embedding(key_ids)
        
        pos_embeddings = self.pos_embedding[:, :seq_len]
        
        transformer_input = text_embeddings + key_embeddings + pos_embeddings
        transformer_output = self.transformer(transformer_input, transformer_input)
        bounding_box_predictions = self.fc(transformer_output.squeeze(1))
        
        # Reshape the output to [batch_size, max_boxes, 8]
        bounding_box_predictions = bounding_box_predictions.view(-1, self.max_boxes, 8)
        
        return bounding_box_predictions

def custom_collate_fn(batch):
    texts, bounding_boxes = zip(*batch)
    bounding_boxes_padded = pad_sequence(bounding_boxes, batch_first=True, padding_value=0.0).to(device)
    max_boxes = max(b.shape[0] for b in bounding_boxes)
    if max_boxes < model.max_boxes:
        padding = torch.zeros((len(bounding_boxes), model.max_boxes - max_boxes, 8), device=device)
        bounding_boxes_padded = torch.cat([bounding_boxes_padded, padding], dim=1)
    return texts, bounding_boxes_padded


dataset = BooksDataset("childrens_books_dataset.jsonl")
total_count = len(dataset)
train_count = int(0.9 * total_count)
val_count = total_count - train_count

train_dataset, val_dataset = random_split(dataset, [train_count, val_count])

# Create DataLoaders for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

# Initialize model
model = BoundingBoxPredictor().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, bounding_boxes in dataloader:
            bounding_boxes = bounding_boxes.to(device)
            predictions = model(texts)
            loss = F.mse_loss(predictions, bounding_boxes)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop with validation
for epoch in range(35):
    model.train()
    total_train_loss = 0
    for texts, bounding_boxes in train_dataloader:
        optimizer.zero_grad()
        bounding_boxes = bounding_boxes.to(device)
        predictions = model(texts)
        loss = F.mse_loss(predictions, bounding_boxes)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    val_loss = evaluate(model, val_dataloader)
    
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

## Inference
def predict_bounding_boxes(model, texts):
    processed_texts = processor(texts, return_tensors='pt', padding=True, truncation=True, max_length=50)
    processed_texts = {k: v.to(device) for k, v in processed_texts.items()}
    
    model.eval()
    with torch.no_grad():
        predictions = model(**processed_texts)
    return predictions