import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import json
import requests
import easyocr
from PIL import Image
from io import BytesIO

# EasyOCR reader for detecting bounding boxes
reader = easyocr.Reader(['en'])

# Function to fetch and process books
def fetch_and_process_books(limit=5):
    search_url = f"https://openlibrary.org/subjects/children.json?limit={limit}"
    response = requests.get(search_url)
    book_data = response.json()
    books = book_data.get("works", [])

    dataset = []

    for book in books:
        book_id = book.get("key", "").split("/")[-1]
        book_title = book.get("title", "No Title")
        
        # Retrieve detailed book data
        book_url = f"https://openlibrary.org/works/{book_id}.json"
        book_info_response = requests.get(book_url)

        if book_info_response.status_code == 200:
            book_info = book_info_response.json()

            description = book_info.get("description", "No Description")
            if isinstance(description, dict):
                description = description.get("value", "No Description")
            
            # Limit description length
            if len(description) > 60:
                description = description[:60]

            cover_id = book_info.get("covers", [None])[0]

            if cover_id:
                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                cover_image_response = requests.get(cover_url)
                cover_image = Image.open(BytesIO(cover_image_response.content))

                # Perform OCR to find text bounding boxes
                ocr_results = reader.readtext(cover_image)

                # Find bounding boxes for the title
                title_boxes = []
                for result in ocr_results:
                    coordinates, text, confidence = result
                    if book_title.lower() in text.lower():
                        title_boxes.append([[int(coord[0]), int(coord[1])] for coord in coordinates])

                if len(title_boxes) > 0:
                    dataset_entry = {
                        "title": book_title,
                        "description": description,
                        "bounding_boxes": title_boxes
                    }

                    dataset.append(dataset_entry)

    return dataset

# Fetch and save the dataset
dataset = fetch_and_process_books(limit=5)
with open("childrens_books_dataset.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print("Dataset creation complete.")

# Custom Dataset class to handle separate title and description
class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        title = entry["title"]
        description = entry["description"]
        bounding_boxes = entry["bounding_boxes"]
        return {"title": title, "description": description, "bounding_boxes": bounding_boxes}

# Custom collate function to include key and keyword length embeddings
def collate_fn(batch):
    titles = [item['title'] for item in batch]
    descriptions = [item['description'] for item in batch]
    prompts = [f"A book about {desc} titled {title}" for desc, title in zip(descriptions, titles)]

    # Create key and keyword length embeddings
    key_embeddings = torch.zeros((len(batch), 77), dtype=torch.float32)  # max_length of 77 tokens
    keyword_lengths = torch.zeros((len(batch), 77), dtype=torch.float32)

    max_boxes = max(len(item['bounding_boxes']) for item in batch)

    padded_boxes = torch.zeros((len(batch), max_boxes, 8), dtype=torch.float32)
    for i, item in enumerate(batch):
        boxes = item['bounding_boxes']
        flattened_boxes = [coord for box in boxes for coord in box]
        box_tensor = torch.tensor(flattened_boxes, dtype=torch.float32).view(-1, 8)
        padded_boxes[i, :box_tensor.shape[0], :] = box_tensor

        # Set the key embedding for the title keyword positions
        tokens = prompts[i].split()
        title_tokens = titles[i].split()
        for j, token in enumerate(tokens):
            if token in title_tokens:
                key_embeddings[i, j] = 1.0
                keyword_lengths[i, j] = len(token)

    return {
        "prompt": prompts,
        "bounding_boxes": padded_boxes,
        "key_embeddings": key_embeddings,
        "keyword_lengths": keyword_lengths
    }

class BoundingBoxPredictor(nn.Module):
    def __init__(self, pretrained_clip_model="openai/clip-vit-base-patch32"):
        super(BoundingBoxPredictor, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(pretrained_clip_model)
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_model)
        
        hidden_size = self.clip_model.text_model.config.hidden_size
        
        # New layers for embeddings
        self.key_embedding_layer = nn.Embedding(2, 512)  # 2 classes (keyword or not)
        self.length_embedding_layer = nn.Embedding(100, 512)  # Arbitrary maximum word length
        self.fc1 = nn.Linear(hidden_size + 1024, 512)  # Include both key and length embeddings
        self.fc2 = nn.Linear(512, 8)

    def forward(self, prompts, key_embeddings, keyword_lengths):
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_embeddings = self.clip_model.get_text_features(**inputs)

        # Convert key and keyword length embeddings into dense layers
        key_emb = self.key_embedding_layer(key_embeddings)
        length_emb = self.length_embedding_layer(keyword_lengths)

        # Concatenate all features
        combined_embeddings = torch.cat([text_embeddings, key_emb, length_emb], dim=-1)

        x = torch.relu(self.fc1(combined_embeddings))
        bounding_box_predictions = self.fc2(x)
        return bounding_box_predictions

# Training Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoundingBoxPredictor().to(device)
dataset = CustomDataset("childrens_books_dataset.jsonl")
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in data_loader:
        prompts = batch["prompt"]
        bounding_boxes = batch["bounding_boxes"].to(device)
        key_embeddings = batch["key_embeddings"].to(device)
        keyword_lengths = batch["keyword_lengths"].to(device)

        optimizer.zero_grad()
        outputs = model(prompts, key_embeddings, keyword_lengths).reshape(-1, 8)
        loss = criterion(outputs, bounding_boxes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}")

print("Training Complete")


