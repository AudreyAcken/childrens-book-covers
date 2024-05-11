import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import json
import requests
import easyocr
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EasyOCR reader for detecting bounding boxes
reader = easyocr.Reader(['en'])

# Function to fetch and process books
def fetch_and_process_books(limit):
    search_url = f"https://openlibrary.org/subjects/picture_books.json?limit={limit}"
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
                try:
                    cover_image_response = requests.get(cover_url)
                    cover_image = Image.open(BytesIO(cover_image_response.content))

                    # Perform OCR to find text bounding boxes
                    ocr_results = reader.readtext(cover_image)

                    # Find bounding boxes for the title
                    title_boxes = []
                    for result in ocr_results:
                        coordinates, text, confidence = result
                        threshold = 0.5
                        if confidence > threshold:
                            title_boxes.append([[int(coord[0]), int(coord[1])] for coord in coordinates])

                    if len(title_boxes) > 0:
                        dataset_entry = {
                            "title": book_title,
                            "description": description,
                            "bounding_boxes": title_boxes
                        }

                        dataset.append(dataset_entry)
                except:
                    continue

    return dataset

# Fetch and save the dataset
dataset = fetch_and_process_books(limit=1000)
with open("childrens_books_dataset.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Dataset creation complete. Size: {len(dataset)}")