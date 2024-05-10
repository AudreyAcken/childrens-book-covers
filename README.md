# text-images

This repository includes a fine-tuned T5 model optimized for generating book titles based on book descriptions. 
The model is hosted on Google Drive and can be accessed directly from Google Colab.


## Step 1: Mount Google Drive

First, mount your Google Drive in the Colab notebook to access the model file:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Download and Unzip the Model

The model `best_model.zip` is available via a shared Google Drive link. Use the following commands in your Colab notebook to download and unzip the model:

```bash
!gdown https://drive.google.com/uc?id=16RJZ6Obc91_Upl4W32cAqHlfe48nVCa- -O best_model.zip
!unzip best_model.zip
```

## Step 3: Load the model and tokenizer
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained('./best_model').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
```


## Step 4: Use this function to generate title

```python
# Function to generate title
def generate_title(description, model, tokenizer, device):
    input_ids = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=64, num_beams=5, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage of function to generate a book title
description = "The book is about 11 year old Harry Potter, who receives a letter saying that he is invited to attend Hogwarts, school of witchcraft and wizardry. He then learns that a powerful wizard and his minions are after the sorcerer's stone that will make this evil wizard immortal and undefeatable."
title = generate_title(description, model, tokenizer, device)
print("Generated Book Title:", title)
```
