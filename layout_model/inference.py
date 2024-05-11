import torch

# Inference function with key and keyword length embeddings
def predict_bounding_boxes(prompts, titles, model, processor, device="cpu"):
    # Process the input text with the CLIP processor
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    # Prepare key and keyword length embeddings
    batch_size = len(prompts)
    max_length = inputs["input_ids"].shape[1]
    
    key_embeddings = torch.zeros((batch_size, max_length), dtype=torch.int64).to(device)
    keyword_lengths = torch.zeros((batch_size, max_length), dtype=torch.int64).to(device)

    for i, (prompt, title) in enumerate(zip(prompts, titles)):
        prompt_tokens = processor.tokenizer.tokenize(prompt)
        title_tokens = processor.tokenizer.tokenize(title)
        
        for j, token in enumerate(prompt_tokens):
            if token in title_tokens:
                key_embeddings[i, j] = 1  # Indicate as keyword
                keyword_lengths[i, j] = len(token)  # Token length as keyword length

    # Run the model's forward pass
    with torch.no_grad():
        predictions = model(prompts, key_embeddings, keyword_lengths).reshape(-1, 8)

    return predictions.cpu().numpy()

# Example usage
checkpoint_path = "model_checkpoint.pth"  # Replace with your trained model path
model = BoundingBoxPredictor().to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
prompts = [
    "A book about adventure titled 'Journey to the West'.",
    "A book about science fiction titled 'Space Odyssey'."
]
titles = ["Journey to the West", "Space Odyssey"]

# Run inference
predictions = predict_bounding_boxes(prompts, titles, model, processor, device)
for prompt, bbox in zip(prompts, predictions):
    print(f"Prompt: {prompt}")
    print(f"Predicted Bounding Box Coordinates: {bbox}")

