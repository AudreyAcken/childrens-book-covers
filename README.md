# text-images

This repository includes a fine-tuned T5 model optimized for generating book titles based on book descriptions. 
The model is hosted on Google Drive and can be accessed directly from Google Colab.

It also includes instructions to download a layout generator model and a fine-tuned stable diffusion model to generate children's storybook covers. 

The instrutions below explain how to run the full pipeline on your own.

## Step 1: Download and Unzip the Model

The model `best_model.zip` is available via a shared Google Drive link. Use the following commands in your Colab notebook to download and unzip the model:

```bash
!gdown https://drive.google.com/uc?id=16RJZ6Obc91_Upl4W32cAqHlfe48nVCa- -O best_model.zip
!unzip best_model.zip
```

If gdown is not working, this is the link to download the model: https://drive.google.com/uc?id=16RJZ6Obc91_Upl4W32cAqHlfe48nVCa-

## Step 2: Load the model and tokenizer
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = T5ForConditionalGeneration.from_pretrained('./best_model').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
```


## Step 3: Use this function to generate title

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

## Step 4: Clone the official TextDiffuser Repo and Install Requirements

```bash
git clone https://github.com/microsoft/unilm.git
```

Follow the installation requirements at https://github.com/microsoft/unilm/tree/master/textdiffuser

## Step 5: Download the Fine-tuned Diffusion Model


```
!gdown https://drive.google.com/uc?id=1s9Ss44TWi8etxiCLMUp04vMxSnpcm75L -O diffusion_fine_tune.zip
!unzip diffusion_fine_tune.zip
```

If gdown is not working, this is the link to download the model: https://drive.google.com/uc?id=1s9Ss44TWi8etxiCLMUp04vMxSnpcm75L

Place the model under the ```textdiffuser-ckpt``` folder.

## Step 6: Load the Layout Model


## Step 7: Run the inference code

The following command line prompt will run the inference code. Please replace the prompt with your desired prompt. For the specific application of children's storybook covers, you can try a prompt of the format as "A children's storybook cover with the title [insert title here]". Given a summary of the book, this title may be obtained from the fine-tuned T5 model. 

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-to-image" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --pretrained_model_name_or_path="textdiffuser-ckpt/diffusion_fine_tune"
  --prompt="A sign that says 'Hello'" \
  --output_dir="./output" \
  --vis_num=4
```

## Step 8 (Optional): Explore the Textual Inversion Model
Access the textual inversion inference code ```experiments/textual-inversion-inference.ipynb``` under the experiments folder. You can download the notebook file and modify the prompt. For example, the current prompt is ```prompt = "a children's book cover in the style of <dr-seuss-book-cover>"```. Ensure that you include the learned vocabulary <dr-seuss-book-cover> in your prompt. The current loaded concept is <dr-seuss-book-cover>, but you may also explore with other concepts at https://huggingface.co/sd-concepts-library.

You can also include the title from the fine-tuned T-5 model in the prompt. However, this model has not been combined with the TextDiffuser model yet so the generated image will likely have a misspelled title. 
