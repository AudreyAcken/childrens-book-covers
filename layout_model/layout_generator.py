# ------------------------------------------
# Adapted from TextDiffuser: Diffusion Models as Text Painters
# Copyright (c) Microsoft Corporation.
# ------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPTokenizer
from PIL import Image, ImageDraw, ImageFont
from util import get_width, get_key_words, adjust_overlap_box, shrink_box, adjust_font_size, alphabet_dic
from model.layout_transformer import LayoutTransformer, TextConditioner
from termcolor import colored
from inference_layout import *

# import layout transformer
model = LayoutTransformer().cuda().eval()
model.load_state_dict(torch.load('textdiffuser-ckpt/layout_transformer.pth'))

# import text encoder and tokenizer
text_encoder = TextConditioner().cuda().eval()
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')


def process_caption(font_path, caption, keywords):
    # remove punctuations. please remove this statement if you want to paint punctuations
    caption = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", " ", caption) 
    
    # tokenize it into ids and get length
    caption_words = tokenizer([caption], truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    caption_words_ids = caption_words['input_ids'] # (1, 77)
    length = caption_words['length'] # (1, )
    
    # convert id to words
    words = tokenizer.convert_ids_to_tokens(caption_words_ids.view(-1).tolist())
    words = [i.replace('</w>', '') for i in words]
    words_valid = words[:int(length)]

    # store the box coordinates and state of each token
    info_array = np.zeros((77,5)) # (77, 5)

    # split the caption into words and convert them into lower case
    caption_split = caption.split() 
    caption_split = [i.lower() for i in caption_split]

    start_dic = {} # get the start index of each word
    state_list = [] # 0: start, 1: middle, 2: special token
    word_match_list = [] # the index of the word in the caption
    current_caption_index = 0 
    current_match = ''
    for i in range(length): 

        # the first and last token are special tokens
        if i == 0 or i == length-1:
            state_list.append(2) 
            word_match_list.append(127)
            continue

        if current_match == '':
            state_list.append(0)
            start_dic[current_caption_index] = i
        else:
            state_list.append(1)

        current_match += words_valid[i]
        word_match_list.append(current_caption_index)
        if current_match == caption_split[current_caption_index]:
            current_match = ''
            current_caption_index += 1

    while len(state_list) < 77:
        state_list.append(127)
    while len(word_match_list) < 77:
        word_match_list.append(127)

    length_list = []
    width_list =[]
    for i in range(len(word_match_list)):
        if word_match_list[i] == 127:
            length_list.append(0)
            width_list.append(0)
        else:
            length_list.append(len(caption.split()[word_match_list[i]]))
            width_list.append(get_width(font_path, caption.split()[word_match_list[i]]))

    while len(length_list) < 77:
        length_list.append(127)
        width_list.append(0)

    length_list = torch.Tensor(length_list).long() # (77, )
    width_list = torch.Tensor(width_list).long() # (77, )

    boxes = []
    duplicate_dict = {} # some words may appear more than once
    for keyword in keywords: 
        keyword = keyword.lower()
        if keyword in caption_split:
            if keyword not in duplicate_dict:
                duplicate_dict[keyword] = caption_split.index(keyword) 
                index = caption_split.index(keyword)
            else:
                if duplicate_dict[keyword]+1 < len(caption_split) and keyword in caption_split[duplicate_dict[keyword]+1:]:
                    index = duplicate_dict[keyword] + caption_split[duplicate_dict[keyword]+1:].index(keyword)
                    duplicate_dict[keyword] = index
                else:
                    continue
                
            index = caption_split.index(keyword) 
            index = start_dic[index] 
            info_array[index][0] = 1 

            box = [0,0,0,0] 
            boxes.append(list(box))
            info_array[index][1:] = box
    
    boxes_length = len(boxes)
    if boxes_length > 8:
        boxes = boxes[:8]
    while len(boxes) < 8:
        boxes.append([0,0,0,0])

    return caption, length_list, width_list, torch.from_numpy(info_array), words, torch.Tensor(state_list).long(), torch.Tensor(word_match_list).long(), torch.Tensor(boxes), boxes_length


def get_layout_from_prompt(args):

    # prompt = args.prompt
    font_path = args.font_path
    keywords = get_key_words(args.prompt)
    
    print(f'{colored("[!]", "red")} Detected keywords: {keywords} from prompt {args.prompt}')
    
    text_embedding, mask = text_encoder(args.prompt) # (1, 77 768) / (1, 77)

    # process all relevant info
    caption, length_list, width_list, target, words, state_list, word_match_list, boxes, boxes_length = process_caption(font_path, args.prompt, keywords)
    target = target.cuda().unsqueeze(0) # (77, 5)
    width_list = width_list.cuda().unsqueeze(0) # (77, )
    length_list = length_list.cuda().unsqueeze(0) # (77, )
    state_list = state_list.cuda().unsqueeze(0) # (77, )
    word_match_list = word_match_list.cuda().unsqueeze(0) # (77, )

    padding = torch.zeros(1, 1, 4).cuda()
    boxes = boxes.unsqueeze(0).cuda()
    right_shifted_boxes = torch.cat([padding, boxes[:,0:-1,:]],1) # (1, 8, 4)
   
    # inference

    ##we replace the original textdiffuser layout generator code in this part with our bounding box code
    return_boxes= []
    with torch.no_grad():
        box_predictions = predict_bounding_boxes(prompts, titles, model, processor, device)
        for each box in boxes:
            xmin = box[0][0]
            ymin = box[0][1]
            xmax = box[1][0]
            ymax = box[2][1]
            return_boxes.append([xmin, ymin, xmax, ymax])
            
    # print the location of keywords
    print(f'index\tkeyword\tx_min\ty_min\tx_max\ty_max')
    for index, keyword in enumerate(keywords):
        x_min = int(return_boxes[index][0] * 512)
        y_min = int(return_boxes[index][1] * 512)
        x_max = int(return_boxes[index][2] * 512)
        y_max = int(return_boxes[index][3] * 512)
        print(f'{index}\t{keyword}\t{x_min}\t{y_min}\t{x_max}\t{y_max}')
    
    
    # paint the layout
    render_image = Image.new('RGB', (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(render_image)
    segmentation_mask = Image.new("L", (512,512), 0)
    segmentation_mask_draw = ImageDraw.Draw(segmentation_mask)

    for index, box in enumerate(return_boxes):
        box = [int(i*512) for i in box]
        xmin, ymin, xmax, ymax = box
        
        width = xmax - xmin
        height = ymax - ymin
        text = keywords[index]

        font_size = adjust_font_size(args, width, height, draw, text)
        font = ImageFont.truetype(args.font_path, font_size)

        # draw.rectangle([xmin, ymin, xmax,ymax], outline=(255,0,0))
        draw.text((xmin, ymin), text, font=font, fill=(0, 0, 0))
            
        boxes = []
        for i, char in enumerate(text):
            
            # paint character-level segmentation masks
            # https://github.com/python-pillow/Pillow/issues/3921
            bottom_1 = font.getsize(text[i])[1]
            right, bottom_2 = font.getsize(text[:i+1])
            bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
            width, height = font.getmask(char).size
            right += xmin
            bottom += ymin
            top = bottom - height
            left = right - width
            
            char_box = (left, top, right, bottom)
            boxes.append(char_box)
            
            char_index = alphabet_dic[char]
            segmentation_mask_draw.rectangle(shrink_box(char_box, scale_factor = 0.9), fill=char_index)
    
    print(f'{colored("[√]", "green")} Layout is successfully generated')
    return render_image, segmentation_mask
