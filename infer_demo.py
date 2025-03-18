import os
import json
import torch
import argparse
import transformers
from PIL import Image

processor = transformers.AutoProcessor.from_pretrained("./layoutlmv3-large", apply_ocr=False)
spatial_position_id = 150000
img_patch_id = 150001

# model args
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='model directory')
parser.add_argument('--img_dir', type=str, help='image directory')
parser.add_argument('--ocr_dir', type=str, help='ocr directory')
parser.add_argument('--instruction', type=str, help='question of the image')
args = parser.parse_args()

def normalize_bbox(bbox, src_size, dst_size):
    """
    Normalize bounding box coordinates.

    Args:
        bbox (List[Union[int, float]]): Bounding box coordinates.
        src_size (Dict[str, Union[int, float]]): Source image size.
        dst_size (Dict[str, Union[int, float]]): Destination image size.

    Returns:
        List[Union[int, float]]: Normalized bounding box coordinates.
    """
    src_w, src_h = src_size["width"], src_size["height"]
    dst_w, dst_h = dst_size["width"], dst_size["height"]
    x1, y1, x2, y2 = bbox
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    x1 = int(x_min / src_w * dst_w)
    y1 = int(y_min / src_h * dst_h)
    x2 = int(x_max / src_w * dst_w)
    y2 = int(y_max / src_h * dst_h)

    x1 = max(0, min(x1, dst_w))
    y1 = max(0, min(y1, dst_h))
    x2 = max(0, min(x2, dst_w))
    y2 = max(0, min(y2, dst_h))

    return [x1, y1, x2, y2]


def main():
    # Load the config
    config = transformers.AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    generator_config = transformers.GenerationConfig.from_pretrained(
        args.model_dir
    ).to_dict()

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        trust_remote_code=True,
    )
    model = model.eval()
    model = model.to(torch.float32)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)

    # Load the image
    img_dir = args.img_dir
    image = Image.open(img_dir).convert('RGB')
    width, height = image.size
    image = processor(image,
                      [''],
                      boxes=[[0,0,0,0],],
                      return_tensors="pt",
                      padding=True)['pixel_values'][0]

    # Load the OCR
    ocr_dir = args.ocr_dir
    ocr_data = json.load(open(ocr_dir))

    # Load the instruction
    prompt = args.instruction
    fore_prompt = '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>'
    fore_prompt += '<|start_header_id|>user<|end_header_id|>\n\nGiving the document image patches,'
    fore_llm_value_ids = tokenizer.encode(fore_prompt, add_special_tokens=False,)
    fore_llm_value_ids = [tokenizer.bos_token_id,] + fore_llm_value_ids
    fore_llm_value_ids = fore_llm_value_ids + [img_patch_id,] * 196
    fore_prompt = ', and text content and its location in form of "text, [left, top, right, bottom]":\n'
    fore_llm_value_ids = fore_llm_value_ids + tokenizer.encode(fore_prompt, add_special_tokens=False,)
    aft_prompt = "\n" + prompt +  '<|eot_id|>'
    aft_prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
    aft_llm_value_ids = tokenizer.encode(aft_prompt, add_special_tokens=False,)

    bbox = []
    for _, line in enumerate(ocr_data):
        line_text = line['text'].strip()
        tokenized = tokenizer.encode(line_text, add_special_tokens=False) + \
            [spatial_position_id, ] + \
                tokenizer.encode("\n", add_special_tokens=False)

        fore_llm_value_ids += tokenized
        
        line_box = line['box']
        norm_box = normalize_bbox(line_box, {"width": width, "height": height},
                                  {"width": 1000, "height": 1000})
        bbox += [norm_box,]

    input_ids = fore_llm_value_ids + aft_llm_value_ids
    position_ids = list(range(len(input_ids)))

    input_ids = torch.LongTensor(input_ids)
    position_ids = torch.LongTensor(position_ids)
    bbox = torch.LongTensor(bbox)

    input = {
        "input_ids": input_ids.unsqueeze(0).to(model.device),
        "position_ids": position_ids.unsqueeze(0).to(model.device),
        "bbox": bbox.unsqueeze(0).to(model.device),
        "pixel_values": image.unsqueeze(0).to(model.device),
    }

    output = model.generate(**input, **generator_config).cpu()
    response = tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)
    print(f"Response: {response}")

if __name__ == '__main__':
    main()
