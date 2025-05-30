import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_qwen_model(model_path="Qwen/Qwen2.5-1.5B", use_fp16=True):
    log.info(f"Loading Qwen model from {model_path}...")
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def chat_with_model(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    # Extract content between <context>: and <semantic_knowledge> if present
    prompt = "Please generate questions based on the <context> provided:<context>:" + prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return assistant_response

def get_data(data_path):
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif data_path.endswith('.json'):
        return json.load(open(data_path, "r"))
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Only .json and .jsonl files are supported.")

def main():
    # model_path = "/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/4_result/all/checkpoints"
    model_path = "/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/4_result/all/checkpoints"
    model, tokenizer = load_qwen_model(model_path=model_path)
    data = get_data("/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/test/test_data/main_test_11877.jsonl")
    for item in data:
        full_prompt = item["prompt"] + item["input"]
        log.info(f"Input: {full_prompt}")
        response = chat_with_model(model, tokenizer, item["input"])
        
        log.info(f"Answer: {item['output']}")
        log.info(f"Response: {response}")
        log.info("-"*50)
if __name__ == "__main__":
    main()
