import os
import torch
import transformers
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from loguru import logger as log


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
MODEL_NAME = "/home/qhn/Codes/Models/Qwen2-7B-Instruct/"  
OUTPUT_DIR = "/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/4_result/all"
DATA_PATH = "/home/qhn/Codes/Projects/zz-exp/qg-zhengli/data/4_result/all"


MICRO_BATCH_SIZE = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
EPOCHS = 3
MAX_LENGTH = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_LENGTH_Q = 256 - 2  # default=128 - 2
MAX_LENGTH_A = 256 - 2  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
ID_PAD = 151643
ID_EOS = 151643  # endoftext
ID_SOP = 151644  # start
ID_EOP = 151645  # end
ID_BR = 198  # "\n"

# LoRA target modules for Qwen2
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Set up logging
log.add(
    "./log/lora_training.log",
    encoding="utf-8",
    format="{level} | {time:YYYY-MM-DD HH:mm:ss} | {file} | {line} | {message}",
    rotation="500 MB",
)

def data_collator_fn(batch):
    len_max_batch = [
        len(batch[i].get("input_ids")) + len(batch[i].get("labels"))
        for i in range(len(batch))
    ]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        labels = [-100] * len(x) + y + [-100] * len_padding
        input_ids = x + y + [ID_PAD] * len_padding
        attention_mask = [1] * (len(x) + len(y)) + [0] * len_padding
        # 先暂时用 0 作为 task_idx
        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {
        "attention_mask": batch_attention_mask,
        "input_ids": batch_input_ids,
        "labels": batch_labels,
    }
    return input_dict


def train_lora_model():
    """Train a LoRA model on the dataset."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model and tokenizer
    log.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Create LoRA model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    def generate_prompt(data_point, is_logger=False):
        """直接encode，最后强制补上"""
        # print(f"data_point: {data_point}")
        text_input = data_point.get("instruction", "")
        text_out = data_point.get("output", "")
        task_idx = data_point.get("task_idx", 0)

        system_str = "You are a helpful assistant."
        prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(system_str)
        prompt_text_1 = prompt_system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        prompt_text_2 = "assistant\n{}<|im_end|><|endoftext|>"
        text_1 = prompt_text_1.format(text_input.strip())
        text_2 = prompt_text_2.format(text_out)
        # end with <|im_end|><|endoftext|>
        x = tokenizer.encode(text_1)
        y = tokenizer.encode(text_2)
        if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
            x = x[:MAX_LENGTH_Q]
            y = y[:MAX_LENGTH_A]
        if not x:
            x = [ID_SOP, ID_EOP, ID_BR, ID_SOP]
        x[-3:] = [ID_EOP, ID_BR, ID_SOP]
        if not y:
            y = [ID_EOP, ID_EOS]
        y[-2:] = [ID_EOP, ID_EOS]
        out = {"input_ids": x, "labels": y}
        if is_logger:
            print(text_1)
            print(text_2)
            print(out)
        return out
    # Load and process dataset
    try:
        dataset = load_dataset("json", data_files=f"{DATA_PATH}/train.json")
        train_dataset = dataset["train"].shuffle().map(generate_prompt)
        log.info(f"Training dataset size: {len(train_dataset)}")
    except Exception as e:
        log.error(f"Error loading dataset: {e}")
        raise

    # Data collator
    class DataCollatorForCausalLM:
        def __call__(self, examples):
            batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
            # Set labels to -100 for non-assistant tokens to avoid computing loss on them
            batch["labels"] = batch["input_ids"].clone()
            # Find positions of <|im_start|>assistant and <|im_end|> tokens
            for i in range(len(examples)):
                # Get the input_ids for this example
                input_ids = batch["input_ids"][i]
                
                # Find the token IDs for the special tokens
                assistant_start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
                
                # Find positions in the input_ids
                assistant_positions = []
                for j in range(len(input_ids) - len(assistant_start_tokens) + 1):
                    if all(input_ids[j + k] == assistant_start_tokens[k] for k in range(len(assistant_start_tokens))):
                        assistant_positions.append(j + len(assistant_start_tokens))
                
                end_positions = []
                for j in range(len(input_ids) - len(im_end_tokens) + 1):
                    if all(input_ids[j + k] == im_end_tokens[k] for k in range(len(im_end_tokens))):
                        end_positions.append(j)
                
                # If we found the assistant start token
                if assistant_positions:
                    tokens_before_assistant = assistant_positions[0]
                    # Set labels to -100 for tokens before assistant response
                    batch["labels"][i, :tokens_before_assistant] = -100
                    
                    # Find the corresponding end token after the assistant start
                    for end_pos in end_positions:
                        if end_pos > tokens_before_assistant:
                            # Set labels to -100 for tokens after assistant response
                            batch["labels"][i, end_pos:] = -100
                            break
            return batch

    data_collator = DataCollatorForCausalLM()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=1000,
        fp16=True,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator_fn
    )

    # Train model
    log.info("Starting training...")
    trainer.train()

    # Save the final model
    # model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    # tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    log.info(f"Model saved to {OUTPUT_DIR}/final_model")
    
    return model, tokenizer

def run_inference(model_path=f"{OUTPUT_DIR}/final_model", prompt="Please generate questions based on the <context> provided", input_text="<context>:one of the roles of computational complexity theory is to determine the practical limits on what computers can and can not do ."):
    """Run inference with the fine-tuned model."""
    log.info("Testing inference with the fine-tuned model...")
    
    # Load the fine-tuned model for inference
    inference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    inference_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Combine prompt and input text
    full_prompt = f"{prompt}\n{input_text}"
    
    # Generate and print response
    response = generate_response(full_prompt, inference_model, inference_tokenizer)
    log.info(f"Prompt: {full_prompt}")
    log.info(f"Response: {response}")
    
    return response

def generate_response(prompt, model, tokenizer, max_new_tokens=512):
    """Generate a response using the fine-tuned model."""
    # Format the prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return assistant_response

def main():
    train_lora_model()
    
# Example usage
if __name__ == "__main__":
    main()