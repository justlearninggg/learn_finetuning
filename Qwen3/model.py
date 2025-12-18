import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda"

BASE_MODEL_PATH = "/home/lcx/self-llm/models/Qwen3/Qwen/Qwen3-1.7B"
LORA_PATH = "/home/lcx/temp/llamafactory/LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_2025-12-18-14-19-35"


class QwenModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            use_fast=False,
            trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载 LoRA，外挂lora adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            LORA_PATH
        )

        self.model.eval()

    @torch.no_grad()
    def generate(self, instruction: str, user_input: str, max_new_tokens=512):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return response
