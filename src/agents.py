from transformers import AutoTokenizer, GenerationConfig, AutoProcessor
import torch
from PIL import Image
import os
import json

from .agent_utils import (
    get_inference_llm_prompt_from_template,
    get_inference_vlm_prompt_from_template,
    log_inference
)
from .prompts import *

class HuggingfaceLLMAgent():
    def __init__(self, model, log_path, model_name, system_prompt_func_config, user_input_func_config, t = -1, max_length = 4096):
        self.log_path = log_path
        self.model = model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.generation_config.max_length = max_length
        self.t = t
        
        system_prompt_func = globals()[system_prompt_func_config["type"]](**{k: system_prompt_func_config[k] for k in system_prompt_func_config.keys() if k != "type"})
        self.system_prompt_func = system_prompt_func
        user_input_func = globals()[user_input_func_config["type"]](**{k: user_input_func_config[k] for k in user_input_func_config.keys() if k != "type"})
        self.user_input_func = user_input_func
    
    def __call__(self, system_prompt_kwargs, user_input_kwargs):
        system_prompt = self.system_prompt_func(**system_prompt_kwargs)
        user_input = self.user_input_func(**user_input_kwargs)
        prompt = get_inference_llm_prompt_from_template(system_prompt, user_input, self.tokenizer)
        output = self.generate(prompt)

        cur_log = {
            "model_name":self.model_name,
            "system_prompt":system_prompt,
            "user_prompt":user_input,
            "output":output
        }
        log_inference(self.log_path, cur_log)
        return output
    
    def generate(self, prompt):
        do_sample = True
        if self.t == 0:
            do_sample = False
        elif self.t != -1:
            self.generation_config.temperature = self.t
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        tokenized_prompt_ids = tokenized_prompt["input_ids"].cuda()
        tokenized_prompt_mask = tokenized_prompt["attention_mask"].cuda()
        inputs = {"input_ids":tokenized_prompt_ids, "attention_mask":tokenized_prompt_mask}

        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=do_sample, generation_config=self.generation_config).detach().cpu()
        decoded = []
        for i in range(output.shape[0]):
            ans = self.tokenizer.decode(output[i][len(tokenized_prompt[0]):], skip_special_tokens=True)
            decoded.append(ans)
        decoded = decoded[0]
        return decoded

class HuggingfaceVLMAgent():
    def __init__(self, model, log_path, model_name, system_prompt_func_config, user_input_func_config, t = -1, max_length = 4096):
        self.log_path = log_path
        self.model = model
        self.tokenizer = AutoProcessor.from_pretrained(model_name)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.generation_config.max_length = max_length
        self.t = t
        
        system_prompt_func = globals()[system_prompt_func_config["type"]](**{k: system_prompt_func_config[k] for k in system_prompt_func_config.keys() if k != "type"})
        self.system_prompt_func = system_prompt_func
        user_input_func = globals()[user_input_func_config["type"]](**{k: user_input_func_config[k] for k in user_input_func_config.keys() if k != "type"})
        self.user_input_func = user_input_func
    
    def __call__(self, system_prompt_kwargs, user_input_kwargs):
        system_prompt = self.system_prompt_func(**system_prompt_kwargs)
        user_input = self.user_input_func(**user_input_kwargs)
        text, images = get_inference_vlm_prompt_from_template(system_prompt, user_input, self.tokenizer)
        output = self.generate(text, images)

        cur_log = {
            "model_name":self.model_name,
            "system_prompt":system_prompt,
            "user_prompt":user_input,
            "output":output
        }
        log_inference(self.log_path, cur_log)
        return output
    
    def generate(self, text, images):
        do_sample = True
        if self.t == 0:
            do_sample = False
        elif self.t != -1:
            self.generation_config.temperature = self.t
        images = [Image.open(path) for path in images]
        inputs = self.tokenizer(
                images,
                text,
                add_special_tokens=False,
                return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=do_sample, generation_config=self.generation_config).detach().cpu()
        decoded = []
        for i in range(output.shape[0]):
            ans = self.tokenizer.decode(output[i])
            decoded.append(ans)
        decoded = decoded[0]
        return decoded