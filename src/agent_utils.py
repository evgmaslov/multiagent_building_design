import os
import json

def get_train_llm_prompt_from_tokens(system_prompt, user_input, assistant_output, b_inst_token, e_inst_token, eos_token):
  output_format = """{output}{eos}
  """
  inp = get_inference_llm_prompt_from_tokens(system_prompt, user_input, b_inst_token, e_inst_token, eos_token)
  out = inp + output_format.format(output=assistant_output, eos=eos_token)
  return out

def get_train_llm_prompt_from_template(system_prompt, user_input, assistant_output, tokenizer):
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_input},
      {"role": "assistant", "content": assistant_output},
  ]
  out = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
  )
  return out

def get_inference_llm_prompt_from_tokens(system_prompt, user_input, b_inst_token, e_inst_token, eos_token):
  input_format_without_prompt = """{b_inst}user{e_inst}
  {text}{eos}{b_inst}assistant{e_inst}
  """
  input_format_with_prompt = """{b_inst}system{e_inst}
  {prompt}{eos}""" + input_format_without_prompt
  input_text = ""
  if len(system_prompt) == 0:
    input_text = input_format_without_prompt.format(b_inst=b_inst_token, text=user_input, e_inst=e_inst_token, eos=eos_token)
  else:
    input_text = input_format_with_prompt.format(prompt=system_prompt, b_inst=b_inst_token, text=user_input, e_inst=e_inst_token, eos=eos_token)
  return input_text

def get_inference_llm_prompt_from_template(system_prompt, user_input, tokenizer):
  messages = [
      {"role": "system", "content": system_prompt[0]["text"]},
      {"role": "user", "content": user_input[0]["text"]}
  ]
  input_text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  return input_text

def get_inference_vlm_prompt_from_template(system_prompt_chunks, user_prompt_chunks, tokenizer):
  images = []
  simple_system_prompt_chunks = []
  for chunk in system_prompt_chunks:
    if chunk["type"] == "text":
      simple_system_prompt_chunks.append(chunk)
    else:
      simple_system_prompt_chunks.append({"type":"image"})
      images.append(chunk["text"])
  simple_user_prompt_chunks = []
  for chunk in user_prompt_chunks:
    if chunk["type"] == "text":
      simple_user_prompt_chunks.append(chunk)
    else:
      simple_user_prompt_chunks.append({"type":"image"})
      images.append(chunk["text"])
  messages = [
      {"role": "system", "content": simple_system_prompt_chunks},
      {"role": "user", "content": simple_user_prompt_chunks}
  ]
  input_text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  return input_text, images

def log_inference(log_path, inference_dict):
  if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
  else:
      log = []
  log.append(inference_dict)
  with open(log_path, "w+") as f:
      json.dump(log, f, ensure_ascii=False)