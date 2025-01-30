import json
from string import Formatter
import inspect

class TextFewShots():
    def __init__(self, instr, template, few_shots_path, few_shot_inds):
        self.instr = instr
        self.template = template
        with open(few_shots_path, "r") as f:
            total_few_shots_base = json.load(f)
        self.few_shots_base = [total_few_shots_base[i] for i in few_shot_inds]
    def __call__(self):
        few_shots = ""
        for ind, shot in enumerate(self.few_shots_base):
            shot_keys = {k: shot[k] for k in shot.keys()}
            shot_keys["n_example"] = str(ind + 1)
            shot = self.template.format(**shot_keys)
            if ind == 0:
                few_shots = shot
            else:
                few_shots = few_shots + "\n" + shot
        output = self.instr.format(few_shots = few_shots)
        return output

class MultimodalFewShots():
    def __init__(self, instr, template, few_shots_path, few_shot_inds, few_shot_args_modalities):
        self.instr = instr
        self.template = template
        with open(few_shots_path, "r") as f:
            total_few_shots_base = json.load(f)
        self.few_shots_base = [total_few_shots_base[i] for i in few_shot_inds]
        self.few_shot_args_modalities = few_shot_args_modalities
    def __call__(self):
        instr = self.instr.format(few_shots = "")
    
        few_shots = [{"type":"text", "text":instr}]
        for ind, shot in enumerate(self.few_shots_base):
            shot_keys = {k: shot[k] for k in shot.keys()}
            shot_keys["n_example"] = str(ind + 1)
            
            split_keys = {}
            for key in shot_keys.keys():
                if key not in self.few_shot_args_modalities.keys():
                    split_keys[key] = shot_keys[key]
                else:
                    split_keys[key] = "{" + key + "}"
            raw_few_shot = self.template.format(**split_keys)
            chunks = [{"type":"text", "text":raw_few_shot}]
            for arg in self.few_shot_args_modalities.keys():
                new_chunks = []
                for chunk in chunks:
                    if chunk["type"] != "text":
                        new_chunks.append(chunk)
                        continue
                    text = chunk["text"]
                    splitted = text.split("{" + arg + "}")
                    for i, part in enumerate(splitted):
                        text_chunk = {"type":"text", "text":part}
                        new_chunks.append(text_chunk)
                        if i < len(splitted) - 1:
                            modal_chunk = {"type":self.few_shot_args_modalities[arg], "text":shot_keys[arg]}
                            new_chunks.append(modal_chunk)
                chunks = new_chunks
            
            if ind > 0 and few_shots[len(few_shots) - 1]["type"] == "text" and chunks[0]["type"] == "text":
                few_shots[len(few_shots) - 1]["text"] = few_shots[len(few_shots) - 1]["text"] + "\n"
            for chunk in chunks:
                if chunk["type"] != "text":
                    few_shots.append(chunk)
                else:
                    if few_shots[len(few_shots) - 1]["type"] != "text":
                        few_shots.append(chunk)
                    else:
                        few_shots[len(few_shots) - 1]["text"] = few_shots[len(few_shots) - 1]["text"] + chunk["text"]
            
        return few_shots

class UserPrompt():
    def __init__(self, template, args_modalities = None):
        self.template = template
        self.args_modalities = args_modalities
    
    def __call__(self, **kwargs):
        prompt = []
        if self.args_modalities == None:
            prompt.append({"type":"text", "text":self.template.format(**kwargs)})
        else:
            split_keys = {}
            for key in kwargs.keys():
                if key not in self.args_modalities.keys():
                    split_keys[key] = kwargs[key]
                else:
                    split_keys[key] = "{" + key + "}"
            raw_few_shot = self.template.format(**split_keys)
            chunks = [{"type":"text", "text":raw_few_shot}]
            for arg in self.args_modalities.keys():
                new_chunks = []
                for chunk in chunks:
                    if chunk["type"] != "text":
                        new_chunks.append(chunk)
                        continue
                    text = chunk["text"]
                    splitted = text.split("{" + arg + "}")
                    for i, part in enumerate(splitted):
                        text_chunk = {"type":"text", "text":part}
                        new_chunks.append(text_chunk)
                        if i < len(splitted) - 1:
                            modal_chunk = {"type":self.args_modalities[arg], "text":kwargs[arg]}
                            new_chunks.append(modal_chunk)
                chunks = new_chunks
            prompt = chunks
        return prompt

class SystemPrompt():
    def __init__(self, instructions, instruction_context_path, linebreak_after = []):
        self.instructions = []
        self.linebreak_after = linebreak_after
        for instr in instructions:
            if type(instr) == type(""):
                self.instructions.append(instr)
            elif type(instr) == type({}):
                instr_type = globals()[instr["type"]]
                instruction = instr_type(**{k: instr[k] for k in instr.keys() if k != "type"})
                self.instructions.append(instruction)
            else:
                raise TypeError
            
        with open(instruction_context_path, "r") as f:
            instruction_context = json.load(f)
        self.instruction_context = instruction_context
            
    def __call__(self, **input_context):
        prompt = []
        for i, instr in enumerate(self.instructions):
            if type(instr) == type(""):
                args = [k for _, k, _, _ in Formatter().parse(instr) if k]
            elif callable(instr):
                params = inspect.signature(instr).parameters
                args = [name for name, param in params.items()]
            else:
                raise TypeError
            
            kwargs = {}
            for arg in args:
                if arg in input_context.keys():
                    value = input_context[arg]
                elif arg in self.instruction_context.keys():
                    value = self.instruction_context[arg]
                else:
                    raise KeyError
                kwargs[arg] = value
            
            if type(instr) == type(""):
                prompt_part = instr.format(**kwargs)
            elif callable(instr):
                prompt_part = instr(**kwargs)
            
            if type(prompt_part) == type(""):
                prompt_part = [{"type":"text", "text":prompt_part}]
            
            add_space = prompt == "" or i - 1 in self.linebreak_after
            add_linebreak = i in self.linebreak_after
            
            for ind, chunk in enumerate(prompt_part):
                if chunk["type"] != "text" or len(prompt) == 0:
                    prompt.append(chunk)
                elif len(prompt) > 0:
                    if prompt[-1]["type"] != "text":
                        prompt.append(chunk)
                        continue
                    if ind == 0 and add_space:
                        prompt[-1]["text"] = prompt[-1]["text"] + " " + chunk["text"]
                    elif ind == len(prompt_part) - 1 and add_linebreak:
                        prompt[-1]["text"] = prompt[-1]["text"] + chunk["text"] + "\n"
                    else:
                        prompt[-1]["text"] = prompt[-1]["text"] + chunk["text"]
                        
        return prompt

