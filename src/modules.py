from transformers import *
from huggingface_hub import login
import inspect

from .agents import *
from .module_utils import *

class MultiagentSystem():
    def __init__(self, config, log_path):
        self.context = {}
        self.models = {}
        self.log_path = log_path
        for model_config in config["model_configs"]:
            model_name = model_config["name"]
            model_type = globals()[model_config["type"]]
            model = model_type.from_pretrained(model_name,
                                             device_map='auto', low_cpu_mem_usage=True, 
                                  offload_state_dict=True)
            self.models[model_name] = model
        self.modules = []
        for module_config in config["module_configs"]:
            module_type = globals()[module_config["type"]]
            module = module_type(context = self.context, models = self.models, log_path = self.log_path, **{k: module_config[k] for k in module_config.keys() if k != "type"})
            self.modules.append(module)
    
    def __call__(self, user_input):
        self.context["input"] = user_input
        for module in self.modules:
            module()
        output = self.context.copy()
        self.context.clear()
        return output

class TerminalModule():
    def __init__(self, context, models, output_path):
        self.context = context
        self.models = models
        self.output_path = output_path
    
    def __call__(self):
        inp = input("Input: ")
        set_context_value(self.context, self.output_path, inp)

class AgentModule():
    def __init__(self, context, models, log_path, agent_config, input_paths, output_path, output_processor_config = None, output_processor_args_paths = None):
        self.context = context
        self.models = models
        self.log_path = log_path
        
        agent_type = globals()[agent_config["type"]]
        args = [name for name, param in inspect.signature(agent_type).parameters.items()]
        if "model" in args:
            model = models[agent_config["model_name"]]
            self.agent = agent_type(model = model, log_path = self.log_path, **{k: agent_config[k] for k in agent_config.keys() if k != "type"})
        else:
            self.agent = agent_type(**{k: agent_config[k] for k in agent_config.keys() if k != "type"})
        
        self.input_paths = input_paths
        self.output_path = output_path
        
        self.output_processor = None
        if output_processor_config != None:
            output_processor_type = globals()[output_processor_config["type"]]
            self.output_processor = output_processor_type(**{k: output_processor_config[k] for k in output_processor_config.keys() if k != "type"})
        self.output_processor_args_paths = output_processor_args_paths
    
    def __call__(self):
        system_prompt_kwargs = {prompt_key: get_context_value(self.context, self.input_paths["system"][prompt_key]) for prompt_key in self.input_paths["system"].keys()}
        user_prompt_kwargs = {prompt_key: get_context_value(self.context, self.input_paths["user"][prompt_key]) for prompt_key in self.input_paths["user"].keys()}
        output = self.agent(system_prompt_kwargs, user_prompt_kwargs)
        if self.output_processor != None:
            output_processor_args = {}
            if self.output_processor_args_paths != None:
                output_processor_args = {key: get_context_value(self.context, self.output_processor_args_paths[key]) for key in self.output_processor_args_paths.keys()}
            output = self.output_processor(output, **output_processor_args)
        set_context_value(self.context, self.output_path, output)

class IterativeModule():
    def __init__(self, context, models, log_path, splitting_input_path, splitting_func_config, splitted_buffer_path, output_path, result_buffer_path, result_base_value, base_module_config):
        self.context = context
        self.models = models
        self.log_path = log_path
        
        self.splitting_input_path = splitting_input_path
        splitting_func_type = globals()[splitting_func_config["type"]]
        self.splitting_func = splitting_func_type(**{k: splitting_func_config[k] for k in splitting_func_config.keys() if k != "type"})
        self.splitted_buffer_path = splitted_buffer_path
        self.output_path = output_path
        self.result_buffer_path = result_buffer_path
        self.result_base_value = result_base_value
        base_module_type = globals()[base_module_config["type"]]
        self.base_module = base_module_type(context = context, models= models, log_path = self.log_path, **{k: base_module_config[k] for k in base_module_config.keys() if k != "type"})
    
    def __call__(self):
        set_context_value(self.context, self.output_path, [])
        set_context_value(self.context, self.result_buffer_path, self.result_base_value)
        splitted_input = self.splitting_func(get_context_value(self.context, self.splitting_input_path))
        
        for i in range(len(splitted_input)):
            set_context_value(self.context, self.splitted_buffer_path, splitted_input[i])
            self.base_module()
            last_output_state = get_context_value(self.context, self.output_path)
            cur_output = get_context_value(self.context, self.result_buffer_path)
            last_output_state.append(cur_output)