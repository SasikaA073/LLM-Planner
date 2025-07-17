"""
ALFRED Evaluation Script with LLM Planning
------------------------------------------
Main evaluation framework that tests language model-based planning in AI2-THOR environments.
Connects LLMs with the simulation to perform household tasks, supporting dynamic replanning,
vision integration, and comprehensive metrics collection. Serves as the primary entry point
for running experiments on the ALFRED benchmark.
"""

# ALL in one script to run LLM-Planner on ALFRED tasks

import os
import base64
import sys
import json
import yaml
import cv2
import argparse
import time
import datetime
from tqdm import tqdm
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
import textwrap
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from io import BytesIO
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from my_adapter import DCTAdapter

from alfred.thor_connector import ThorConnector
from alfred.utils import dotdict, load_task_json
from alfred.data.preprocess import Dataset
from hlp_planner import LLM_Planner
from adapter_helper_functions import * 

sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

# Configure the root logger to print to console
logging.basicConfig(
    level=logging.ERROR,  # I changed from DEBUG to ERROR to reduce the clutter # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a logger for this module
log = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class AlfredEvaluator:
    def __init__(self, config_file):
        # Load configuration
        with open(config_file) as reader:
            self.config = yaml.safe_load(reader)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.trainFinished = True #Dulanga
        
        # Initialize LLM planner
        self.llm_planner = LLM_Planner(
            knn_data_path=self.config["llm_planner"]["knn_dataset_path"], 
            emb_model_name=self.config["llm_planner"]["emb_model_name"], 
            debug=self.config["llm_planner"]["debug"]
        )
        
        # Initialize environment
        self.env = ThorConnector(x_display=self.config["alfred"]["x_display"])

        # Load task splits
        with open(self.config["alfred"]["splits"]) as f:
            self.splits = json.load(f)
            
        # Prepare tasks
        self.tasks = self._prepare_tasks()
        
        # Initialize the sentence transformer for object name matching
        self.obj_encoder = SentenceTransformer(self.config["llm_planner"]["emb_model_name"])
        self.obj_sim_threshold = self.config["llm_planner"].get("obj_sim_threshold", 0.8)  # Default similarity threshold
        
        # Checkpoint configuration
        self.checkpoint_dir = self.config.get("checkpoint_dir", "src/checkpoints") # Use src/checkpoints as default
        self.save_every = self.config.get("save_every", 5) # Save every 10 training steps
        self.training_step = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Adaptation
        self.adapter_config_path = "config/adapter_config.yaml"
        
        # Initialize Llama model if specified in config
        if self.config["llm_planner"]["engine"] == "llama-3-vision":
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cuda",  # Changed from "cpu" to "auto" for better performance
            )
            self.llama_processor = AutoProcessor.from_pretrained(model_id)

            # Initialize optimizer for online training
            from torch.optim import AdamW
            self.llama_optimizer = AdamW(
                self.llama_model.parameters(),
                lr=1e-6,  # Very small learning rate for online learning
                weight_decay=0.01
            )

            print("✅ Llama 3 model loaded")
        
        elif self.config["llm_planner"]["engine"] == "llama-3-vision-adapted":
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cuda",  # Changed from "cpu" to "auto" for better performance
                load_in_8bit=True  # 8-bit quantization
            )
            self.llama_processor = AutoProcessor.from_pretrained(model_id)

            
            
            # Initialize optimizer for online training
            from torch.optim import AdamW
            self.llama_optimizer = AdamW(
                self.llama_model.parameters(),
                lr=1e-6,  # Very small learning rate for online learning
                weight_decay=0.01
            )

        # Initialize Mistral model if specified in config
        elif self.config["llm_planner"]["engine"] == "mistral-7b-instruct":
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"

            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="auto", # Use auto for better distribution
                load_in_8bit=True  # 8-bit quantization
            )
            self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_id)

            print("✅ Mistral model loaded")

        elif self.config["llm_planner"]["engine"] == "mistral-7b-instruct-adapted":
            engine = self.config["llm_planner"]["engine"]
            print(f"🦙 Initializing Mistral model ({engine})...")
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            print(f"📥 Loading model: {model_id}")

            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="auto", # Use auto for better distribution
                load_in_8bit=True  # 8-bit quantization
            )
            print("📥 Loading tokenizer...")
            self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Do adaptation
            logger.info(f"Loading adapter configuration from {self.adapter_config_path}")
            with open(self.adapter_config_path, 'r') as f:
                adapter_config_yaml = yaml.safe_load(f)

            adapter_params_from_yaml = adapter_config_yaml.get('adapter', {}).get('params', {})
            # Ensure 'input_dim' from yaml is correctly named 'input_dim' for DCTAdapter constructor
            # The DCTAdapter class expects 'input_dim'. The yaml has 'input_dim'.
            # num_components is also in yaml and DCTAdapter constructor.
            
            adapter_layers_from_yaml = adapter_config_yaml.get('adapter', {}).get('layers', [])
            
            if not adapter_layers_from_yaml:
                logger.error("No adapter layers specified in the YAML configuration. Exiting.")
                exit(1)
            if not adapter_params_from_yaml:
                logger.warning("No adapter parameters (params) specified in the YAML. Using defaults for DCTAdapter if any.")

            # Inject DCT Adapter 
            print("Injecting DCT Adapters...")
            self.mistral_model = inject_adapters(self.mistral_model, DCTAdapter, 
                                    base_adapter_args=adapter_params_from_yaml, 
                                    layers_config=adapter_layers_from_yaml)
            
           
            adapter_device = self.mistral_model.device
            for module in self.mistral_model.modules():
                if isinstance(module, DCTAdapter):  # Change this according to the Adapter name 
                    module.to(dtype=torch.float16, device=adapter_device)
            # self.mistral_model.to(self.mistral_model.device)

            # CHECK THE DTYPE 
            for name, param in self.mistral_model.named_parameters():
                if 'adapter' in name:
                    print(f"[DEBUG] {name} – dtype: {param.dtype}, device: {param.device}")

            print("✅ Adapted Mistral model ready!")
            from torch.optim import AdamW

            freeze_model_except_adapters(self.mistral_model)

            trainable_params = filter(lambda p: p.requires_grad, self.mistral_model.parameters())
            
            print(f"[DEBUG] trainable params :\n{trainable_params}")
            self.mistral_optimizer = AdamW(
                trainable_params,
                lr=1e-6,  # Very small learning rate for online learning
                weight_decay=0.01
            )
            print(f"Adapter Configuration : \n{adapter_config_yaml}")

        elif self.config["llm_planner"]["engine"] == "qwen2.5-3B-instruct":
            model_id = "Qwen/Qwen2.5-3B-Instruct" 

            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 ,
                device_map="cuda",
                load_in_8bit=True # 8-bit quanitization 
            )

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)

            print("✅ Qwen2.5-3B model loaded")

        elif self.config["llm_planner"]["engine"] == "gemma-2-9b-it":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.gemma2_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
            self.gemma2_model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-9b-it",
                device_map="auto",
                torch_dtype=torch.float16,
            )

            print("Gemma 2 model is loaded")

        elif self.config["llm_planner"]["engine"] == "gemma-2-9b-it-adapted":
            engine = self.config["llm_planner"]["engine"]
            print(f"Initializing Gemma 2 9B instruct ")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_id = "google/gemma-2-9b-it"
            self.gemma2_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.gemma2_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            print("Gemma 2 model is loaded")
            print("[GEMMA 2 Model Architecure]")
            # print(self.gemma2_model)

            logger.info(f"Loading adapter configuration from {self.adapter_config_path}")
            with open(self.adapter_config_path, 'r') as f:
                adapter_config_yaml = yaml.safe_load(f)

            adapter_params_from_yaml = adapter_config_yaml.get('adapter', {}).get('gemma_2_params', {})
           
            # Ensure 'input_dim' from yaml is correctly named 'input_dim' for DCTAdapter constructor
            # The DCTAdapter class expects 'input_dim'. The yaml has 'input_dim'.
            # num_components is also in yaml and DCTAdapter constructor.
            
            adapter_layers_from_yaml = adapter_config_yaml.get('adapter', {}).get('gemma_2_layers', [])
            
            if not adapter_layers_from_yaml:
                logger.error("No adapter layers specified in the YAML configuration. Exiting.")
                exit(1)
            if not adapter_params_from_yaml:
                logger.warning("No adapter parameters (params) specified in the YAML. Using defaults for DCTAdapter if any.")

            # Inject DCT Adapter 
            print("Injecting DCT Adapters...")
            self.gemma2_model = inject_adapters(self.gemma2_model, DCTAdapter, 
                                    base_adapter_args=adapter_params_from_yaml, 
                                    layers_config=adapter_layers_from_yaml)
            
            adapter_device = self.gemma2_model.device
            for module in self.gemma2_model.modules():
                if isinstance(module, DCTAdapter):
                    module.to(dtype=torch.float16, device=adapter_device)

            # CHECK THE DTY:PE 
            for name, param in self.gemma2_model.named_parameters():
                if 'adapter' in name: 
                    print(f"[DEBUG] {name} - dtype: {param.dtype}, device: {param.device}")

            print("✅ Adapted Gemma2 model ready!")
            from torch.optim import AdamW 

            freeze_model_except_adapters(self.gemma2_model)

            trainable_params = filter(lambda p: p.requires_grad, self.gemma2_model.parameters())

            print(f"[DEBUG] trainable params :\n{trainable_params}")
            self.gemma2_optimizer = AdamW(
                trainable_params,
                lr=1e-6,  # Very small learning rate for online learning
                weight_decay=0.01
            )
            print(f"Adapter Configuration : \n{adapter_config_yaml}")
            print(f"[Adpated model archi]", self.gemma2_model)


        # TODO: Add the model {GEMMA}
        elif self.config["llm_planner"]["engine"] == "gemma-3-4b-it":
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            
            import requests
            import torch

            model_id = "google/gemma-3-12b-it"

            self.gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cuda", 
                torch_dtype=torch.float16
            ).eval()

            self.gemma_processor = AutoProcessor.from_pretrained(model_id)


            print("✅ Gemma model loaded")

        
        self.training_samples = self.config["llm_planner"]["n_training_samples"]  # First 50 samples for training
        self.dry_run_samples = self.config["llm_planner"]["n_dry_run_samples"]
        # sys.exit()
        
    def _prepare_tasks(self):
        """Prepare tasks for evaluation"""
        assert self.config["alfred"]["eval_set"] in self.splits.keys()
        tasks = []
        
        # exclude two obj task
        for e in self.splits[self.config["alfred"]["eval_set"]]:
            if 'pick_two_obj_and_place' not in e['task']:
                tasks.append(e)
                
        # Debug mode
        if self.config.get("debug", False):
            for task in tasks:
                if 'trial_T20190906_201106_979461' in task['task']: #NOTE Change this to the task you want to debug
                    new_task = [task]
                    break
            tasks = new_task
            
        return tasks
    
    def llm(self, prompt, engine, images=None, stop=["\n"], do_train=False):
        """Interface to LLM models"""
        if engine in ['gpt-4o-mini', 'gpt-4o']:
            # Create the base message content
            message_content = []
            
            # Add image if provided
            if images and len(images) > 0:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{images[0]}"
                    }
                })
            
            # Add text prompt
            message_content.append({
                "type": "text",
                "text": prompt,
            })
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ],
                max_tokens=300,
                temperature=0.0
            )
            openai_response = response.choices[0].message.content

            return openai_response
        
        elif engine in ["llama-3-vision", "llama-3-vision-adapted"]:
            # Convert base64 image back to PIL Image if provided
            image = None
            if images and len(images) > 0:
                image_data = base64.b64decode(images[0])
                image = Image.open(BytesIO(image_data))
            
            # Format messages for Llama
            messages = [
                {"role": "user", "content": [
                    {"type": "image"} if image else {},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Remove empty image content if no image provided
            if not image:
                messages[0]["content"] = [{"type": "text", "text": prompt}]
            
            # Generate response
            with torch.no_grad():
                input_text = self.llama_processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.llama_processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.llama_model.device)
                
                output = self.llama_model.generate(**inputs, max_new_tokens=300, 
                                                   temperature=1.0,
                                                   do_sample=False # <- important for greedy decoding
                )
                # Decode only the new tokens (skip the input)
                generated_text = self.llama_processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                llama_output = generated_text.strip()

            print(f"Input to Llama :\n{messages}")
            print(f"Llama output :\n{llama_output}")


            # Online training code for adapted version
            if do_train:
                # Get response from the GPT-4o mini as the ground truth 
                groundtruth = self.llm(prompt=prompt, engine="gpt-4o-mini", images=images, do_train=False)
                
                # Perform online training
                self._train_llama_online(messages, groundtruth, image)

            return llama_output

        elif engine == "mistral-7b-instruct":
            # Prepare messages in plain string format
            messages = [{"role": "user", "content": prompt}]

            # Ensure pad_token is set
            if self.mistral_tokenizer.pad_token is None:
                self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

            # Generate response (inference) inside no_grad context
            with torch.inference_mode():
                inputs = self.mistral_tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True
                ).to(self.mistral_model.device)

                generated_ids = self.mistral_model.generate(
                    inputs,
                    pad_token_id=self.mistral_tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95
                )
                decoded_output = self.mistral_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                mistral_output = decoded_output[0].strip()

            # Perform online training if using the adapted model
            if do_train:
                groundtruth = self.llm(prompt=prompt, engine="gpt-4o-mini", images=images, do_train=False)
                
                # Perform online training (this method has its own gradient calculation)
                self._train_mistral_online(messages, groundtruth)

            return mistral_output
        
        elif engine == "mistral-7b-instruct-adapted":
            # Define preference for generating aligned HLPs
            
            messages = [{"role": "user", "content": prompt}]

            # Ensure pad_token is set
            if self.mistral_tokenizer.pad_token is None:
                self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

            # Generate response (inference) inside no_grad context
            with torch.inference_mode():
                inputs = self.mistral_tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True # when True the model echoed the input
                ).to(self.mistral_model.device)    

                # Manually create attention_mask to avoid inference issues
                attention_mask = (inputs != self.mistral_tokenizer.pad_token_id).long()

                # Move to the model's device
                inputs = inputs.to(self.mistral_model.device)
                attention_mask = attention_mask.to(self.mistral_model.device)

                print("[DEBUG] Inputs device =", inputs.device)
                print("[DEBUG] Mistral model device =", next(self.mistral_model.parameters()).device)
                print("[DEBUG] attention mask device =", attention_mask.device)

                generated_ids = self.mistral_model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    pad_token_id=self.mistral_tokenizer.eos_token_id,
                    max_new_tokens=1000,
                    do_sample=False,
                    # temperature=0.7, # Since i do not sample and want a deterministic output
                    # top_p=0.95
                )

                decoded_output = self.mistral_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                mistral_output = decoded_output[0].strip()

            return mistral_output
        
        elif engine == "qwen2.5-3B-instruct":
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            qwen_model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.qwen_model.device)

            generated_ids = self.qwen_model.generate(
                **qwen_model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(qwen_model_inputs.input_ids, generated_ids)
            ]

            response = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        elif engine == "gemma-2-9b-it":
            input_text = prompt
            input_ids = self.gemma2_tokenizer(input_text, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                outputs = self.gemma2_model.generate(**input_ids, max_new_tokens=500)
                response = self.gemma2_tokenizer.decode(outputs[0])

            return response 
        
        elif engine == "gemma-2-9b-it-adapted":
            input_text = prompt
            input_ids = self.gemma2_tokenizer(input_text, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                outputs = self.gemma2_model.generate(**input_ids, max_new_tokens=500)
                response = self.gemma2_tokenizer.decode(outputs[0])

            return response 

        # TODO: Add Processing {GEMMA}
        elif engine == "gemma-3-4b-it":

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            inputs = self.gemma_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.gemma_model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.gemma_model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]

            decoded = self.gemma_processor.decode(generation, skip_special_tokens=True)
            print("Gemma Response: \n",decoded)

            return decoded

           


            
        else:
            raise ValueError(f"{engine} is not supported!")

    def encode_image(self, img):
        """Encode image to base64 string"""
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            numpy_img = np.array(img)
        else:
            numpy_img = img
            
        # Convert RGB to BGR for OpenCV
        if len(numpy_img.shape) == 3 and numpy_img.shape[2] == 3:
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
            
        _, JPEG = cv2.imencode('.jpeg', numpy_img)
        return base64.b64encode(JPEG).decode('utf-8')

    def evaluate_task(self, engine, traj_data, r_idx, dynamic=False, to_print=True, vision=False, ob='',do_train=False):
        """Evaluate a single task"""
        # Initialize frame history and plan tracking
        frame_history = []
        completed_plans = []
        failed_plans = []
        
        # Setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = f'FloorPlan{scene_num}'
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # Initialize and save initial frame
        self.env.step(dict(traj_data['scene']['init_action']))
        frame_history.append(self.env.last_event.frame.copy())
        self.env.set_task(traj_data, dotdict(self.config["alfred"]["env_args"]), reward_type='dense')

        # Get task instructions
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        # Always get step instructions, but control their usage with includeLow flag
        step_instrs = [ann["high_descs"] for ann in traj_data["turk_annotations"]["anns"]]
        
        # Get configuration for using step instructions
        use_step_instructions = self.config["llm_planner"].get("use_step_instructions", True)
        
        log.debug(f"Task: {goal_instr}")
        if use_step_instructions:
            log.debug("Step instructions: Enabled")
        else:
            log.debug("Step instructions: Disabled")

        # Setup for evaluation
        done, success = False, False
        reward = 0

        # Get initial frames if vision model is used
        encoded_frames = []
        if vision:
            init_frames = [Image.fromarray(self.env.last_event.frame)]
            encoded_frames = [self.encode_image(frame) for frame in init_frames]

        # Setup for LLM-Planner
        seen_objs = self.env.get_visible_objects().split(", ")

        # Get initial high-level plan
        curr_task = {
            "task_instr": [goal_instr],
            "step_instr": step_instrs,
            "vis_objs": ", ".join(seen_objs), 
            "completed_plans": []
        }

        # Format the current task for better readability
        formatted_task = {
            "task_instr": curr_task["task_instr"][0],
            "step_instr": [step for sublist in curr_task["step_instr"] for step in sublist] if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list) else curr_task["step_instr"],
            "vis_objs": curr_task["vis_objs"],
            "completed_plans": curr_task["completed_plans"]
        }
        
        log.debug("Current task:")
        log.debug(f"  Task instruction: {formatted_task['task_instr']}")
        if use_step_instructions:
            log.debug(f"  Step instructions: {formatted_task['step_instr']}")
        else:
            log.debug("  Step instructions: Disabled")
        log.debug(f"  Visible objects: {formatted_task['vis_objs']}")
        log.debug(f"  Completed plans: {formatted_task['completed_plans']}")
        
        # Pass the use_step_instructions flag to generate_gpt_prompt
        init_prompt = self.llm_planner.generate_gpt_prompt(
            curr_task, 
            k=self.config["llm_planner"]["num_in_context_examples"],
            includeLow=use_step_instructions,
            dynamic=dynamic
        )

        # Show the full prompt if debug is enabled
        if self.config["llm_planner"].get("debug", False):
            log.debug("=" * 50)
            log.debug("FULL PROMPT:")
            log.debug(init_prompt)
            log.debug("=" * 50)

        llm_out = self.llm(init_prompt, engine=engine, images=encoded_frames, stop=['\n'], do_train=do_train)
        
        print(f"[DEBUG] LLM output from : {engine} \n{llm_out}")
        
        high_level_plans = self.clean_llm_output(llm_out)
        # print(f"[DEBUG] HLPs from {engine}:\n{high_level_plans}")
        
        # TRAINING 
        if self.trainFinished == False:
            # TODO : Change how the preference works in this case 
            # preference_prompt = init_prompt + " + The output should be more smooth"
            preference_prompt = init_prompt
            # print("Preference Prompt",preference_prompt)
            
            synthetic_llm_out = self.llm(preference_prompt, engine=self.config["llm_planner"]["synthetic_HLP_engine"], images=encoded_frames, stop=['\n'], do_train=do_train)

            synthetic_high_level_plans = self.clean_llm_output(synthetic_llm_out)

            print("[DEBUG] Synthetic HLPs \n", synthetic_high_level_plans)

            if engine=="mistral-7b-instruct-adapted":
                self._train_adapted_mistral(preference_prompt, original_HLP=high_level_plans, synthetic_HLP=synthetic_high_level_plans)
            elif engine=="gemma-2-9b-it-adapted":
                self._train_adapted_gemma_2(preference_prompt, original_HLP=high_level_plans, synthetic_HLP=synthetic_high_level_plans )
            

        initial_high_level_plans = high_level_plans.copy()  # Store the initial plans
        
        # Display the full high-level plan more prominently
        log.debug("=" * 50)
        log.debug("GENERATED HIGH-LEVEL PLAN:")
        for i, plan in enumerate(high_level_plans, 1):
            log.debug(f"  {i}. {plan}")
        log.debug("=" * 50)
        log.debug(f"Visible objects: {seen_objs}")
        
        # Get max retries and max replanning from config
        max_retries = self.config["llm_planner"].get("max_retries", 3)
        max_replanning = self.config["llm_planner"].get("max_replanning", 10)
        
        # Setup counters
        retry_count = 0
        replanning_count = 0  # Track total replanning attempts for this task
        
        # Run until high-level plans are exhausted
        while high_level_plans and replanning_count <= max_replanning:
            plan = high_level_plans.pop(0).strip()
            log.debug(f"Plan: {plan}")
            
            # Try to parse the plan to extract object names
            try:
                plan_parts = plan.split()
                action = plan_parts[0]
                object_name = ' '.join(plan_parts[1:]) if len(plan_parts) > 1 else ""
            except:
                action = plan
                object_name = ""
            
            try:
                action_ret = self.env.llm_skill_interact(plan)
                # Save frame after each action
                frame_history.append(self.env.last_event.frame.copy())
            except Exception as e:
                # Handle assertion errors or other exceptions from Thor connector
                log.warning(f"Error executing '{plan}': {str(e)}")
                if "instruction not supported" in str(e).lower():
                    log.warning(f"Instruction '{plan}' not supported. Skipping and continuing.")
                    # Add to completed plans with a note that it was skipped
                    completed_plans.append(f"{plan} [SKIPPED - UNSUPPORTED]")
                    retry_count = 0  # Reset retry counter
                    continue
                elif "object" in str(e).lower() and "not found" in str(e).lower() and object_name:
                    # Try fuzzy matching if object not found
                    available_objects = self.env.get_visible_objects().split(", ")
                    matched_obj, similarity = self.match_object_name(object_name, available_objects)
                    
                    if matched_obj:
                        # Create a new plan with the matched object name
                        matched_plan = f"{action} {matched_obj}"
                        log.debug(f"Object '{object_name}' not found. Using fuzzy match '{matched_obj}' (similarity: {similarity:.2f})")
                        
                        try:
                            # Try again with the matched object
                            action_ret = self.env.llm_skill_interact(matched_plan)
                            if action_ret['success']:
                                log.debug(f"SUCCESS: '{matched_plan}' executed successfully (fuzzy match)")
                                completed_plans.append(matched_plan)
                                retry_count = 0  # Reset retry counter on success
                                continue
                        except Exception as new_e:
                            log.warning(f"Error executing fuzzy matched plan: {str(new_e)}")
                    else:
                        log.warning(f"No matching object found for '{object_name}'. Best similarity: {similarity:.2f} (threshold: {self.obj_sim_threshold})")
                else:
                    # For other exceptions, re-raise
                    raise

            if not action_ret['success']:
                log.warning(action_ret['message'])
                failed_plans.append({"plan": plan, "error": action_ret['message']})
                
                # Check for unsupported instruction
                if "instruction not supported" in action_ret['message'].lower() or "not supported" in action_ret['message'].lower():
                    log.warning(f"Instruction '{plan}' not supported. Skipping and continuing.")
                    # Add to completed plans with a note that it was skipped
                    completed_plans.append(f"{plan} [SKIPPED - UNSUPPORTED]")
                    retry_count = 0  # Reset retry counter
                    continue
                
                # Check for object not found and try fuzzy matching
                if "object" in action_ret['message'].lower() and "not found" in action_ret['message'].lower() and object_name:
                    # Try fuzzy matching if object not found
                    available_objects = self.env.get_visible_objects().split(", ")
                    matched_obj, similarity = self.match_object_name(object_name, available_objects)
                    
                    if matched_obj:
                        # Create a new plan with the matched object name
                        matched_plan = f"{action} {matched_obj}"
                        log.debug(f"Object '{object_name}' not found. Using fuzzy match '{matched_obj}' (similarity: {similarity:.2f})")
                        
                        try:
                            # Try again with the matched object
                            action_ret = self.env.llm_skill_interact(matched_plan)
                            if action_ret['success']:
                                log.debug(f"SUCCESS: '{matched_plan}' executed successfully (fuzzy match)")
                                completed_plans.append(matched_plan)
                                retry_count = 0  # Reset retry counter on success
                                continue
                        except Exception as new_e:
                            log.warning(f"Error executing fuzzy matched plan: {str(new_e)}")
                    else:
                        log.warning(f"No matching object found for '{object_name}'. Best similarity: {similarity:.2f} (threshold: {self.obj_sim_threshold})")
                
                # Dynamic re-planning if enabled
                if dynamic:
                    # Check if we've exceeded max retries for this action
                    if retry_count >= max_retries:
                        log.warning(f"Exceeded maximum retries ({max_retries}) for this action. Moving on.")
                        retry_count = 0
                        continue
                    
                    # Check if we've exceeded max replanning for this task
                    if replanning_count >= max_replanning:
                        log.warning(f"Exceeded maximum replanning attempts ({max_replanning}) for this task. Continuing with remaining plans.")
                        retry_count = 0
                        continue
                    
                    retry_count += 1
                    replanning_count += 1
                    log.debug(f"Dynamic replanning attempt {retry_count}/{max_retries} for current action (Total: {replanning_count}/{max_replanning})")
                    
                    curr_vis_objs = self.env.get_visible_objects().split(", ")
                    
                    # Add new objects to seen_objs without duplicates
                    for obj in curr_vis_objs:
                        if obj and obj not in seen_objs:  # Check if obj is not empty
                            seen_objs.append(obj)
                    
                    # Sort for consistent output
                    seen_objs.sort()
                    
                    # Update the task with current visible objects and completed plans
                    curr_task = {
                        "task_instr": [goal_instr],
                        "step_instr": step_instrs,
                        "vis_objs": ", ".join(seen_objs),  # Convert back to comma-separated string
                        "completed_plans": completed_plans
                    }
                    
                    # Format the updated task for better readability
                    formatted_task = {
                        "task_instr": curr_task["task_instr"][0],
                        "step_instr": [step for sublist in curr_task["step_instr"] for step in sublist] if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list) else curr_task["step_instr"],
                        "vis_objs": curr_task["vis_objs"],
                        "completed_plans": curr_task["completed_plans"]
                    }
                    
                    log.debug("Updated task for dynamic replanning:")
                    log.debug(f"  Task instruction: {formatted_task['task_instr']}")
                    if use_step_instructions:
                        log.debug(f"  Step instructions: {formatted_task['step_instr']}")
                    else:
                        log.debug("  Step instructions: Disabled")
                    log.debug(f"  Visible objects: {formatted_task['vis_objs']}")
                    log.debug(f"  Completed plans: {formatted_task['completed_plans']}")
                    
                    # Pass the use_step_instructions flag to generate_gpt_prompt
                    new_prompt = self.llm_planner.generate_gpt_prompt(
                        curr_task, 
                        k=self.config["llm_planner"]["num_in_context_examples"],
                        includeLow=use_step_instructions,
                        dynamic=dynamic
                    )

                    # Show the full prompt if debug is enabled
                    if self.config["llm_planner"].get("debug", False):
                        log.debug("=" * 50)
                        log.debug("FULL DYNAMIC REPLANNING PROMPT:")
                        log.debug(new_prompt)
                        log.debug("=" * 50)

                    encoded_frames = []
                    
                    # Get current frame if vision is used
                    if vision:
                        curr_frame = [Image.fromarray(self.env.last_event.frame)]
                        encoded_frames = [self.encode_image(frame) for frame in curr_frame]

                    # Generate new plans for dynamic replanning
                    llm_out = self.llm(new_prompt, engine=engine, images=encoded_frames, stop=['\n'])
                    
                    
                    high_level_plans = self.clean_llm_output(llm_out)

                    # Display the dynamically generated high-level plan more prominently
                    log.debug("=" * 50)
                    log.debug("DYNAMICALLY GENERATED HIGH-LEVEL PLAN:")
                    for i, plan in enumerate(high_level_plans, 1):
                        log.debug(f"  {i}. {plan}")
                    log.debug("=" * 50)
                    log.debug(f"Visible objects: {seen_objs}")
            else:
                log.debug(f"SUCCESS: '{plan}' executed successfully")
                completed_plans.append(plan)
                retry_count = 0  # Reset retry counter on success
        
        # Check if we hit the replanning limit
        if replanning_count >= max_replanning:
            log.warning(f"Task ended because maximum replanning attempts ({max_replanning}) were reached.")
        
        # Check if goal was satisfied
        goal_satisfied = self.env.get_goal_satisfied()
        log.debug('target goal: ' + json.dumps(self.env.task.get_targets()))
        log.debug('success: ' + str(goal_satisfied))
        if goal_satisfied:
            success = True

        # Record results with detailed plan information
        log_entry = {
            'trial': traj_data['task_id'],
            'scene': scene_name,
            'type': traj_data['task_type'],
            'repeat_idx': int(r_idx),
            'goal_instr': goal_instr,
            'initial_high_level_plans': initial_high_level_plans,
            'completed_plans': completed_plans,
            'failed_plans': failed_plans,
            'success': success,
            'frame_history': frame_history
        }

        return log_entry
                
    def run_evaluation(self, dry_run=False, do_train=False):
        """Run evaluation on all tasks"""
        results = []
        save_path = self.config["out_dir"]
        
        # Create the output directory if it doesn't exist
        if self.config.get("save_results", False):
            save_path = os.path.join(self.config["out_dir"], "results")
            os.makedirs(save_path, exist_ok=True)

        if dry_run:
            log.info(f"Dry run mode enabled. Only evaluating {self.dry_run_samples} task(s).")
            self.tasks = self.tasks[:self.dry_run_samples]
        
        # Configuration for adaptive training
        
        is_adapted_mistral = self.config["llm_planner"]["engine"] == "mistral-7b-instruct-adapted"
        is_adapted_gemma_2 = self.config["llm_planner"]["engine"] == "gemma-2-9b-it-adapted"
        start = time.time()
        if (is_adapted_mistral or is_adapted_gemma_2) and do_train:
            log.info(f"Training mode enabled: Will train on first {self.training_samples} samples, then evaluate on remaining samples")
            start = time.time()
            self.trainFinished = False 
            self.train_tasks = self.tasks[:self.training_samples]
            self.evaluation_tasks = self.tasks[self.training_samples:]

            

            # Training 
            for task_idx, task in tqdm(enumerate(self.train_tasks),
                                       desc="Training adapted layers"):
                try:
                    log.debug(task)
                    traj_data = load_task_json(task)
                    r_idx = task['repeat_idx']
                    log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")

                    result = self.evaluate_task(
                        self.config["llm_planner"]["engine"],
                        traj_data, 
                        r_idx,
                        dynamic=self.config["llm_planner"]["dynamic"],
                        vision=self.config["llm_planner"]["vision"],
                        do_train=do_train         
                    )
                
                except Exception as e: 
                    print(e)

            self.trainFinished = True 
            print("[TRAINING] Finished")

            # Evaluation after training the adapted layers
            for task_idx, task in tqdm(enumerate(self.evaluation_tasks), 
                                    desc="Tasks"):
                try:
                    log.debug(task)
                    traj_data = load_task_json(task)
                    r_idx = task['repeat_idx']
                    log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")
                    
                    result = self.evaluate_task(
                        self.config["llm_planner"]["engine"],
                        traj_data, 
                        r_idx,
                        dynamic=self.config["llm_planner"]["dynamic"],
                        vision=self.config["llm_planner"]["vision"]          
                    )
                    results.append(result)
                    
                    # Save result for debugging if enabled
                    if self.config.get("save_results", False):
                        self.save_result(result, save_path)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.error(f"Error processing task {task_idx+1}/{len(self.tasks)}: {repr(e)}")
                    
                    # Create a failure result to keep tracking progress
                    try:
                        failure_result = {
                            'trial': task['task_id'] if 'task_id' in task else f"unknown_task_{task_idx}",
                            'scene': f"scene_{task_idx}",
                            'type': "failed",
                            'repeat_idx': task['repeat_idx'] if 'repeat_idx' in task else 0,
                            'goal_instr': "Task processing failed with exception",
                            'inferred_steps': [f"ERROR: {repr(e)}"],
                            'success': False
                        }
                        results.append(failure_result)
                        
                        # Save failure result if saving is enabled
                        if self.config.get("save_results", False):
                            self.save_result(failure_result, save_path)
                    except:
                        # If we can't even create a failure result, just continue
                        log.error("Failed to create failure result entry")

        else: # Normal Inference 
             
            for task_idx, task in tqdm(enumerate(self.tasks), 
                                  desc="Tasks"):
                try:
                    log.debug(task)
                    traj_data = load_task_json(task)
                    r_idx = task['repeat_idx']
                    log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")
                    
                    result = self.evaluate_task(
                        self.config["llm_planner"]["engine"],
                        traj_data, 
                        r_idx,
                        dynamic=self.config["llm_planner"]["dynamic"],
                        vision=self.config["llm_planner"]["vision"]          
                    )
                    results.append(result)
                    
                    # Save result for debugging if enabled
                    if self.config.get("save_results", False):
                        self.save_result(result, save_path)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.error(f"Error processing task {task_idx+1}/{len(self.tasks)}: {repr(e)}")
                    
                    # Create a failure result to keep tracking progress
                    try:
                        failure_result = {
                            'trial': task['task_id'] if 'task_id' in task else f"unknown_task_{task_idx}",
                            'scene': f"scene_{task_idx}",
                            'type': "failed",
                            'repeat_idx': task['repeat_idx'] if 'repeat_idx' in task else 0,
                            'goal_instr': "Task processing failed with exception",
                            'inferred_steps': [f"ERROR: {repr(e)}"],
                            'success': False
                        }
                        results.append(failure_result)
                        
                        # Save failure result if saving is enabled
                        if self.config.get("save_results", False):
                            self.save_result(failure_result, save_path)
                    except:
                        # If we can't even create a failure result, just continue
                        log.error("Failed to create failure result entry")
                
        
        # Print results
        self._print_results(results, start)
        return results

    def _print_results(self, results, start_time):
        """Print evaluation results"""
        n = len(results)
        
        if n == 0:
            log.warning("No results collected - all tasks may have failed with exceptions")
            log.debug(f'success rate: 0.00 % (0/0)')
            log.debug(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start_time)))}')
            log.debug('------------------------')
            log.debug(yaml.dump(self.config))
            return
        
        n_success = sum(1 for e in results if e['success'])
        
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        print(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start_time)))}')
        log.info('------------------------')
        log.info(yaml.dump(self.config))
    
    def save_result(self, result_dict, base_path):
        """Save result for debugging with complete trajectory"""
        # Create task-specific directory
        task_dir = os.path.join(base_path, f"{result_dict['trial']}_{result_dict['repeat_idx']}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Save result as JSON (excluding frame_history to avoid huge files)
        result_for_json = {k: v for k, v in result_dict.items() if k != 'frame_history'}
        with open(os.path.join(task_dir, "result.json"), "w") as f:
            json.dump(result_for_json, f, indent=2)
        
        # Save all frames from the trajectory
        for i, frame in enumerate(result_dict['frame_history']):
            # Create PIL image
            img = Image.fromarray(frame)
            
            # Add text for frame number and success status
            draw = ImageDraw.Draw(img)
            text = f"Frame {i}"
            if i == len(result_dict['frame_history']) - 1:  # Last frame
                success_str = 'SUCCESS' if result_dict['success'] else 'FAIL'
                text += f" ({success_str})"
            
            # Add the text to the image
            draw.text((10, 10), text, fill=(255, 255, 255))
            
            # Save the frame
            img.save(os.path.join(task_dir, f"frame_{i:04d}.png"))
        
        # Save detailed plan information
        with open(os.path.join(task_dir, "plans.txt"), "w") as f:
            # Task description
            f.write(f"Task: {result_dict['goal_instr']}\n\n")
            
            # Initial high-level plans
            f.write("Initial High-Level Plans:\n")
            for i, plan in enumerate(result_dict['initial_high_level_plans'], 1):
                f.write(f"{i}. {plan}\n")
            f.write("\n")
            
            # Completed plans
            f.write("Completed Plans:\n")
            if result_dict['completed_plans']:
                for i, plan in enumerate(result_dict['completed_plans'], 1):
                    f.write(f"{i}. {plan}\n")
            else:
                f.write("None\n")
            f.write("\n")
            
            # Failed plans (only if there are any)
            if result_dict['failed_plans']:
                f.write("Failed Plans:\n")
                for i, plan_data in enumerate(result_dict['failed_plans'], 1):
                    f.write(f"{i}. {plan_data['plan']}\n")
                    f.write(f"   Error: {plan_data['error']}\n")
                f.write("\n")
            
            # Final result
            f.write(f"Final Status: {'SUCCESS' if result_dict['success'] else 'FAILURE'}")
    
    def preprocess_dataset(self):
        """Preprocess dataset if needed"""
        args_dict = self.config["alfred"]["env_args"]
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None  
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(self.splits)

    def clean_llm_output(self, llm_out, DEBUG=False):
        """
        Clean the LLM output by removing the 'Next Plans:' prefix and converting to a comma-separated list.
        
        Args:
            llm_out (str): Raw LLM output string like "Next Plans: Navigation countertop, OpenObject microwave, ..."
            
        Returns:
            list: List of cleaned plan steps
        """
        # Remove "Next Plans:" prefix if present
        if self.config["llm_planner"]["engine"] in ["gpt-4o-mini","gpt-4o"] and "Next Plans:" in llm_out:
            cleaned_text = llm_out.split("Next Plans:")[1].strip()
        elif self.config["llm_planner"]["engine"] in ["mistral-7b-instruct", "mistral-7b-instruct-adapted"]:
            try:
                cleaned_text = llm_out.split("[/INST]")[1].strip()
            except IndexError as e:
                print("Error at cleaning mistral output  :", e)
                cleaned_text = llm_out.strip()
        
        elif self.config["llm_planner"]["engine"] == "gemma-2-9b-it":
            try:
                cleaned_text = llm_out.split("Next Plans: ")[-1].split("<end_of_turn>")[0].strip()
            except Exception as e:
                cleaned_text = llm_out.strip()
        
        elif self.config["llm_planner"]["engine"] == "gemma-2-9b-it-adapted":
            if "<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>" in llm_out:
                cleaned_text = ""
            
        else: # Qwen output, Gemma Output, Pixtral Output
            cleaned_text = llm_out.strip()
        
        if DEBUG==True:
            print(f"[DEBUG] Cleaned text from {self.config['llm_planner']['engine']}:\n", cleaned_text)
        # Split by comma and strip whitespace from each item
        plans = [plan.strip() for plan in cleaned_text.split(',')]
        return plans

    def match_object_name(self, generated_name, available_objects):
        """
        Match a generated object name to available objects using sentence embeddings.
        
        Args:
            generated_name: Object name generated by the LLM
            available_objects: List of available objects in the environment
            
        Returns:
            Tuple of (matched_object, similarity_score) or (None, 0) if no match above threshold
        """
        if not generated_name or not available_objects:
            return None, 0
            
        # Try direct matching first (case insensitive)
        generated_name_lower = generated_name.lower()
        for obj in available_objects:
            if obj.lower() == generated_name_lower:
                return obj, 1.0  # Perfect match
                
        # If no direct match, use sentence embeddings for fuzzy matching
        try:
            # Encode the generated name
            generated_embedding = self.obj_encoder.encode(generated_name, convert_to_tensor=True, show_progress_bar=False)
            
            # Encode all available objects
            available_embeddings = self.obj_encoder.encode(available_objects, convert_to_tensor=True, show_progress_bar=False)
            
            # Calculate similarity scores
            similarities = cos_sim(generated_embedding, available_embeddings)[0]
            
            # Find the best match above threshold
            best_match_idx = similarities.argmax().item()
            best_match_score = similarities[best_match_idx].item()
            
            # Check if the best match exceeds the threshold
            if best_match_score >= self.obj_sim_threshold:
                return available_objects[best_match_idx], best_match_score
                
            return None, best_match_score
            
        except Exception as e:
            log.warning(f"Error during fuzzy object matching: {str(e)}")
            return None, 0

    def _train_mistral_online(self, input_messages, ground_truth_response):
        """
        Perform online training of Mistral model using ground truth from GPT-4o mini.
        
        Args:
            input_messages: Original input messages
            ground_truth_response: Ground truth response from GPT-4o mini
        """
        try:
            # Proactively clear memory
            gc.collect()
            torch.cuda.empty_cache()

            # Set model to training mode
            self.mistral_model.train()
            
            # Prepare the full conversation with ground truth as target
            full_conversation = input_messages + [{"role": "assistant", "content": ground_truth_response}]
            
            # Tokenize the full conversation
            tokenized = self.mistral_tokenizer.apply_chat_template(
                full_conversation,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=False  # We want the full conversation including response
            ).to(self.mistral_model.device)
            
            # Prepare input and target
            input_ids = tokenized
            target_ids = tokenized.clone()
            
            # Mask the input tokens in the target (we only want to compute loss on the response)
            # Find where the assistant response starts
            assistant_start_tokens = self.mistral_tokenizer.encode("assistant", add_special_tokens=False)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Forward pass
            with torch.amp.autocast():  # Use mixed precision for efficiency
                outputs = self.mistral_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids
                )
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.mistral_model.parameters(), max_norm=1.0)
            
            # Update parameters using optimizer
            self.mistral_optimizer.step()
            
            # Zero gradients for next iteration
            self.mistral_optimizer.zero_grad()
            
            # Set model back to eval mode
            self.mistral_model.eval()

            # Checkpoint saving logic
            self.training_step += 1
            if self.training_step % self.save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, "mistral", f"step_{self.training_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model and optimizer
                self.mistral_model.save_pretrained(checkpoint_path)
                torch.save(self.mistral_optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
            
        except Exception as e:
            # Ensure model is back in eval mode even if training fails
            self.mistral_model.eval()
            # Clear any accumulated gradients
            self.mistral_optimizer.zero_grad()
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def _train_llama_online(self, input_messages, ground_truth_response, image=None):
        """
        Perform online training of Llama Vision model using ground truth from GPT-4o mini.
        
        Args:
            input_messages: Original input messages
            ground_truth_response: Ground truth response from GPT-4o mini
            image: PIL Image object if vision input is provided
        """
        try:
            # Proactively clear memory
            gc.collect()
            torch.cuda.empty_cache()

            # Set model to training mode
            self.llama_model.train()

            # Freeze all the model parameters except Adapter Weights for model 

            
            # Prepare the full conversation with ground truth as target
            full_conversation = input_messages + [{"role": "assistant", "content": ground_truth_response}]
            
            # Process inputs for training
            input_text = self.llama_processor.apply_chat_template(
                full_conversation, 
                add_generation_prompt=False  # We want the full conversation including response
            )
            
            # Process both image and text
            inputs = self.llama_processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.llama_model.device)
            
            # Create labels (same as input_ids for causal LM)
            labels = inputs['input_ids'].clone()
            
            # Find the assistant response start to mask previous tokens
            # We only want to compute loss on the assistant's response
            input_text_tokens = self.llama_processor.tokenizer.encode(
                self.llama_processor.apply_chat_template(input_messages, add_generation_prompt=True),
                add_special_tokens=False
            )
            assistant_start_idx = len(input_text_tokens)
            
            # Mask tokens before assistant response (set to -100 to ignore in loss)
            if assistant_start_idx < labels.shape[1]:
                labels[:, :assistant_start_idx] = -100
            
            # Forward pass
            with torch.amp.autocast():  # Use mixed precision for efficiency
                outputs = self.llama_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    pixel_values=inputs.get('pixel_values'),  # Vision input
                    aspect_ratio_ids=inputs.get('aspect_ratio_ids'),  # Llama-specific
                    aspect_ratio_mask=inputs.get('aspect_ratio_mask'),  # Llama-specific
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.llama_model.parameters(), max_norm=1.0)
            
            # Update parameters using optimizer
            self.llama_optimizer.step()
            
            # Zero gradients for next iteration
            self.llama_optimizer.zero_grad()
            
            # Set model back to eval mode
            self.llama_model.eval()

            # Checkpoint saving logic
            self.training_step += 1
            if self.training_step % self.save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, "llama", f"step_{self.training_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model and optimizer
                self.llama_model.save_pretrained(checkpoint_path)
                torch.save(self.llama_optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
            
        except Exception as e:
            # Ensure model is back in eval mode even if training fails
            self.llama_model.eval()
            # Clear any accumulated gradients
            self.llama_optimizer.zero_grad()
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e


    def _train_adapted_mistral(self, original_prompt, original_HLP, synthetic_HLP):
        """
        Perform online training of adapted Mistral model to align HLPs.
        
        Args:
            original_prompt: The original input prompt that generated the HLP
            original_HLP: HLP generated by the adapted model (for comparison/logging)
            synthetic_HLP: More preference aligned HLP (target for training)
        """
        try:
            # Proactively clear memory
            gc.collect()
            torch.cuda.empty_cache()

            # Set model to training mode
            self.mistral_model.train()

            # Ensure only adapter layers are trainable (if using LoRA/adapter approach)
            for name, param in self.mistral_model.named_parameters():
                if 'adapt' not in name.lower():
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.mistral_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.mistral_model.parameters())
            
            print("Trainable layers:")
            for name, param in self.mistral_model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.numel():,} parameters")

            print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
            percent = (trainable_params / total_params) * 100 if total_params > 0 else 0
            print(f"Trainable parameters percentage: ({percent:.2f}%)")

            # Zero gradients before training
            self.mistral_optimizer.zero_grad()
            
            # Convert HLP lists to formatted strings
            original_HLP_str = self._format_hlp_for_training(original_HLP)
            synthetic_HLP_str = self._format_hlp_for_training(synthetic_HLP)
            
            # Prepare the training conversation using the synthetic HLP as target
            training_messages = [
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": synthetic_HLP_str}
            ]
            
            
            # Tokenize the full conversation
            tokenized = self.mistral_tokenizer.apply_chat_template(
                training_messages,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=False
            ).to(self.mistral_model.device)
            
            # Prepare input and target
            input_ids = tokenized
            target_ids = tokenized.clone()
            
            # Find where the assistant response starts to mask input tokens
            user_input_tokens = self.mistral_tokenizer.apply_chat_template(
                [{"role": "user", "content": original_prompt}],
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.mistral_model.device)
            
            # Mask tokens before assistant response (set to -100 to ignore in loss)
            mask_length = user_input_tokens.shape[1]
            if mask_length < target_ids.shape[1]:
                target_ids[:, :mask_length] = -100
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)

            print("before passing through mistral model ")
            # Forward pass with mixed precision
            # with torch.amp.autocast(device_type="cuda"):
            outputs = self.mistral_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids, 
                
            )
            # print('output:', target_ids)
            loss = outputs.loss
                # print(loss)
            
            print(f"[TRAINING] Loss: {loss.item():.4f}")
            
            # Store training metrics
            training_metrics = {
                "step": self.training_step + 1,
                "loss": loss.item(),
                "original_prompt": original_prompt,
                "original_hlp": original_HLP_str,
                "target_hlp": synthetic_HLP_str,
                "original_hlp_list": original_HLP,
                "target_hlp_list": synthetic_HLP
            }
            
            # Save training metrics to file
            self._save_training_metrics(training_metrics)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.mistral_model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            # Update parameters using optimizer
            self.mistral_optimizer.step()
            
            # Zero gradients for next iteration
            self.mistral_optimizer.zero_grad()
            
            # Set model back to eval mode
            self.mistral_model.eval()

            # Checkpoint saving logic
            self.training_step += 1
            if self.training_step % self.save_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "mistral_adapted", f"step_{self.training_step}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model and optimizer
                self.mistral_model.save_pretrained(checkpoint_path)
                torch.save(
                    self.mistral_optimizer.state_dict(), 
                    os.path.join(checkpoint_path, "optimizer.pt")
                )
                
                log.info(f"Checkpoint saved at step {self.training_step}")
            
        except Exception as e:
            # Ensure model is back in eval mode even if training fails
            self.mistral_model.eval()
            # Clear any accumulated gradients
            self.mistral_optimizer.zero_grad()
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.error(f"Training failed: {str(e)}")
            raise e

    def _train_adapted_gemma_2(self, original_prompt, original_HLP, synthetic_HLP):
        """
        Perform online training of adapted Gemma2 model to align HLPs.

        Args:
            original_prompt: The original input prompt that generated the HLP
            original_HLP: HLP generated by the adapted model (for comparison/logging)
            synthetic_HLP: More preference-aligned HLP (target for training)
        """
        try: 
            import gc
            import torch.nn.functional as F

            # Proactively clear memory
            gc.collect()
            torch.cuda.empty_cache()

            # Set model to training mode 
            self.gemma2_model.train()

            # Freeze all except adapter layers
            for name, param in self.gemma2_model.named_parameters():
                param.requires_grad = 'adapt' in name.lower()

            # Print trainable layers info
            trainable_params = sum(p.numel() for p in self.gemma2_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.gemma2_model.parameters())
            print(f"[Trainable] {trainable_params:,} / {total_params:,} ({(trainable_params / total_params)*100:.2f}%)")

            for name, param in self.gemma2_model.named_parameters():
                if param.requires_grad:
                    print(f"  ✓ {name}: {param.numel():,}")

            # Prepare training inputs
            prompt = original_prompt
            target = self._format_hlp_for_training(synthetic_HLP)  # This is the desired response

            # Tokenize
            input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target, return_tensors="pt").to(self.device)["input_ids"]

            # Forward pass
            outputs = self.gemma2_model(**input, labels=labels)
            loss = outputs.loss

            print(f"[Train Loss] {loss.item():.4f}")

            # Backpropagate only through adapters
            loss.backward()
            self.gemma2_optimizer.step()
            self.gemma2_optimizer.zero_grad()

        except Exception as e:
            print("[Error during adapter fine-tuning]:", e)


    def _format_hlp_for_training(self, hlp_list):
        """
        Convert HLP list to formatted string for training.
        
        Args:
            hlp_list: List of HLP actions
            
        Returns:
            Formatted string representation of HLP
        """
        if isinstance(hlp_list, list):
            # Convert list to string format - you can customize this based on your needs
            return str(hlp_list)
        return hlp_list

    def _save_final_adapted_model(self):
        """Save the final adapted model after training phase completion."""
        try:
            final_model_path = os.path.join(self.checkpoint_dir, "mistral_adapted_final")
            os.makedirs(final_model_path, exist_ok=True)
            
            # Save the final model
            self.mistral_model.save_pretrained(final_model_path)
            torch.save(self.mistral_optimizer.state_dict(), os.path.join(final_model_path, "optimizer.pt"))
            
            # Save training metadata
            training_metadata = {
                "total_training_steps": self.training_step,
                "model_type": "mistral-7b-instruct-adapted",
                "training_completed": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(os.path.join(final_model_path, "training_metadata.json"), "w") as f:
                json.dump(training_metadata, f, indent=2)
            
            log.info(f"Final adapted model saved to: {final_model_path}")
            log.info(f"Total training steps completed: {self.training_step}")
            
            # Analyze and log training progress
            self._analyze_training_progress()
            
        except Exception as e:
            log.error(f"Failed to save final adapted model: {str(e)}")

    def _measure_hlp_alignment(self, predicted_hlp, aligned_hlp):
        """
        Measure the alignment between predicted and target HLPs using semantic similarity.
        
        Args:
            predicted_hlp: HLP generated by current model
            aligned_hlp: Target aligned HLP from GPT-4o mini
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Clean the HLPs for comparison
            predicted_clean = self.clean_llm_output(predicted_hlp)
            aligned_clean = self.clean_llm_output(aligned_hlp)
            
            # Convert to strings for embedding
            predicted_str = ', '.join(predicted_clean)
            aligned_str = ', '.join(aligned_clean)
            
            # Calculate semantic similarity using the sentence transformer
            pred_embedding = self.obj_encoder.encode(predicted_str, convert_to_tensor=True, show_progress_bar=False)
            aligned_embedding = self.obj_encoder.encode(aligned_str, convert_to_tensor=True, show_progress_bar=False)
            
            # Calculate cosine similarity
            similarity = cos_sim(pred_embedding, aligned_embedding).item()
            
            return similarity
            
        except Exception as e:
            log.warning(f"Failed to measure HLP alignment: {str(e)}")
            return 0.0

    def _save_training_metrics(self, metrics):
        """
        Save training metrics to a JSON file for progress tracking.
        
        Args:
            metrics: Dictionary containing training metrics
        """
        try:
            metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.jsonl")
            
            # Append metrics to file (JSONL format)
            with open(metrics_file, 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
                
        except Exception as e:
            log.warning(f"Failed to save training metrics: {str(e)}")

    def _analyze_training_progress(self):
        """
        Analyze and log training progress from saved metrics.
        """
        try:
            metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.jsonl")
            
            if not os.path.exists(metrics_file):
                return
                
            # Load all metrics
            metrics = []
            with open(metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line.strip()))
            
            if not metrics:
                return
                
            # Calculate progress statistics
            latest_metrics = metrics[-1]
            first_metrics = metrics[0]
            
            # Calculate averages for recent steps
            recent_steps = min(10, len(metrics))
            recent_metrics = metrics[-recent_steps:]
            
            avg_loss = sum(m['loss'] for m in recent_metrics) / len(recent_metrics)
            avg_alignment = sum(m['alignment_score'] for m in recent_metrics) / len(recent_metrics)
            
            # Log progress summary
            log.info(f"Training Progress Summary:")
            log.info(f"  Total steps: {len(metrics)}")
            log.info(f"  Latest loss: {latest_metrics['loss']:.4f}")
            log.info(f"  Latest alignment: {latest_metrics['alignment_score']:.4f}")
            log.info(f"  Average loss (last {recent_steps} steps): {avg_loss:.4f}")
            log.info(f"  Average alignment (last {recent_steps} steps): {avg_alignment:.4f}")
            log.info(f"  Alignment improvement: {latest_metrics['alignment_score'] - first_metrics['alignment_score']:.4f}")
            
        except Exception as e:
            log.warning(f"Failed to analyze training progress: {str(e)}")

def main():

    openai_config_file = "config/openai_config.yaml"
    with open(openai_config_file) as reader:
        openai_config = yaml.safe_load(reader)
        OPENAI_API_KEY = openai_config["OPENAI_API_KEY"]
        
        # Set environment variable 
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_alfred.yaml')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AlfredEvaluator(args.config)
    
    print("Args, config\n", args.config)
    # Run preprocess step only once on new installation
    evaluator.preprocess_dataset()
    
    # Run evaluation
    results = evaluator.run_evaluation(dry_run=args.dry_run, do_train=True)
    
    # Save results if configured
    if evaluator.config.get("save_results", False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(evaluator.config["out_dir"], f"results_{timestamp}.json")
        
        # Create JSON-serializable version of results (without frame_history)
        json_results = []
        for result in results:
            # Create a copy without frame_history
            json_result = {k: v for k, v in result.items() if k != 'frame_history'}
            json_results.append(json_result)
            
        with open(result_file, 'w') as f:
            json.dump(json_results, f, indent=2)

if __name__ == '__main__':
    main()