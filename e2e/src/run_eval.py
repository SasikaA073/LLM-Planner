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

from alfred.thor_connector import ThorConnector
from alfred.utils import dotdict, load_task_json
from alfred.data.preprocess import Dataset
from hlp_planner import LLM_Planner

sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

# Configure the root logger to print to console
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a logger for this module
log = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class AlfredEvaluator:
    def __init__(self, config_file):
        print("🚀 Initializing ALFRED Evaluator...")
        
        # Load configuration
        print("📝 Loading configuration file...")
        with open(config_file) as reader:
            self.config = yaml.safe_load(reader)
        print(f"✅ Configuration loaded! Engine: {self.config['llm_planner']['engine']}")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize LLM planner
        print("🧠 Initializing LLM planner...")
        self.llm_planner = LLM_Planner(
            knn_data_path=self.config["llm_planner"]["knn_dataset_path"], 
            emb_model_name=self.config["llm_planner"]["emb_model_name"], 
            debug=self.config["llm_planner"]["debug"]
        )
        print("✅ LLM planner initialized!")
        
        # Initialize environment
        print("🏠 Initializing THOR environment...")
        self.env = ThorConnector(x_display=self.config["alfred"]["x_display"])
        print("✅ Environment initialized!")
        
        # Load task splits
        print("📋 Loading task splits...")
        with open(self.config["alfred"]["splits"]) as f:
            self.splits = json.load(f)
        print("✅ Task splits loaded!")
            
        # Prepare tasks
        print("⚙️ Preparing tasks...")
        self.tasks = self._prepare_tasks()
        print(f"✅ {len(self.tasks)} tasks prepared!")
        
        # Initialize the sentence transformer for object name matching
        print("🔍 Initializing sentence transformer for object matching...")
        self.obj_encoder = SentenceTransformer(self.config["llm_planner"]["emb_model_name"])
        self.obj_sim_threshold = self.config["llm_planner"].get("obj_sim_threshold", 0.8)  # Default similarity threshold
        print("✅ Sentence transformer ready!")
        
        # Initialize Llama model if specified in config
        if self.config["llm_planner"]["engine"] == "llama-3-vision":
            print("🦙 Initializing Llama-3 Vision model...")
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            print(f"📥 Loading model: {model_id}")
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cuda",  # Changed from "cpu" to "auto" for better performance
            )
            print("📥 Loading processor...")
            self.llama_processor = AutoProcessor.from_pretrained(model_id)
            print("✅ Llama-3 Vision model ready!")

        # Initialize Mistral model if specified in config
        if self.config["llm_planner"]["engine"] == "mistral-7b-instruct":
            print(" Intializing Mistral-7B-Instruct model...")
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            print(f"📥 Loading model: {model_id}")

            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            print("📥 Loading processor...")
            self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_id)
            print("✅ Mistral-7B-Instruct model ready!")

        else:
            self.llama_model = None
            self.llama_processor = None
            print("💡 Using OpenAI models - no local model loading needed!")
        
        print("🎉 ALFRED Evaluator initialization complete!")
        
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
    
    def llm(self, prompt, engine, images=None, stop=["\n"]):
        """Interface to LLM models"""
        print(f"🤖 LLM call started with engine: {engine}")

        if engine == 'gpt-4o-mini' or engine == 'gpt-4o':
            print("🔄 Using OpenAI API...")
            # Create the base message content
            message_content = []
            
            # Add image if provided
            if images and len(images) > 0:
                print("🖼️ Adding image to OpenAI request...")
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
            print("📡 Making OpenAI API call...")
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
            print("✅ OpenAI response received!")
            return response.choices[0].message.content
        
        elif engine == "llama-3-vision":
            print("🦙 Using Llama-3 Vision model...")
            # Convert base64 image back to PIL Image if provided
            image = None
            if images and len(images) > 0:
                print("🖼️ Converting base64 image to PIL...")
                image_data = base64.b64decode(images[0])
                image = Image.open(BytesIO(image_data))
                print(f"✅ Image loaded: {image.size}")
            else:
                print("📝 No image provided - text-only mode")
            
            # Format messages for Llama
            print("💬 Formatting messages for Llama...")
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
            print("⚡ Generating response with Llama model...")
            with torch.no_grad():
                print("🔄 Applying chat template...")
                input_text = self.llama_processor.apply_chat_template(messages, add_generation_prompt=True)
                print("🔄 Processing inputs...")
                inputs = self.llama_processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.llama_model.device)
                
                print("🎯 Generating text...")
                output = self.llama_model.generate(**inputs, max_new_tokens=300, 
                                                   temperature=1.0,
                                                   do_sample=False # <- important for greedy decoding
                )
                # Decode only the new tokens (skip the input)
                print("📝 Decoding response...")
                generated_text = self.llama_processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print("✅ Llama response generated!")
                return generated_text.strip()
        
        elif engine == "mistral-7b-instruct":
            print("🦙 Using Mistral-7B-Instruct model...")

            # Mistral is a text-only model — image input will be ignored
            if images and len(images) > 0:
                print("📝 Images provided, but Mistral-7B-Instruct only supports text. Proceeding in text-only mode.")
            else:
                print("📝 No image provided - proceeding in text-only mode.")

            # Prepare messages in plain string format
            print("💬 Formatting messages for Mistral...")
            messages = [
                {"role": "user", "content": prompt}
            ]

            # Ensure pad_token is set
            if self.mistral_tokenizer.pad_token is None:
                print("⚠️ Setting pad_token to eos_token")
                self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token


            # Generate response
            print("⚡ Generating response with Mistral model...")
            with torch.no_grad():
                print("🔄 Applying chat template...")
                inputs = self.mistral_tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True
                ).to(self.mistral_model.device)

                print("🎯 Generating text...")
                generated_ids = self.mistral_model.generate(
                    inputs,
                    pad_token_id=self.mistral_tokenizer.eos_token_id,
                    max_new_tokens=512,         # Reduce if OOM occurs
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95
                )

                print("📝 Decoding response...")
                decoded_output = self.mistral_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                print("✅ Mistral response generated!")
                return decoded_output[0].strip()

        
        else:
            print(f"❌ Engine {engine} is not supported!")
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

    def evaluate_task(self, engine, traj_data, r_idx, dynamic=False, to_print=True, vision=False, ob=''):
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

        llm_out = self.llm(init_prompt, engine=engine, images=encoded_frames, stop=['\n'])
        high_level_plans = self.clean_llm_output(llm_out)
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
                
    def run_evaluation(self, dry_run=False):
        """Run evaluation on all tasks"""
        results = []
        save_path = self.config["out_dir"]
        
        # Create the output directory if it doesn't exist
        if self.config.get("save_results", False):
            save_path = os.path.join(self.config["out_dir"], "results")
            os.makedirs(save_path, exist_ok=True)

        if dry_run:
            log.info("Dry run mode enabled. Only evaluating 3 task.")
            self.tasks = self.tasks[:3]
        
        # Main eval loop
        start = time.time()
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
            vocab = None  # todo
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(self.splits)

    def clean_llm_output(self, llm_out):
        """
        Clean the LLM output by removing the 'Next Plans:' prefix and converting to a comma-separated list.
        
        Args:
            llm_out (str): Raw LLM output string like "Next Plans: Navigation countertop, OpenObject microwave, ..."
            
        Returns:
            list: List of cleaned plan steps
        """
        # Remove "Next Plans:" prefix if present
        if "Next Plans:" in llm_out:
            cleaned_text = llm_out.split("Next Plans:")[1].strip()
        else:
            cleaned_text = llm_out.strip()
        
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

def main():
    print("🎬 Starting ALFRED Evaluation Script...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_alfred.yaml')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    print(f"📁 Using config file: {args.config}")
    if args.dry_run:
        print("🧪 Running in DRY RUN mode (limited tasks)")

    # Initialize evaluator
    print("🔧 Creating evaluator instance...")
    evaluator = AlfredEvaluator(args.config)
    
    # Run preprocess step only once on new installation
    print("⚙️ Checking dataset preprocessing...")
    evaluator.preprocess_dataset()
    
    # Run evaluation
    print("🏃 Starting task evaluation...")
    results = evaluator.run_evaluation(dry_run=args.dry_run)
    
    # Save results if configured
    if evaluator.config.get("save_results", False):
        print("💾 Saving results to file...")
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
        print(f"✅ Results saved to {result_file}")
    
    print("🏁 Evaluation complete!")

if __name__ == '__main__':
    main()





