import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from runner.evaluate import evaluate_model
from tqdm import tqdm



def freeze_model_except_adapters(model):
    for name, param in model.named_parameters():
        # print("Adapter? ",name)
        # if "Sequential" in name or "adapter" or "model.layers.27.input_layernorm" in name:
        if "Sequential" in name or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # print number of trainable parameters here.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    all_params = sum(p.numel() for p in model.parameters())

    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {trainable_params/all_params*100}")

def train_model(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    if config['models'][0]['name'] == 'qwen':
        train_model_qwen(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence-large': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'mistral-7b-instruct':
        train_model_mistral(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)


def train_model_florence(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    loader = dataset['train']
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):  
        # TODO: Adapter train mode
        # linear_layer.train()
        total_train_loss = 0  

        for idx,batch in enumerate(loader):
            # break
            inputs, answers = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            image_features = model._encode_image(inputs['pixel_values']).to(device)
            input_embeds = model.get_input_embeddings()(inputs['input_ids']).to(device)
            input_embeds, attention_mask = model._merge_input_ids_with_image_features(input_embeds, image_features)

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"] 
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to('cuda')
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

            # Compute loss
            # loss = loss_criterion(outputs.logits.view(-1, outputs.logits.size(-1)),labels.view(-1))
            loss = outputs.loss
            # print('loss',loss)
            # print("NAN: ",torch.isnan(outputs.logits).any())
            # print("NAN: ",outputs.logits)  # Should be False
            # print("NAN: ",torch.isnan(labels).any())  # Should be False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()  
            # wandb.log({
            # "train_loss": loss.item(),
            # "batch_id": idx
            # })
            if idx == 50:
                break
        avg_train_loss = total_train_loss / 50 
        print(avg_train_loss)

        # Validation Loop
        # Adapter validation 
        # linear_layer.eval()
        if epoch % 20 == 0:
            evaluate_model(model, config, dataset,processor)
            # print(f"Accuracy: {acc:.4f}")



def train_model_qwen(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = dataset['train']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        idx =1
        pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}")
        for batch in pbar:
            inputs = preprocess_fn(batch) if preprocess_fn else batch
            labels = batch["labels"].to(device)
            # print(labels.shape)

            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            # refs = inputs.pop('ref')
            outputs = model(**inputs)
            loss = outputs.loss
            # print(outputs)
            # loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss/idx:.4f}")
            idx+=1

            if idx==50:
                break

        if epoch%20 == 0:
            acc = evaluate_model(model, config, dataset,processor)
            print(f"Accuracy: {acc:.4f}")


        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")

# --------------
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_and_tokenize_dataset(dataset_hf, tokenizer, max_seq_length):
    """
    Formats the conversational history and tokenizes it for Mistral instruction fine-tuning.
    Masks prompt tokens in the labels.

    Args:
        dataset_hf (datasets.Dataset): The Hugging Face dataset.
                                       Expected to have a 'history' column.
        tokenizer: The tokenizer.
        max_seq_length (int): Maximum sequence length for truncation.

    Returns:
        dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    processed_examples = {'input_ids': [], 'attention_mask': [], 'labels': []}

    # Determine BOS and EOS tokens from the tokenizer
    bos = tokenizer.bos_token if tokenizer.bos_token else "<s>"
    eos = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    inst_open = "[INST]"
    inst_close = "[/INST]"

    logger.info(f"Using BOS: '{bos}', EOS: '{eos}' for formatting.")

    for item_idx, item_history in enumerate(dataset_hf['history']):
        full_concatenated_input_ids = []
        full_concatenated_labels = []

        if not isinstance(item_history, list):
            logger.warning(f"Item at index {item_idx} has history of type {type(item_history)}, expected list. Skipping.")
            continue

        for turn_idx, turn in enumerate(item_history):
            if not isinstance(turn, dict) or 'user' not in turn or 'bot' not in turn:
                logger.warning(f"Turn {turn_idx} in item {item_idx} is malformed: {turn}. Skipping turn.")
                continue
            
            user_query = str(turn['user'])
            bot_response = str(turn['bot'])

            # Format for Mistral: <s>[INST] User Query [/INST] Bot Response</s>
            # Note: A space is often added before the bot_response if not handled by tokenizer.
            prompt_str = f"{bos}{inst_open} {user_query} {inst_close}"
            answer_str = f" {bot_response}{eos}" # Leading space for the answer part

            # Tokenize prompt and answer parts separately to correctly create labels
            # add_special_tokens=False because we are manually adding BOS/EOS per turn segment
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
            
            current_turn_input_ids = prompt_tokens + answer_tokens
            # For labels, mask the prompt part by setting tokens to -100
            current_turn_labels = [-100] * len(prompt_tokens) + answer_tokens
            
            full_concatenated_input_ids.extend(current_turn_input_ids)
            full_concatenated_labels.extend(current_turn_labels)

        # Truncate if the full concatenated history exceeds max_seq_length
        if len(full_concatenated_input_ids) > max_seq_length:
            full_concatenated_input_ids = full_concatenated_input_ids[:max_seq_length]
            full_concatenated_labels = full_concatenated_labels[:max_seq_length]
        elif len(full_concatenated_input_ids) == 0: # Handle empty history cases
            logger.warning(f"Item at index {item_idx} resulted in empty tokenized output. Skipping.")
            continue
            
        # Create attention mask (1 for real tokens, 0 for padding - padding handled by collator)
        attention_mask = [1] * len(full_concatenated_input_ids)

        processed_examples['input_ids'].append(full_concatenated_input_ids)
        processed_examples['attention_mask'].append(attention_mask)
        processed_examples['labels'].append(full_concatenated_labels)
        
    return processed_examples


class ConversationDataset(Dataset):
    """PyTorch Dataset for conversational data."""
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn_conversations(batch, tokenizer):
    """Collate function to pad batch elements to the same length."""
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Determine max length in this batch for padding
    max_len = max(len(ids) for ids in input_ids_list)
    if max_len == 0: # Should not happen if empty examples are filtered
        return None 

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback if pad_token_id is not set, common to use eos_token_id
        pad_token_id = tokenizer.eos_token_id 
        logger.warning(f"tokenizer.pad_token_id is None. Using eos_token_id ({pad_token_id}) for padding.")
        if pad_token_id is None: # Critical error if no pad token can be determined
             raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as fallback for padding.")


    for i in range(len(batch)):
        input_ids = input_ids_list[i]
        attention_mask = attention_mask_list[i]
        labels = labels_list[i]
        
        padding_length = max_len - len(input_ids)
        
        # Pad right
        padded_input_ids.append(torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([attention_mask, torch.full((padding_length,), 0, dtype=torch.long)])) # Pad attention mask with 0
        padded_labels.append(torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])) # Pad labels with -100 (ignore index)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }

import time

def train_model_mistral(original_model, tokenizer, train_dataset_hf, args):
    """
    Trains a Mistral model using the provided dataset and arguments.

    Args:
        original_model: The pre-trained Mistral model (e.g., from AutoModelForCausalLM.from_pretrained).
        tokenizer: The tokenizer for the model (e.g., from AutoTokenizer.from_pretrained).
        train_dataset_hf (datasets.Dataset): The Hugging Face training dataset.
                                            Must contain a 'history' column, where each item is a list of turns,
                                            and each turn is a dict {'user': str, 'bot': str}.
        args: An object or Namespace containing training arguments:
              - num_epochs (int): Number of training epochs.
              - model_save_path (str): Path to save the fine-tuned model and tokenizer.
              - learning_rate (float): Optimizer learning rate (e.g., 2e-5, 5e-5).
              - batch_size (int): Training batch size (e.g., 1, 2, 4, adjust based on GPU memory).
              - max_seq_length (int): Maximum sequence length for tokenization and padding (e.g., 512, 1024, 2048).
              - gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before an optimizer update. Defaults to 1.
              - warmup_steps (int, optional): Number of warmup steps for the learning rate scheduler. Defaults to 0.
              - logging_steps (int, optional): Log training loss every X steps. Defaults to 10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    original_model.to(device)
    original_model.train() # Set model to training mode

    # Ensure tokenizer has a pad token. This is crucial for batching.
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: If you add a new token or change pad_token such that vocab size changes,
        # you might need to resize model token embeddings:
        # original_model.resize_token_embeddings(len(tokenizer))
        # However, just setting pad_token = eos_token usually means using an existing token.
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")


    # 1. Preprocess and tokenize the dataset
    logger.info("Preprocessing and tokenizing dataset...")
    tokenized_data_dict = format_and_tokenize_dataset(train_dataset_hf, tokenizer, args.max_seq_length)
    
    if not tokenized_data_dict['input_ids']:
        logger.error("Tokenization resulted in an empty dataset. Please check your data and formatting.")
        return None

    # Create a PyTorch Dataset
    pytorch_train_dataset = ConversationDataset(tokenized_data_dict)
    logger.info(f"Created PyTorch Dataset with {len(pytorch_train_dataset)} examples.")


    # 2. Create DataLoader
    logger.info(f"Creating DataLoader with batch size: {args.batch_size}...")
    train_dataloader = DataLoader(
        pytorch_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_conversations(batch, tokenizer)
    )

    # 3. Set up Optimizer and Scheduler
    logger.info(f"Setting up optimizer with learning rate: {args.learning_rate}...")

    # Freeze all parameters
    for name, param in original_model.named_parameters():
        param.requires_grad = False

    # Unfreeze only adapter layers based on name match
    adapter_layer_prefixes = [layer['name'] for layer in args.adapter_layers_json]

    for name, param in original_model.named_parameters():
        for prefix in adapter_layer_prefixes:
            if name.startswith(prefix):
                param.requires_grad = True
                
                break

    print("trainable layers: ")
    # Print layers which require gradients (i.e., will be updated during training)
    for name, param in original_model.named_parameters():
        if param.requires_grad:
            print(name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in original_model.parameters())
    trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


    # Only pass trainable parameters to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, original_model.parameters()), lr=args.learning_rate, eps=1e-8) # Added eps for stability

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    num_training_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_training_steps_per_epoch +=1 # account for the last partial step

    total_training_steps = num_training_steps_per_epoch * args.num_epochs
    
    num_warmup_steps = getattr(args, 'warmup_steps', 0)
    if isinstance(num_warmup_steps, float): # if warmup_steps is a ratio
        num_warmup_steps = int(total_training_steps * num_warmup_steps)

    logger.info(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging_steps = getattr(args, 'logging_steps', 10)

    # 4. Training Loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    original_model.zero_grad() # Clear gradients before starting

    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        epoch_total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(original_model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients for the next accumulation

            epoch_total_loss += loss.item() * gradient_accumulation_steps # De-normalize for logging

            

            start_time = time.time()

            if (step + 1) % (logging_steps * gradient_accumulation_steps) == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                step_time = elapsed * 1000 / logging_steps  # in ms
                tokens_per_second = int(input_ids.numel() * logging_steps / elapsed)

                grad_norm = 0.0
                for p in original_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                print(
                    f"step {step+1}/{len(train_dataloader)} | "
                    f"loss {loss.item() * gradient_accumulation_steps:.6f} (+nanz)| "
                    f"norm {grad_norm:.4f} (+nanz)| "
                    f"lr {current_lr:.2e} | "
                    f"{step_time:.2f} ms | "
                    f"{tokens_per_second} tok/s",
                    flush=True
                )

                start_time = time.time()


        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        logger.info(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")

    

    return original_model


def train_model_adapted_mistral(original_model, tokenizer, train_dataset_hf, args):
    """
    Trains a Mistral model using the provided dataset and arguments.

    Args:
        original_model: The pre-trained Mistral model (e.g., from AutoModelForCausalLM.from_pretrained).
        tokenizer: The tokenizer for the model (e.g., from AutoTokenizer.from_pretrained).
        train_dataset_hf (datasets.Dataset): The Hugging Face training dataset.
                                            Must contain a 'history' column, where each item is a list of turns,
                                            and each turn is a dict {'user': str, 'bot': str}.
        args: An object or Namespace containing training arguments:
              - num_epochs (int): Number of training epochs.
              - model_save_path (str): Path to save the fine-tuned model and tokenizer.
              - learning_rate (float): Optimizer learning rate (e.g., 2e-5, 5e-5).
              - batch_size (int): Training batch size (e.g., 1, 2, 4, adjust based on GPU memory).
              - max_seq_length (int): Maximum sequence length for tokenization and padding (e.g., 512, 1024, 2048).
              - gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before an optimizer update. Defaults to 1.
              - warmup_steps (int, optional): Number of warmup steps for the learning rate scheduler. Defaults to 0.
              - logging_steps (int, optional): Log training loss every X steps. Defaults to 10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    original_model.to(device)
    original_model.train() # Set model to training mode

    # Ensure tokenizer has a pad token. This is crucial for batching.
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: If you add a new token or change pad_token such that vocab size changes,
        # you might need to resize model token embeddings:
        # original_model.resize_token_embeddings(len(tokenizer))
        # However, just setting pad_token = eos_token usually means using an existing token.
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")


    # 1. Preprocess and tokenize the dataset
    logger.info("Preprocessing and tokenizing dataset...")
    tokenized_data_dict = format_and_tokenize_dataset(train_dataset_hf, tokenizer, args.max_seq_length)
    
    if not tokenized_data_dict['input_ids']:
        logger.error("Tokenization resulted in an empty dataset. Please check your data and formatting.")
        return None

    # Create a PyTorch Dataset
    pytorch_train_dataset = ConversationDataset(tokenized_data_dict)
    logger.info(f"Created PyTorch Dataset with {len(pytorch_train_dataset)} examples.")


    # 2. Create DataLoader
    logger.info(f"Creating DataLoader with batch size: {args.batch_size}...")
    train_dataloader = DataLoader(
        pytorch_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_conversations(batch, tokenizer)
    )

    # 3. Set up Optimizer and Scheduler
    logger.info(f"Setting up optimizer with learning rate: {args.learning_rate}...")

    print("trainable layers: ")
    # Print layers which require gradients (i.e., will be updated during training)
    for name, param in original_model.named_parameters():
        if param.requires_grad:
            print(name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in original_model.parameters())
    trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Only pass trainable parameters to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, original_model.parameters()), lr=args.learning_rate, eps=1e-8) # Added eps for stability

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    num_training_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_training_steps_per_epoch +=1 # account for the last partial step

    total_training_steps = num_training_steps_per_epoch * args.num_epochs
    
    num_warmup_steps = getattr(args, 'warmup_steps', 0)
    if isinstance(num_warmup_steps, float): # if warmup_steps is a ratio
        num_warmup_steps = int(total_training_steps * num_warmup_steps)

    logger.info(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging_steps = getattr(args, 'logging_steps', 10)

    # 4. Training Loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    original_model.zero_grad() # Clear gradients before starting

    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        epoch_total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(original_model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients for the next accumulation

            epoch_total_loss += loss.item() * gradient_accumulation_steps # De-normalize for logging
            start_time = time.time()

            if (step + 1) % (logging_steps * gradient_accumulation_steps) == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                step_time = elapsed * 1000 / logging_steps  # in ms
                tokens_per_second = int(input_ids.numel() * logging_steps / elapsed)

                grad_norm = 0.0
                for p in original_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                print(
                    f"step {step+1}/{len(train_dataloader)} | "
                    f"loss {loss.item() * gradient_accumulation_steps:.6f} (+nanz)| "
                    f"norm {grad_norm:.4f} (+nanz)| "
                    f"lr {current_lr:.2e} | "
                    f"{step_time:.2f} ms | "
                    f"{tokens_per_second} tok/s",
                    flush=True
                )

                start_time = time.time()


        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        logger.info(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")

    

    return original_model