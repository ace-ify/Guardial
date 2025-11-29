
import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
import math
import logging

logger = logging.getLogger(__name__)

def calculate_perplexity(model, tokenizer, dataset_path):
    """Calculates perplexity of a model on a given dataset."""
    # Handle both CSV and text files
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        if 'text' in df.columns:
            text = ' '.join(df['text'].dropna().astype(str).tolist())
        else:
            text = df.iloc[:, 0].dropna().astype(str).str.cat(sep=' ')
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text for perplexity calculation")
        return float('inf')
    
    encodings = tokenizer(text, return_tensors="pt")
    max_length = min(model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 1024, 512)
    stride = 256
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return float('inf')
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def check_info_presence(model, tokenizer, info_to_check, num_samples=5):
    """
    Checks if the model generates a specific piece of information.
    Returns a score from 0 (doesn't know) to 1 (knows).
    Uses multiple prompts for robustness.
    """
    prompts = [
        f"What do you know about {info_to_check}?",
        f"Tell me about {info_to_check}.",
        f"Who is {info_to_check}?",
        f"Describe {info_to_check}.",
        f"{info_to_check} is",
    ]
    
    presence_count = 0
    for prompt in prompts[:num_samples]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if the info is in the response
        if info_to_check.lower() in generated_text.lower():
            presence_count += 1
    
    return presence_count / num_samples

def calculate_accuracy_metrics(pre_perplexity, post_perplexity, pre_score, post_score):
    """
    Converts perplexity and presence scores into percentage-based accuracy metrics.
    
    Retain Accuracy: How well the model maintains its knowledge (lower perplexity change is better)
    Forget Accuracy: How well the model forgets specific info (lower presence score is better)
    """
    # Retain accuracy: if perplexity stays similar or improves slightly, accuracy is high
    # We expect a small increase in perplexity (degradation), but not too much
    perplexity_change_ratio = post_perplexity / pre_perplexity if pre_perplexity > 0 else 1.0
    
    # If perplexity doubles or more, retain accuracy is low
    # If perplexity stays same, retain accuracy is 100%
    if perplexity_change_ratio <= 1.0:
        retain_accuracy = 100.0
    elif perplexity_change_ratio <= 1.2:
        retain_accuracy = 95.0
    elif perplexity_change_ratio <= 1.5:
        retain_accuracy = 85.0
    elif perplexity_change_ratio <= 2.0:
        retain_accuracy = 70.0
    else:
        retain_accuracy = max(50.0, 100.0 - (perplexity_change_ratio - 1.0) * 30)
    
    # Forget accuracy: how much the presence score decreased
    forget_improvement = pre_score - post_score
    if forget_improvement >= 0.8:
        forget_accuracy = 95.0
    elif forget_improvement >= 0.6:
        forget_accuracy = 85.0
    elif forget_improvement >= 0.4:
        forget_accuracy = 70.0
    elif forget_improvement >= 0.2:
        forget_accuracy = 55.0
    else:
        forget_accuracy = max(30.0, 50.0 + forget_improvement * 100)
    
    return retain_accuracy, forget_accuracy


def run_unlearning(model_path, forget_set_path, info_to_forget):
    """
    Performs unlearning using PEFT (LoRA) with optimized settings for 30-40 minute execution.

    Args:
        model_path (str): Path to the base model.
        forget_set_path (str): Path to the dataset to retain (without forgotten info).
        info_to_forget (str): Specific information to be forgotten (for evaluation).

    Returns:
        dict: A dictionary containing the unlearning metrics with percentage accuracies.
    """
    try:
        logger.info(f"Starting unlearning process for: {info_to_forget}")
        
        # 1. Load Model and Tokenizer
        logger.info(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Model loaded successfully")

        # 2. Evaluate model BEFORE unlearning
        logger.info("Evaluating model before unlearning...")
        pre_retain_perplexity = calculate_perplexity(model, tokenizer, forget_set_path)
        pre_forget_score = check_info_presence(model, tokenizer, info_to_forget)
        logger.info(f"Pre-unlearning metrics - Perplexity: {pre_retain_perplexity:.2f}, Forget score: {pre_forget_score:.2f}")

        # 3. Configure PEFT (LoRA) with optimized parameters
        logger.info("Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=16,  # Rank of the adapter
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"] if hasattr(model, 'q_proj') else None,  # Target attention layers
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA adapter configured")

        # 4. Prepare dataset
        logger.info("Preparing forget dataset...")
        
        # Convert CSV to text file
        df = pd.read_csv(forget_set_path)
        if 'text' not in df.columns:
            raise ValueError("Forget set must contain a 'text' column.")
        
        # Filter out empty entries
        df = df[df['text'].notna()]
        df = df[df['text'].astype(str).str.strip() != '']
        
        if df.empty:
            raise ValueError("The forget set contains no valid text entries.")
        
        logger.info(f"Loaded {len(df)} text entries from forget set")
        
        # For very small datasets, concatenate all text with newlines
        # This ensures we have enough content to create training samples
        all_text = '\n'.join(df['text'].astype(str).tolist())
        
        # Create temporary text file
        # Safeguard: Truncate very large datasets for demo performance
        if len(all_text) > 20000:
            logger.warning(f"Dataset too large ({len(all_text)} chars). Truncating to 20000 chars for demo performance.")
            all_text = all_text[:20000]
        temp_forget_file = "temp_forget.txt"
        with open(temp_forget_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        
        # Determine adaptive block size based on total text length
        total_tokens = len(tokenizer.encode(all_text))
        logger.info(f"Total tokens in dataset: {total_tokens}")
        
        # Set block size to ensure at least a few samples
        if total_tokens < 32:
            # Very small dataset - use minimal block size
            block_size = max(8, total_tokens // 4)
        else:
            # Calculate based on average entry length
            sample_texts = df['text'].astype(str).head(min(10, len(df))).tolist()
            avg_tokens = sum(len(tokenizer.encode(text)) for text in sample_texts) / len(sample_texts)
            block_size = max(8, min(64, int(avg_tokens * 1.5)))
        
        logger.info(f"Using adaptive block_size={block_size}")
        
        try:
            forget_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=temp_forget_file,
                block_size=block_size,
            )
            
            logger.info(f"Created dataset with {len(forget_dataset)} blocks")
        except Exception as e:
            # If TextDataset fails, create a simple custom dataset
            logger.warning(f"TextDataset failed, using fallback approach: {e}")
            
            # Fallback: tokenize the entire text as one sample
            encodings = tokenizer(
                all_text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Create a simple dataset wrapper
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings
                
                def __len__(self):
                    return 1
                
                def __getitem__(self, idx):
                    return {key: val[idx] for key, val in self.encodings.items()}
            
            forget_dataset = SimpleDataset(encodings)
            logger.info(f"Created fallback dataset with {len(forget_dataset)} sample")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        
        if len(forget_dataset) == 0:
            if os.path.exists(temp_forget_file):
                os.remove(temp_forget_file)
            raise ValueError(
                f"Could not create any training samples. The forget set text is too short. "
                f"Please provide more text data (at least a few sentences)."
            )

        # 5. Train the adapter on the "forget set" (optimized for 30-40 min)
        logger.info("Starting adapter training (unlearning phase)...")
        
        # Calculate adaptive logging steps
        total_steps = (len(forget_dataset) // 8) * 2  # Rough estimate
        logging_steps = max(1, min(10, total_steps // 5))
        
        training_args = TrainingArguments(
            output_dir="./unlearning_output",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Reduced for faster hackathon demos
            per_device_train_batch_size=8,  # Increased batch size for speed
            gradient_accumulation_steps=2,  # Effective batch size = 16
            learning_rate=5e-4,  # Higher learning rate for faster adaptation
            logging_steps=logging_steps,
            logging_first_step=True,
            save_steps=500,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),  # Mixed precision for speed
            dataloader_num_workers=2,  # Parallel data loading
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        logger.info(f"Training configuration: logging_steps={logging_steps}, total_steps≈{total_steps}")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=forget_dataset,
        )
        
        logger.info("Training adapter on forget set...")
        trainer.train()
        logger.info("Adapter training complete")
        
        # Clean up temporary file
        if os.path.exists(temp_forget_file):
            os.remove(temp_forget_file)

        # 6. Save the unlearned model
        output_model_path = "./unlearned_model"
        logger.info(f"Saving unlearned model to {output_model_path}")
        model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

        # 7. Evaluate model AFTER unlearning
        logger.info("Evaluating model after unlearning...")
        post_retain_perplexity = calculate_perplexity(model, tokenizer, forget_set_path)
        post_forget_score = check_info_presence(model, tokenizer, info_to_forget)
        logger.info(f"Post-unlearning metrics - Perplexity: {post_retain_perplexity:.2f}, Forget score: {post_forget_score:.2f}")

        # 8. Calculate accuracy percentages
        retain_accuracy, forget_accuracy = calculate_accuracy_metrics(
            pre_retain_perplexity, post_retain_perplexity,
            pre_forget_score, post_forget_score
        )

        logger.info(f"Unlearning complete - Retain: {retain_accuracy:.1f}%, Forget: {forget_accuracy:.1f}%")

        return {
            "retain_accuracy": f"{retain_accuracy:.1f}%",
            "forget_accuracy": f"{forget_accuracy:.1f}%",
            "pre_perplexity": f"{pre_retain_perplexity:.2f}",
            "post_perplexity": f"{post_retain_perplexity:.2f}",
            "pre_forget_score": f"{pre_forget_score:.2f}",
            "post_forget_score": f"{post_forget_score:.2f}",
            "model_path": output_model_path
        }

    except Exception as e:
        logger.error(f"Error during unlearning: {str(e)}")
        raise e
