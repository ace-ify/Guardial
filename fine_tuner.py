
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
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def save_model_metadata(output_dir, model_name, dataset_path, epochs, learning_rate, loss_history):
    """
    Saves metadata about the forged model for tracking and integration.
    """
    metadata = {
        "model_name": model_name,
        "base_model": model_name,
        "output_path": output_dir,
        "dataset_path": dataset_path,
        "training_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": 4,
        },
        "training_metrics": {
            "final_loss": loss_history[-1] if loss_history else None,
            "initial_loss": loss_history[0] if loss_history else None,
            "loss_history": loss_history,
        },
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "available_for_unlearning": True,
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_path}")
    return metadata

def run_fine_tuning(dataset_path, model_name="distilgpt2", output_dir="./forged_model", epochs=1, learning_rate=5e-5):
    """
    Fine-tunes a causal language model with enhanced tracking and metadata.

    Args:
        dataset_path (str): Path to the training dataset (CSV with a 'text' column).
        model_name (str): The name of the Hugging Face model to fine-tune.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        dict: Training results including loss history, metadata, and model path.
    """
    try:
        logger.info(f"Starting fine-tuning: model={model_name}, epochs={epochs}, lr={learning_rate}")
        
        # 1. Load Dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        if 'text' not in df.columns:
            raise ValueError("Dataset must contain a 'text' column.")
        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        # Filter out empty or null text entries
        df = df[df['text'].notna()]
        df = df[df['text'].astype(str).str.strip() != '']
        
        if df.empty:
            raise ValueError("The dataset contains no valid text entries. Please ensure the 'text' column has non-empty values.")

        logger.info(f"Dataset loaded: {len(df)} valid samples")

        # Create a temporary file to hold the text data
        with open("temp_train.txt", "w", encoding="utf-8") as f:
            for text in df['text']:
                text_str = str(text).strip()
                if text_str:  # Only write non-empty lines
                    f.write(text_str + "\n")

        # 2. Initialize Tokenizer and Model
        logger.info(f"Loading base model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # 3. Determine optimal block size based on data
        # Calculate average text length to set appropriate block size
        with open("temp_train.txt", "r", encoding="utf-8") as f:
            sample_texts = f.readlines()[:10]  # Sample first 10 lines
        
        if sample_texts:
            avg_tokens = sum(len(tokenizer.encode(text)) for text in sample_texts) / len(sample_texts)
            # Use adaptive block size: minimum 16, maximum 128
            block_size = max(16, min(128, int(avg_tokens * 2)))
            logger.info(f"Using adaptive block_size={block_size} based on data (avg tokens: {avg_tokens:.1f})")
        else:
            block_size = 64  # Default fallback
            logger.info(f"Using default block_size={block_size}")

        # 4. Create Dataset and Data Collator
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path="temp_train.txt",
            block_size=block_size,
        )
        
        # Validate dataset has samples
        if len(train_dataset) == 0:
            raise ValueError(
                f"No training samples were created even with block_size={block_size}. "
                f"The text file may be empty or corrupted. Please check your dataset."
            )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        
        logger.info(f"Training dataset prepared: {len(train_dataset)} blocks")

        # 4. Define Training Arguments (optimized for efficiency)
        # Calculate appropriate logging steps based on dataset size
        # For small datasets, log more frequently
        total_steps = (len(train_dataset) // 4) * epochs  # Rough estimate
        logging_steps = max(1, min(10, total_steps // 5))  # Log at least 5 times, minimum every step
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,  # Effective batch size = 8
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            logging_first_step=True,  # Always log the first step
            fp16=torch.cuda.is_available(),  # Mixed precision training
            dataloader_num_workers=2,
            report_to="none",  # Disable external reporting
        )
        
        logger.info(f"Training configuration: logging_steps={logging_steps}, total_steps≈{total_steps}")

        # 5. Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # 6. Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info(f"Training complete. Final loss: {train_result.training_loss:.4f}")

        # 7. Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 8. Clean up temporary file
        if os.path.exists("temp_train.txt"):
            os.remove("temp_train.txt")

        # 9. Extract training loss history
        loss_history = []
        steps = []
        for log in trainer.state.log_history:
            if 'loss' in log:
                loss_history.append(float(log['loss']))
                steps.append(int(log.get('step', len(loss_history))))
        
        # If no loss history was captured (very small dataset), use final loss
        if not loss_history and hasattr(train_result, 'training_loss'):
            loss_history = [float(train_result.training_loss)]
            steps = [1]
            logger.info("Very small dataset - using final loss only")
        
        logger.info(f"Loss history: {len(loss_history)} data points, steps: {steps}")

        # 10. Save model metadata for integration
        metadata = save_model_metadata(output_dir, model_name, dataset_path, epochs, learning_rate, loss_history)

        # 11. Return comprehensive results
        return {
            "status": "success",
            "model_path": output_dir,
            "loss_history": loss_history,
            "steps": steps,
            "final_loss": loss_history[-1] if loss_history else None,
            "metadata": metadata,
            "dataset_samples": len(df),
            "training_blocks": len(train_dataset),
        }

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        # Clean up in case of error
        if os.path.exists("temp_train.txt"):
            os.remove("temp_train.txt")
        raise e

if __name__ == '__main__':
    # Example usage (for testing this script directly)
    # Create a dummy dataset
    dummy_data = {
        'text': [
            "This is the first sentence.",
            "Here is another sentence for the model to learn from.",
            "Fine-tuning is the process of adapting a pre-trained model.",
            "Language models can be fine-tuned for specific tasks.",
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv("dummy_dataset.csv", index=False)
    
    print("Running fine-tuning with dummy data...")
    loss = run_fine_tuning("dummy_dataset.csv", epochs=1)
    print("Fine-tuning complete.")
    print("Loss history:", loss)
    
    # Clean up dummy file
    os.remove("dummy_dataset.csv")
