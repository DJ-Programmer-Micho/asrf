"""
Train Wav2Vec2 model on Kurdish Sorani ASR dataset using Hugging Face Trainer.
"""

import json
import librosa
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Trainer
from dataset_utilities.prepare_dataset import prepare_datasets
from model_utilities.tokenizer_setup import load_processor
from model_utilities.load_model import load_model_and_processor
from model_utilities.trainer_setup import load_training_arguments, load_data_collator


def compute_metrics(pred):
    wer_metric = evaluate.load("wer")
    pred_ids = pred.predictions.argmax(-1)
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def tokenize(batch, config):
    """
    Tokenize function that processes audio and text for training.
    
    Args:
        batch: Dictionary containing 'path' and 'transcript'
        config: Configuration dictionary
    
    Returns:
        Dictionary with processed inputs and labels, or None if invalid
    """
    audio_path = batch["path"]
    transcript = batch["transcript"]
    
    # Debug: Print the types and values
    print(f"Audio path type: {type(audio_path)}, value: {audio_path}")
    print(f"Transcript type: {type(transcript)}, value: {repr(transcript)}")
    
    # Handle different input types for transcript
    if isinstance(transcript, list):
        if len(transcript) == 0:
            print("Empty transcript list, skipping...")
            return None
        transcript = transcript[0] if len(transcript) == 1 else ' '.join(str(x) for x in transcript)
    
    # Convert to string if it's not already
    if not isinstance(transcript, str):
        if pd.isna(transcript):
            print(f"NaN transcript, skipping...")
            return None
        transcript = str(transcript)
    
    # Skip if transcript is empty or invalid
    if (transcript.strip() == "" or 
        transcript.strip().lower() == "nan" or
        transcript.strip().lower() == "none"):
        print(f"Invalid transcript: '{transcript}', skipping...")
        return None
    
    # Clean transcript
    transcript = transcript.strip()
    
    try:
        # Handle audio path - could be string or list
        if isinstance(audio_path, list):
            audio_path = audio_path[0] if len(audio_path) == 1 else audio_path
        
        if isinstance(audio_path, list):
            print(f"Multiple audio paths found: {audio_path}, using first one")
            audio_path = audio_path[0]
        
        # Convert audio_path to string if needed
        if not isinstance(audio_path, str):
            audio_path = str(audio_path)
        
        # Load audio
        audio_array, _ = librosa.load(audio_path, sr=config["dataset"]["sampling_rate"])
        
        # Process input audio - don't use return_tensors here for dataset mapping
        inputs = processor(
            audio_array, 
            sampling_rate=config["dataset"]["sampling_rate"],
            padding=False  # Let data collator handle padding
        )
        
        # Process text labels - ensure transcript is a string
        print(f"Processing transcript: '{transcript}' (type: {type(transcript)})")
        
        # Double-check transcript is string before processing
        if not isinstance(transcript, str):
            raise ValueError(f"Transcript is not string: {type(transcript)} - {repr(transcript)}")
        
        with processor.as_target_processor():
            # Process the transcript text
            try:
                labels = processor(transcript)
                inputs["labels"] = labels.input_ids
            except Exception as label_error:
                print(f"Error processing transcript '{transcript}': {str(label_error)}")
                print(f"Transcript type: {type(transcript)}")
                print(f"Transcript repr: {repr(transcript)}")
                raise label_error
        
        print(f"Successfully processed: {audio_path} -> {transcript[:50]}...")
        return inputs
        
    except Exception as e:
        print(f"Error processing {audio_path} with transcript '{transcript}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_processor_with_sample_data(processor, datasets):
    """Test the processor with a few sample entries to debug the issue."""
    print("\n=== DEBUGGING PROCESSOR ===")
    
    # Test with first few samples
    for i in range(min(3, len(datasets["train"]))):
        sample = datasets["train"][i]
        print(f"\nSample {i}:")
        print(f"  Path: {sample['path']} (type: {type(sample['path'])})")
        print(f"  Transcript: {repr(sample['transcript'])} (type: {type(sample['transcript'])})")
        
        # Test if transcript is processable
        transcript = sample['transcript']
        
        # Convert to string if needed
        if not isinstance(transcript, str):
            if pd.isna(transcript):
                print(f"  Skipping NaN transcript")
                continue
            transcript = str(transcript)
        
        print(f"  Cleaned transcript: '{transcript}'")
        
        # Test processor with this transcript
        try:
            with processor.as_target_processor():
                result = processor(transcript)
                print(f"  Processor result: {result}")
                print(f"  Success!")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print(f"  Transcript type: {type(transcript)}")
            print(f"  Transcript repr: {repr(transcript)}")
            import traceback
            traceback.print_exc()
            
    print("=== END DEBUGGING ===\n")


if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    print("Loading processor...")
    # Load processor
    processor = load_processor(
        config["dataset"]["vocab_path"],
        config["dataset"]["sampling_rate"]
    )

    print("Loading model...")
    # Load model
    model, _ = load_model_and_processor(config)

    print("Preparing datasets...")
    # Prepare datasets
    datasets = prepare_datasets(config)
    
    print(f"Dataset sizes - Train: {len(datasets['train'])}, Validation: {len(datasets['validation'])}, Test: {len(datasets['test'])}")
    
    # Print sample data to understand structure
    print("\nSample from train dataset:")
    print(datasets["train"][0])
    
    # Test processor with sample data first
    test_processor_with_sample_data(processor, datasets)
    
    print("\nTokenizing datasets...")
    # Tokenize datasets with error handling
    def tokenize_wrapper(batch):
        return tokenize(batch, config)
    
    # Process datasets one by one for better error handling
    tokenized_train = datasets["train"].map(
        tokenize_wrapper,
        remove_columns=datasets["train"].column_names,
        num_proc=1,
        desc="Tokenizing train dataset"
    )
    
    tokenized_validation = datasets["validation"].map(
        tokenize_wrapper, 
        remove_columns=datasets["validation"].column_names,
        num_proc=1,
        desc="Tokenizing validation dataset"
    )
    
    tokenized_test = datasets["test"].map(
        tokenize_wrapper,
        remove_columns=datasets["test"].column_names,
        num_proc=1,
        desc="Tokenizing test dataset"
    )
    
    # Filter out None results (caused by skipped transcripts)
    print("Filtering invalid samples...")
    original_train_size = len(tokenized_train)
    original_val_size = len(tokenized_validation)
    original_test_size = len(tokenized_test)
    
    tokenized_train = tokenized_train.filter(lambda x: x is not None and "labels" in x)
    tokenized_validation = tokenized_validation.filter(lambda x: x is not None and "labels" in x)
    tokenized_test = tokenized_test.filter(lambda x: x is not None and "labels" in x)
    
    print(f"Filtered train: {original_train_size} -> {len(tokenized_train)}")
    print(f"Filtered validation: {original_val_size} -> {len(tokenized_validation)}")
    print(f"Filtered test: {original_test_size} -> {len(tokenized_test)}")
    
    # Combine back into dataset dict
    tokenized_datasets = {
        "train": tokenized_train,
        "validation": tokenized_validation,
        "test": tokenized_test
    }
    
    # Verify we have data
    if len(tokenized_datasets["train"]) == 0:
        raise ValueError("No valid training samples found after tokenization!")
    
    if len(tokenized_datasets["validation"]) == 0:
        raise ValueError("No valid validation samples found after tokenization!")
    
    print("Loading training arguments and data collator...")
    # Load training args and collator
    training_args = load_training_arguments(config)
    data_collator = load_data_collator(processor, config)

    print("Initializing trainer...")
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    # Train model
    trainer.train()

    print("Saving model...")
    # Save final model + processor
    trainer.save_model(config["model"]["output_dir"])
    processor.save_pretrained(config["model"]["output_dir"])

    print("Plotting training metrics...")
    # Plot WER over training steps
    log_history = trainer.state.log_history
    steps, wers = [], []

    for log in log_history:
        if "eval_wer" in log:
            steps.append(log["step"])
            wers.append(log["eval_wer"])

    if steps and wers:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, wers, marker='o')
        plt.title("WER over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Word Error Rate (WER)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_wer_plot.png")
        plt.show()
        print("Training completed successfully!")
    else:
        print("No WER metrics found in training logs.")