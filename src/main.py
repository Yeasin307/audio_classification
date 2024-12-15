import os
import librosa
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
import torch
# import matplotlib.pyplot as plt

# Verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define dataset directory
dataset_dir = '../dataset'

# Define label mapping
intent_class_map = {'abnormal': 0, 'normal': 1}

# def plot_audio_waveform(audio_array, sampling_rate, title="Audio Waveform"):
#     plt.figure(figsize=(10, 4))
#     time_axis = np.linspace(0, len(audio_array) / sampling_rate, num=len(audio_array))
#     plt.plot(time_axis, audio_array, color='blue')
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.title(title)
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# Function to load the data
def load_data(split):
    dataset = []
    for class_folder, intent_class in intent_class_map.items():
        class_path = os.path.join(dataset_dir, split, class_folder)

        if not os.path.exists(class_path):
            continue

        for file_name in os.listdir(class_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(class_path, file_name)

                audio_array, sampling_rate = librosa.load(file_path, sr=16000)
                audio_array = audio_array.astype(np.float32)
                # plot_audio_waveform(audio_array, sampling_rate, title=intent_class)

                dataset_entry = {
                    "path": file_path,
                    "audio": {
                        "path": file_path,
                        "array": audio_array,
                        "sampling_rate": sampling_rate,
                    },
                    "intent_class": intent_class,
                }

                dataset.append(dataset_entry)
    return dataset

# Load train and test data
train_data = load_data("train")
test_data = load_data("test")

# Prepare dataset
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test": Dataset.from_list(test_data)
})

print(dataset['train'][0])

dataset = dataset.remove_columns('path')

# Label mapping for normal and abnormal classes
intent_class_label = ['abnormal', 'normal']
label2id = {label: i for i, label in enumerate(intent_class_label)}
id2label = {str(i): label for i, label in enumerate(intent_class_label)}

print(f"ID to Label Mapping: {id2label}")

# Load XLSR-Wav2Vec2 processor (pretrained model) using AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Cast the dataset column to Audio format
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Preprocess the audio data
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=16_000, return_tensors="pt", padding=True, truncation=True,max_length=100000,
    )
    return inputs

# Encode dataset
encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
encoded_dataset = encoded_dataset.rename_column("intent_class", "label")

# Initialize accuracy metric
accuracy = evaluate.load("accuracy")

# Function to compute metrics
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# Initialize the model with XLSR-Wav2Vec2 for sequence classification
num_labels = len(intent_class_label)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", num_labels=num_labels, label2id=label2id, id2label=id2label
).to(device)

training_args = TrainingArguments(
    output_dir="../wav2vec2_bengali_model",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save at the end of each epoch
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_dir="../logs",
    logging_steps=50,
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="accuracy",  # Use accuracy for the best model selection
    save_total_limit=1,  # Keep only the best model, delete previous checkpoints
    push_to_hub=False,
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    # Pass the feature_extractor if required
    # feature_extractor=feature_extractor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()