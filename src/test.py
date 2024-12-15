import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

# Load the best model
best_model = Wav2Vec2ForSequenceClassification.from_pretrained("../wav2vec2_bengali_model/???")

# Load the feature extractor (use the same feature extractor as during training)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Inference on a new audio file
audio_file = "../dataset/test/abnormal/2.wav"
audio_array, sr = librosa.load(audio_file, sr=16000)  # Load the audio file

# Preprocess the audio file
inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = best_model(**inputs).logits

# Get the predicted class
predicted_class_id = torch.argmax(logits).item()

# Convert predicted class ID to label
predicted_label = best_model.config.id2label[predicted_class_id]

print(f"Predicted label: {predicted_label}")
