from transformers import BertConfig, BertModel, TFBertModel, AutoTokenizer
import torch
import tensorflow as tf

# Building the configuration
config = BertConfig()

# Building the model from the config
pytorch_model = BertModel(config)
tensorflow_model = TFBertModel(config)

# Loading the model from a checkpoint
checkpoint = "bert-base-uncased"
pytorch_model = BertModel.from_pretrained(checkpoint)
tensorflow_model = TFBertModel.from_pretrained(checkpoint)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Saving the models locally
pytorch_model.save_pretrained("./course/models/pytorch_03")
tensorflow_model.save_pretrained("./course/models/tensorflow_03")

# Define raw inputs
raw_inputs = [
    "Hello!", 
    "Cool.", 
    "Nice!"
]

# Tokenize the inputs
pytorch_encoded_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
tensorflow_encoded_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")

# Generate outputs
pytorch_outputs = pytorch_model(**pytorch_encoded_inputs)
tensorflow_outputs = tensorflow_model(**tensorflow_encoded_inputs)

# Print output
print(f"Config: {config}")
print(f"Pytorch model: {pytorch_model}")
print(f"Tensorflow model: {tensorflow_model}")
print(f"Pytorch outputs: {pytorch_outputs}")
print(f"Tensorflow outputs: {tensorflow_outputs}")