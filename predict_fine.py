from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

# Load the fine-tuned model
model = ElectraForSequenceClassification.from_pretrained("your_fine_tuned_model_directory")

# Tokenize the new text
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
new_text = "prime minister is killed"
inputs = tokenizer(new_text, return_tensors="pt")

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# If it's a binary classification, you can use a sigmoid function to get the probability
# For multi-class classification, you can apply a softmax function
predicted_probabilities = torch.sigmoid(logits)  # For binary classification
# predicted_probabilities = torch.softmax(logits, dim=1)  # For multi-class classification

# Get the class with the highest probability
predicted_class = predicted_probabilities.argmax(dim=1).item()

# If you want to get the probability scores for all classes
class_probabilities = predicted_probabilities[0].tolist()

print("Predicted Class:", predicted_class)
print("Class Probabilities:", class_probabilities)
