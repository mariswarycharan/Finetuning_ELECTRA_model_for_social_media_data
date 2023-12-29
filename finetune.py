from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

# 1. Prepare Your Dataset
# Replace these with your own data and labels.
texts = ["hello it is  government","prime minister is killed"]
labels = [0, 1]
num_classes = len(labels)

# 2. Tokenize Your Data
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

# 3. Create DataLoaders
dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, torch.tensor(labels))
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Define the Fine-Tuning Model
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=num_classes)

# 5. Fine-Tuning Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Save the fine-tuned model
model.save_pretrained("your_fine_tuned_model_directory")
tokenizer.save_pretrained("your_fine_tuned_model_directory")










# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

# your_text_data = ["hello it is  government","prime minister is killed"]
# your_labels = [ 0,1]
# # Tokenize your text data
# inputs = tokenizer(your_text_data, padding=True, truncation=True, return_tensors="tf")
# print(inputs)

# import tensorflow as tf

# dataset = tf.data.Dataset.from_tensor_slices({
#     'input_ids': inputs['input_ids'],
#     'attention_mask': inputs['attention_mask'],
#     'token_type_ids': inputs['token_type_ids'],
#     'labels': your_labels  # Your label data
# })
# print(dataset)

# # Define the size of the splits as a percentage
# train_size = 0.50  # 50% of the data for training
# valid_size = 0.25  # 25% of the data for validation
# test_size = 0.25   # 25% of the data for testing

# # Calculate the number of samples for each split
# total_samples = len(dataset)
# train_samples = int(train_size * total_samples)
# valid_samples = int(valid_size * total_samples)

# # Split the dataset using take() and skip()
# train_dataset = dataset.take(train_samples)
# valid_dataset = dataset.skip(train_samples).take(valid_samples)
# test_dataset = dataset.skip(train_samples + valid_samples)


# from transformers import TFElectraForSequenceClassification

# num_classes = 2  # Replace with the actual number of classes in your dataset
# model = TFElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=num_classes)


# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# metrics = ['accuracy']

# model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# batch_size = 32
# history = model.fit(
#     train_dataset.shuffle(buffer_size=1024).batch(batch_size),
#     validation_data=valid_dataset.batch(batch_size),
#     epochs=3
# )


# test_loss, test_accuracy = model.evaluate(test_dataset.batch(batch_size))
# model.save("fine_tuned_electra_model")
