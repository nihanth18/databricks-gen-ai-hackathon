
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_scheduler
import pandas as pd

# COMMAND ----------


df = spark.table('gcp.default.cars_details')
data = df.select(
    'location', 'manufactured year', 'car segment', 'transmission', 
    'fuel type', 'km driven', 'full name', 'brand', 'model', 
    'city', 'price', 'price in numbers', 'transmissionType', 
    'car type', 'top features', 'comfort features', 
    'interior features', 'exterior features', 'safety features', 
    'Color', 'price range segment', 'state', 'owner type'
).limit(10000).toPandas() 


data = data.fillna("Unknown")


data['text'] = (
    "Brand: " + data['brand'] + ", Model: " + data['model'] +
    ", Location: " + data['location'] + ", Year: " + data['manufactured year'].astype(str) +
    ", Transmission: " + data['transmission'] + ", Fuel: " + data['fuel type'] +
    ", Price: " + data['price in numbers'].astype(str)  
)


texts = data['text'].tolist()


print(f"First 5 texts: {texts[:5]}")
print(f"Type of texts: {type(texts)}, Type of first element: {type(texts[0])}")


# COMMAND ----------


class CarDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]  
        encoding = self.tokenizer(
            text,  
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  
            'attention_mask': encoding['attention_mask'].squeeze(0)  
        }



# COMMAND ----------


model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token  

dataset = CarDataset(texts, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



# COMMAND ----------


model.train()
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}...")
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1} completed, Loss: {loss.item()}")


# model.save_pretrained("/dbfs/tmp/car_quotation_gpt2")
# tokenizer.save_pretrained("/dbfs/tmp/car_quotation_gpt2")

# print("Model and tokenizer saved.")


# COMMAND ----------


model.save_pretrained("/dbfs/tmp/car_quotation_gpt2")
tokenizer.save_pretrained("/dbfs/tmp/car_quotation_gpt2")

print("Model and tokenizer saved.")

# COMMAND ----------

from transformers import GPT2Tokenizer, GPT2LMHeadModel


model_path = "/dbfs/tmp/car_quotation_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


model.eval()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# COMMAND ----------


user_input = "Brand: Hyundai, Model: Hyundai i20, Location: Ghaziabad, Year: 2013, Transmission: Manual, Fuel: Diesel"


input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:", generated_text)

