# Databricks notebook source
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import json


# COMMAND ----------

model_path = "/dbfs/tmp/car_quotation_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


# COMMAND ----------

# Move the model to the correct device (GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# COMMAND ----------

dbutils.widgets.text("trade_in_details", "")
dbutils.widgets.text("question", "")  # New parameter
trade_in_details = dbutils.widgets.get("trade_in_details")
question = dbutils.widgets.get("question") 

# COMMAND ----------



# COMMAND ----------

def estimate_trade_in_value(trade_in_details):
    """
    Generate the estimated trade-in value based on the user's input.
    trade_in_details: str, input string with car details like brand, model, year, etc.
    """
    model.eval()  # Set model to evaluation mode

    # Tokenize the user input and encode it into input_ids
    input_ids = tokenizer.encode(trade_in_details, return_tensors='pt').to(device)

    # Generate response from the model
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated output to readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Trade-In Output:", generated_text)

    # Extract the trade-in value from the generated text (you can implement additional parsing logic here)
    try:
        # Assume generated text contains a number representing the trade-in value
        trade_in_value = float(generated_text.split()[-1].replace(",", ""))  # Example parsing
    except ValueError:
        trade_in_value = 0  # Default to 0 if parsing fails

    return trade_in_value

# COMMAND ----------

trade_in_value = estimate_trade_in_value(trade_in_details)

# COMMAND ----------

dbutils.notebook.exit(json.dumps(trade_in_value))
