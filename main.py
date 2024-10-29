# Databricks notebook source
import os
from google.oauth2 import service_account
import vertexai 
from vertexai.generative_models import GenerativeModel
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime
import uuid

# COMMAND ----------

service_account_info = {
}


# COMMAND ----------

credentials = service_account.Credentials.from_service_account_info(service_account_info)


vertexai.init(  
    project="",
    location="",  
    credentials=credentials
)


# COMMAND ----------

def estimate_car_price_from_natural_language(customer_name, question):
    try:
       
        schema = spark.table("cars_details_main_cleaned").schema
        schema_columns = [field.name for field in schema]
        schema_str = ", ".join(schema_columns)

       
        model = GenerativeModel("gemini-1.0-pro")
        chat = model.start_chat()

       
        prompt = (f"Given the following schema: {schema_str}, "
                  f"convert the natural language question into an SQL query that returns the following fields: "
                  f"'brand', 'model', 'manufactured_year' AS 'year' and 'price_in_numbers' AS 'base_price'. "
                  f"The table is 'cars_details_main_cleaned'. "
                  f"do not use GROUP BY clause"
                  f"do not give any extra details except the query"
                  f"Question: '{question}'. Give the SQL query in a single line.")

        response = chat.send_message(prompt, stream=False)
        sql_query = response.text.strip()

       
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

     
        print(f"Generated SQL query: {sql_query}")

       
        car_details = spark.sql(sql_query).collect()

      
        if not car_details:
            return "No data found for the specified car."

       
        car_info = car_details[0]  
        invoice_number = str(uuid.uuid4())
        quote_date = datetime.now().strftime('%Y-%m-%d')
        car_details_dict = {
            'brand': car_info['brand'],
            'model': car_info['model'],
            'year': car_info['year'], 
            'base_price': car_info['base_price'],
        }
        
       
        return {
            'customer_name': customer_name,
            **car_details_dict
        }

    except Exception as e:
        return f"Error processing request: {str(e)}"


# COMMAND ----------

dbutils.widgets.text("customer_name", "")
dbutils.widgets.text("question", "")

# COMMAND ----------

customer_name = dbutils.widgets.get("customer_name")
question = dbutils.widgets.get("question")

# COMMAND ----------

quote_details = estimate_car_price_from_natural_language(customer_name, question)
dbutils.notebook.exit(quote_details)
