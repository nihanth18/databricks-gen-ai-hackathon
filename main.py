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
  "type": "service_account",
  "project_id": "vertex-ai-test-436713",
  "private_key_id": "dbd68a85074cf8877404625172d9a661e8080592",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9SflTnXBF5SYJ\n96msOTMkOV9VgRnwTBGfFeSCP8HfLIoOrO4bd9KIAfolTDqjJwKZPwFWBG4+PKPI\nV0/Qd63/ECqhi84z5IhrC+eNEoUl2Ip9Cky29wQhLFh3IUHx1ko1JAQh77YVUbQq\nnzECtDV9AL5xE5tBR37ZLBg4elCprx07FYlMAH0MZTK0INlVsNtx+hEY7rqNlN46\nT1jQ2xa/x/LAT3dHtDkWgU4Q+70z+U14BHokAid1pTjd6gUd5e79gdJu08SnJSF0\nBQgds3N+zQCQYukDzYLadci4sTV45b9chPsZFvhD9PlcgWAeNMF4HP244/JdWToY\n9SB0jgg3AgMBAAECggEAI0diuzVIJBbBwLDShEFC6VjyDjcaFQGwdRR4+teUNBqS\nfoq3dBIggG3R59qtwRO3tcUa5CO+QkY+BkxTmVh86uFh+KrcWyqYTE8zMEVH4JZb\ntVNSmfUu5VnlAQsHReH5Aa99P6/0IUQRjrINyeG5aFtXoy6SzJddKm+/8eLGHuGs\nOMFQd7TKb2KGtmrKaKpwuCTDSw99RkN8pjkpBEgqoSn2FB8vZKfG2SruILTxCXaI\n7fAJd6XGrriRvVis2t9t4HYFW9w8ICN2QRpcn6g8rY+uUfj9p7M/R4s2VN5HfaLa\n5lNIBIrrP5poJrk3fwCCM+YeLTPqFkZBLL/gL8UVbQKBgQD3AbzzPdIoPVXmykx/\nV742HENER3mfOrtAhR29zmblwGQWqh6KBFNnjgjAUtVq/3BaqkZ5Dh4tacTDY1vn\nKItrMI6GJuABw+gy5vgYa9hlboNVDPkZmV/cnnl6bPnOKVHXO/2ZEULyh9LKYPEA\ni+Np3UHJ0xtqZZ4OlWJWs7WxIwKBgQDELkTD2Yb+fJvqt/f8Jnwwj8p9+avYBRvY\n7BP/DqCFWXVK0fm7VRsOsRVBc0NH6p9jEcu5RERb3wUYdJAP3NjOhUjFdM1FW7JA\nGZN/MtP0EY8JBx8pw8Ge7JK//rjqYLmnEUcQYRJY83C13r+evzi+Lb/Bd4SI+EIn\n0I26WHm/3QKBgDOlamOVroZ5ZKev7tTFfOEFgc8Z/sUbW6G+85wHNx6c3pCam24S\nP3osiYnlB/iqVkyuw7N2DztBnUGZWdL4eEY+Td6g7D+SPc++2WsJyaJTvCQhZUhD\n+HZBsHa3qHfBzfnp8jl6EXxyh9GG+X06wp8VAzd264mQm77C31/vjXLvAoGAT2s3\ntV2Dc9S4Pf456x8dWX9shzEi6zGuQ1PXCINAYpuDi0WT5s2eRYVUyIlD7IJEAhQ7\nOAY18kdHxc2yYGmUb46vlhgh3XkwoRh5iJ3oBj9xe0Mhz4OLB65X/le9Pkzn+4VY\nEn5psg7jSw0g38Aj1YGpxkV/Jv/xsDKSnaShrRUCgYEA68jSVWCEVF8pPpjaE/Kx\nzk4XIBuRX0tGE1sPFDECzgmW5Yf7Rnl4hliIfYRH4k8dZ0w/sJgShVXHk6l9/DKI\nOQ/EMsNgmrd4+O0Rd9FmiUCIAulE3OjPeOxrbj1RNX1tD4sg0SVS8PleFYhiRJKm\nEFdAupy+CoV5tKlZS6uOjB4=\n-----END PRIVATE KEY-----\n",
  "client_email": "911822883978-compute@developer.gserviceaccount.com",
  "client_id": "101157026097934634388",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/911822883978-compute%40developer.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


# COMMAND ----------

credentials = service_account.Credentials.from_service_account_info(service_account_info)


vertexai.init(  
    project="vertex-ai-test-436713",
    location="us-central1",  
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
