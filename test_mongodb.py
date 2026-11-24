
import pymongo
from pymongo import MongoClient

# Replace this with your MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://tanjir233159_db_user:alamtanjir123@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Test connection
    client.admin.command("ping")

    print("MongoDB connection successful!")

    # Show databases
    print("Available databases:", client.list_database_names())

except Exception as e:
    print("MongoDB connection failed!")
    print("Error:", e)
