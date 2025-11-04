from dotenv import load_dotenv
import os

load_dotenv() 

DUNE_API_KEY = os.getenv("DUNE_API_KEY")
DATA_PATH = os.getenv("DATA_PATH", "./data")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def show_config():
    print(f"DUNE_API_KEY={DUNE_API_KEY}")
    print(f"DATA_PATH={DATA_PATH}")
    print(f"DEBUG={DEBUG}")

if __name__ == "__main__":
    show_config()
