import sys
import os
import traceback

try:
    import vector_db
    print("SUCCESS: vector_db imported successfully")
except Exception:
    print("FAILURE: Could not import vector_db")
    traceback.print_exc()
