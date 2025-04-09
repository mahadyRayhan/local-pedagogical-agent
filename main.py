# main.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import asyncio # Import asyncio
from typing import Any, List, Dict, Tuple

# Import the assistant class
from robi_assistant import RobiAssistant

# --- Load Environment Variables ---
# Although robi_assistant does it, good practice to do it early here too
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ROBI Assistant API",
    description="API to interact with the ROBI assistant based on PDF documents.",
    version="1.0.0",
)

# --- Global variable to hold the assistant instance ---
robi_instance: RobiAssistant | None = None

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize the RobiAssistant when the server starts."""
    global robi_instance
    print("Server starting up...")
    try:
        # Initialize the assistant - this loads documents and builds the index
        robi_instance = RobiAssistant(
            resource_dir="server_room", # Directory with Server_room_1.pdf [cite: 1]
            user_name="Rayhan" # Set default user name if desired [cite: 1]
            # Add other parameters if needed (e.g., model IDs)
        )
        print("RobiAssistant initialized successfully.")
    except Exception as e:
        print(f"FATAL: Failed to initialize RobiAssistant during startup: {e}")
        # Decide if the server should run without the assistant
        robi_instance = None
        # Consider raising the exception again if the assistant is critical
        # raise e

# --- API Endpoints ---
@app.get("/ask", tags=["Query"], response_model=Dict) # Define response model if possible
async def ask_question(
    query: str = Query(..., description="The question to ask ROBI assistant.")
):
    """
    Sends a query to the ROBI assistant and returns the answer.
    """
    if robi_instance is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="ROBI Assistant is not available due to initialization failure."
        )
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    try:
        # Call the assistant's ask method asynchronously
        result = await robi_instance.ask(query) # Pass the query

        # Return the entire result dictionary
        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        # Provide a more generic error message to the client
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the query.")

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    if robi_instance and robi_instance.vector_db is not None and not robi_instance.vector_db.empty:
         status = "ok"
         details = f"RobiAssistant initialized with {len(robi_instance.vector_db)} chunks."
         status_code = 200
    elif robi_instance and (robi_instance.vector_db is None or robi_instance.vector_db.empty):
         status = "degraded"
         details = "RobiAssistant initialized but failed to load/process resources correctly."
         status_code = 503 # Service Unavailable might be appropriate
    else:
        status = "error"
        details = "RobiAssistant failed to initialize."
        status_code = 503

    return JSONResponse(content={"status": status, "details": details}, status_code=status_code)


# --- Running the Server ---
if __name__ == "__main__":
    # Use uvicorn to run the FastAPI application
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 5005)),
        reload=True # Set reload=False for production
        # Consider adding --log-level info for more visibility
        )