import os
import uvicorn
import json
from fastapi import FastAPI, HTTPException
from models import QuestionRequest
from utils import chat_bot

# Initialize app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, Aman!"}

# PDF Q&A endpoint
@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    try:
        # Process the question using chat_bot function
        response_data = chat_bot(question_request.question)  # Returns a string
        
        # Check if the response is valid
        if not response_data:  # In case of None or empty string
            raise HTTPException(status_code=500, detail="Failed to get a valid response")

        # Return response as a plain dictionary
        return {"response": response_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)