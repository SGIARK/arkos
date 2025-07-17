#!/bin/bash

# Define the API endpoint and model
API_URL="http://localhost:8000/v1/chat/completions"
MODEL_NAME="tgi"

echo "--- Testing Non-Streaming Chat Completion ---"
echo "Sending request to $API_URL with stream=false"

curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "'"$MODEL_NAME"'",
       "messages": [
         {
           "role": "user",
           "content": "Tell me a short, funny story about a robot."
         }
       ],
       "stream": false,
       "temperature": 0.7
     }'

echo -e "\n\n--- Testing Streaming Chat Completion ---"
echo "Sending request to $API_URL with stream=true"
echo "This will stream the response line by line."

curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{
       "model": "'"$MODEL_NAME"'",
       "messages": [
         {
           "role": "user",
           "content": "Tell me a short story about a brave knight and a wise dragon."
         }
       ],
       "stream": true,
       "temperature": 0.7
     }'

echo -e "\n\n--- Test Calls Complete ---"

