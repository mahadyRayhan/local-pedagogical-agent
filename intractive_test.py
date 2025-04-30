import base64
import requests
import json
import sys
import os

# --- Keep your existing functions ---
def reload_resources(base_url="http://127.0.0.1:5005"):
    """
    Tests the /reload_resource endpoint.
    """
    url = f"{base_url}/reload_resource"
    print(f"Testing: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        # print(f"Response: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response")
        return None

def ask_question(query, image_data_base64, base_url="http://127.0.0.1:5005"):
    """
    Sends the query and BASE64 encoded image data via POST request.
    """
    url = f"{base_url}/ask"
    payload = {"query": query, "image_data": image_data_base64}
    headers = {'Content-Type': 'application/json'}
    try:
        # Use POST and send data as JSON
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response")
        return None

def health_check(base_url="http://127.0.0.1:5005"):
    """
    Tests the /health endpoint
    """
    url = f"{base_url}/health"
    # print(f"Testing: {url}") # Less verbose for interactive mode
    try:
        response = requests.get(url, timeout=5) # Add a timeout
        response.raise_for_status()
        data = response.json()
        # print(f"Response: {data}") # Print only if needed or in case of failure
        return data
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response during health check.")
        return None

# --- Main execution block for interactive mode ---    
if __name__ == "__main__":
    print("Checking server health...")
    health_info = health_check()
    if not health_info or health_info.get('status') != 'ok':
        print("Server is not healthy or unreachable. Please check the server and try again.")
        sys.exit(1) # Exit if server isn't healthy
    print("Server health check passed. Ready to chat!")
    print("Type your question, followed by a comma and the image path (e.g., 'What is this?, my_image.png'). Type 'quit' or 'exit' to leave.")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            parts = user_input.rsplit(",", 1)
            query = parts[0].strip()
            image_path_input = parts[1].strip() if len(parts) > 1 else ''

            if not query:
                print("AI: Please enter a question.")
                continue

            encoded_image_data = None
            if image_path_input:
                if os.path.exists(image_path_input):
                    try:
                        with open(image_path_input, "rb") as image_file:
                            # Read image bytes and encode to Base64
                            encoded_image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    except Exception as e:
                        print(f"AI: Error reading or encoding image file '{image_path_input}': {e}")
                        continue # Skip this turn if image reading fails
                else:
                    print(f"AI: Image file not found at path: {image_path_input}")
                    continue # Skip if file doesn't exist

            # Only proceed if we have a query (image is optional, handled by server if None)
            response_data = ask_question(query, encoded_image_data) # Send query and encoded image (or None)

            # ... (rest of the response handling logic - remains the same) ...
            if response_data:
                answer_text = response_data.get('answer', "Sorry, I didn't receive a proper answer.")
                if isinstance(answer_text, list):
                    answer_text = answer_text[0] if answer_text else "Received an empty answer list."
                if not isinstance(answer_text, str):
                     answer_text = str(answer_text)
                print(f"AI:  {answer_text.strip()}")
            else:
                print("AI:  Sorry, I couldn't get a response from the server.")
            print("-" * 10)

        except (KeyboardInterrupt, EOFError):
            break

    print("\n" + "=" * 50)
    print("Exiting interactive session. Goodbye!")
