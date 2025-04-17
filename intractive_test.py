import requests
import json
import sys # Import sys for standard input/output handling

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

def ask_question(query, base_url="http://127.0.0.1:5005"):
    """
    Sends a question to the /ask endpoint.

    Args:
        query (str): The question to ask.
    """
    url = f"{base_url}/ask"
    # Removed the print statement from here to make interactive mode cleaner
    # print(f"Testing: {url} with query: '{query}'")
    try:
        response = requests.get(url, params={"query": query})
        response.raise_for_status()
        data = response.json()
        # print(f"Response: {data}") # Keep commented out for cleaner interaction
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}") # More user-friendly error
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from server.")
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
    # --- Initial Health Check ---
    print("Checking server health...")
    health_info = health_check()
    if not health_info or health_info.get('status') != 'ok':
        print("Server is not healthy or unreachable. Please check the server and try again.")
        sys.exit(1) # Exit if server isn't healthy
    print("Server health check passed. Ready to chat!")
    print("Type your question and press Enter. Type 'quit' or 'exit' to leave.")
    print("-" * 50)

    # --- Interactive Loop ---
    while True:
        try:
            # Get user input
            user_query = input("You: ")

            # Check for exit command
            if user_query.lower() in ['quit', 'exit']:
                break

            # Handle empty input
            if not user_query.strip():
                continue

            # Send the question to the server
            response_data = ask_question(user_query)

            # Process and print the response
            if response_data:
                answer_text = response_data.get('answer', "Sorry, I didn't receive a proper answer.")
                # Handle if the answer is unexpectedly a list (like in your original code)
                if isinstance(answer_text, list):
                    answer_text = answer_text[0] if answer_text else "Received an empty answer list."

                # Ensure answer_text is a string before stripping
                if not isinstance(answer_text, str):
                     answer_text = str(answer_text)

                print(f"AI:  {answer_text.strip()}")
            else:
                # Error message was already printed by ask_question in case of failure
                print("AI:  Sorry, I couldn't get a response from the server.")

            print("-" * 10) # Small separator for clarity

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            break

    print("\n" + "=" * 50)
    print("Exiting interactive session. Goodbye!")