import requests
import json

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
    Tests the /ask endpoint.

    Args:
        query (str): The question to ask.
    """
    url = f"{base_url}/ask"
    print(f"Testing: {url} with query: '{query}'")
    try:
        response = requests.get(url, params={"query": query})
        response.raise_for_status()
        data = response.json()
        # print(f"Response: {data}")
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
    print(f"Testing: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Response: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response")
        return None


if __name__ == "__main__":
    # --- Test /health endpoint ---
    health_check_response = health_check()
    print("\n" + "=" * 50 + "\n")

    # --- Test /reload_resource endpoint ---
    #  Removed from main.py, but keeping the test in case you want to add it back
    # reload_response = reload_resources()
    # print("\n" + "=" * 50 + "\n")

    # --- Test /ask endpoint ---
    questions = [
        "I am at Server Room 1, what should I do?",
        "How many users can I afford to add before the budget gets too low for the other settings?",
        "What happens to the server load if I increase the request frequency by one level?",
        "How can I tell if I'm about to exceed the budget before finalizing my settings?",
        "What does the pie chart on the Server Load Display tell me about my current setup?",
        "If I accidentally overspend, can I undo user additions to recover budget?",
        "How do I match the Request Frequency and Traffic Volume sliders to exactly level 9?",
        "I am at Server Room 5, how can i succees here?",
        "I am at Server Room 4, how can i succees here?",
        "I am at Server Room 3, how can i succees here?"
    ]

    for question in questions:
        response = ask_question(question)
        if response:
            answer_text = response.get('answer', "No answer in response")
            if isinstance(answer_text, list):
                answer_text = answer_text[0]  # Get the first element if it's a list
            query = response.get('query', question)  # Fallback to original question
            metadata = response.get('timings', {}) # Get timings, default to {}

            print(f"Q: {query}\nA: {answer_text.strip()}\n")

            # if metadata:
            #     print("⏳ Response Metadata:")
            #     for key, value in metadata.items():
            #         print(f"   - {key.replace('_', ' ').capitalize()}: {value:.4f} seconds")

        print("\n" + "=" * 50 + "\n")
