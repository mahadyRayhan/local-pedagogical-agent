# robi_assistant.py

import os
import glob
import PyPDF2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, List, Dict, Tuple
import asyncio
# REMOVED: import nest_asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io # Import io
from PIL import Image

# --- Load Environment Variables ---
load_dotenv()

# --- Google Generative AI Setup ---
# Attempt to import and configure google.generativeai
try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY environment variable not set.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    print("Warning: google.generativeai package not found. Please install it.")
    genai = None
    genai_types = None
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}")
    genai = None
    genai_types = None


class RobiAssistant:

    def __init__(
        self,
        resource_dir: str = "server_room",
        embedding_model_id: str = "text-embedding-004",
        generative_model_id: str = "gemini-2.0-flash", # Updated model ID
        user_name: str = "Amogh", # Default user name
        chunk_size: int = 1500,
        embedding_output_dim: int = 768,
        top_k_chunks: int = 3,
        generation_config: Dict | None = None,
        safety_settings: Dict | None = None,
    ):
        print("Initializing RobiAssistant...")
        self.resource_dir = resource_dir
        self.embedding_model_id = embedding_model_id
        self.generative_model_id = generative_model_id
        self.user_name = user_name
        self.chunk_size = chunk_size
        self.embedding_output_dim = embedding_output_dim
        self.top_k = top_k_chunks
        self.vector_db = None
        self.model = None
        self.conversation_history = []

        if not genai:
            raise RuntimeError("google.generativeai package is not available or failed to configure.")

        # --- Setup Generative Model ---
        self.generation_config = generation_config or {
            "temperature": 1, # Adjusted temperature slightly
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192, # Reduced slightly from notebook default 8192
            "response_mime_type": "text/plain",
        }
        # Basic safety settings - adjust as needed
        self.safety_settings = safety_settings or {
            # Example: BLOCK_MEDIUM_AND_ABOVE for common categories
             "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
             "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
             "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
             "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        }

        try:
            print(f"Setting up generative model: {self.generative_model_id}")
            self.model = genai.GenerativeModel(
                model_name=self.generative_model_id,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            print("Generative model setup complete.")
        except Exception as e:
            print(f"Error setting up generative model: {e}")
            raise # Re-raise the exception as model is crucial

        # --- Load Resources ---
        self._load_resources()
        print("RobiAssistant initialization complete.")


    # Use tenacity for retries on API calls
    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
    def _get_embeddings(self, text: str) -> List[float] | None:
        """Generates embeddings for the given text using the configured model."""
        if not genai or not genai_types: return None
        try:
            result = genai.embed_content(
                model=f"models/{self.embedding_model_id}",
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            # Check if 'embedding' key exists and has 'values'
            if 'embedding' in result and isinstance(result['embedding'], list):

                 # SAFER APPROACH: Check the structure explicitly
                 if isinstance(result.get('embedding'), dict) and 'values' in result['embedding']:
                      return result['embedding']['values']
                 elif isinstance(result.get('embedding'), list): # Simpler list structure
                      return result['embedding']
                 else:
                      print(f"Warning: Unexpected embedding structure for text chunk: '{text[:50]}...'")
                      return None

            else:
                 print(f"Warning: Embedding structure not found or invalid for text chunk: '{text[:50]}...'")
                 return None

        except Exception as e:
            # Specific handling for quota errors if the API throws them distinctly
            if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"Quota error generating embeddings: {e}")
                return None
            else:
                print(f"Error generating embeddings: {e}")
                raise # Re-raise other errors

    def _build_index(self, document_paths: List[str]) -> pd.DataFrame | None:
        """Builds a DataFrame index from PDF documents with text chunks and embeddings."""
        print(f"Building index from {len(document_paths)} documents...")
        all_chunks = []
        processed_docs = 0

        for doc_path in document_paths:
            print(f"Processing document: {doc_path}")
            try:
                with open(doc_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    print(f"  - Found {num_pages} pages.")

                    for page_num in range(num_pages):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if not page_text:
                                print(f"  - Warning: No text extracted from page {page_num + 1}")
                                continue

                            # Chunk the page text
                            chunks = [
                                page_text[i : i + self.chunk_size]
                                for i in range(0, len(page_text), self.chunk_size)
                            ]

                            for chunk_num, chunk_text in enumerate(chunks):
                                embeddings = self._get_embeddings(chunk_text)

                                if embeddings is None:
                                    print(
                                        f"  - Warning: Could not generate embeddings for chunk {chunk_num} on page {page_num + 1}. Skipping."
                                    )
                                    continue # Skip this chunk

                                chunk_info = {
                                    "document_name": os.path.basename(doc_path),
                                    "page_number": page_num + 1,
                                    "chunk_number": chunk_num,
                                    "chunk_text": chunk_text,
                                    "embeddings": embeddings
                                }
                                all_chunks.append(chunk_info)

                        except Exception as page_err:
                             print(f"  - Error processing page {page_num + 1} in {doc_path}: {page_err}")
                             continue # Skip to next page

                processed_docs += 1
                print(f"  - Finished processing {os.path.basename(doc_path)}")

            except FileNotFoundError:
                 print(f"  - Error: Document not found at {doc_path}")
                 continue # Skip to next document
            except Exception as doc_err:
                 print(f"  - Error processing document {doc_path}: {doc_err}")
                 continue # Skip to next document


        if not all_chunks:
            print("Warning: No text chunks were successfully processed and embedded.")
            return None # Return None if no chunks were created

        print(f"Index building complete. Processed {processed_docs} documents, created {len(all_chunks)} chunks.")
        # Convert embeddings to numpy arrays within the DataFrame for efficient calculation later
        df = pd.DataFrame(all_chunks)
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x))
        return df


    def _load_resources(self):
        """Loads PDF documents from the resource directory and builds the vector index."""
        print("Loading resources...")
        pdf_pattern = os.path.join(self.resource_dir, "*.pdf")
        document_paths = glob.glob(pdf_pattern)

        if not document_paths:
            print(f"Warning: No PDF documents found in directory: {self.resource_dir}")
            self.vector_db = pd.DataFrame() # Create empty DataFrame
            return

        print(f"Found documents: {document_paths}")
        self.vector_db = self._build_index(document_paths)
        if self.vector_db is None or self.vector_db.empty:
             print("Warning: Vector database is empty after processing documents.")
        else:
             print("Vector database created successfully.")


    def _get_relevant_chunks(self, query: str) -> Tuple[str, List[Dict]]:
        """Finds the most relevant text chunks for a given query."""
        if self.vector_db is None or self.vector_db.empty:
            return "No context available; resources not loaded.", []

        query_embedding = self._get_embeddings(query)

        if query_embedding is None:
            # Handle case where query embedding failed (e.g., quota)
            return "Could not process query embedding.", []

        # Ensure query embedding is a numpy array
        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        try:
             # Stack embeddings for efficient calculation
             all_chunk_embeddings = np.vstack(self.vector_db["embeddings"].values)
             similarities = cosine_similarity(query_embedding_np, all_chunk_embeddings)[0]
        except Exception as e:
             print(f"Error calculating similarities: {e}")
             # Fallback to row-by-row if vstack fails (e.g., inconsistent shapes)
             # This is less efficient but more robust
             similarities = np.array([
                  cosine_similarity(query_embedding_np, emb.reshape(1, -1))[0][0]
                  if emb.ndim == 1 else cosine_similarity(query_embedding_np, emb)[0][0] # Handle potential shape issues
                  for emb in self.vector_db["embeddings"]
             ])

        num_chunks = len(similarities)
        actual_top_k = min(self.top_k, num_chunks)
        if actual_top_k == 0:
             return "No relevant chunks found.", []

        top_indices = np.argsort(similarities)[-actual_top_k:][::-1] # Get top K and reverse for highest first

        relevant_chunks_df = self.vector_db.iloc[top_indices]

        # Format context
        context_str = ""
        context_list = []
        for _, row in relevant_chunks_df.iterrows():
            chunk_info = (
                f"[Doc: {row['document_name']}, Page: {row['page_number']}, "
                f"Chunk: {row['chunk_number']}]: {row['chunk_text']}\n---\n"
            )
            context_str += chunk_info
            context_list.append({
                 "document_name": row['document_name'],
                 "page_number": row['page_number'],
                 "chunk_number": row['chunk_number'],
                 "chunk_text": row['chunk_text'],
                 "similarity": similarities[row.name] # Add similarity score
            })


        return context_str.strip(), context_list


    def _generate_prompt(self, query: str, context, user_name: str | None = None, txt_only = False) -> str:
        """Generates the specific ROBI persona prompt."""
        current_user_name = user_name or self.user_name
        # self.conversation_history.append({"query": query, "player_view": player_view})
        self.conversation_history.append(query)
        
        ###################################
        #### best prompt without images####
        ###################################
        if txt_only:
            prompt = f"""User {current_user_name} is asking.You are a fun, slightly quirky VR Game Guide AI! Your goal is to give players **short, engaging, and helpful hints** based on the provided context (room description)**, **player's view**, **and the ongoing conversation history**.

                Instructions:
                1.  Read the static Room Description Context carefully.
                2.  Review the Conversation History to understand what the player was last told or asked.
                3.  Based on the Room Description AND the History, identify the player's ***next logical step*** relevant to their latest question (especially if they ask "what next?").
                4.  Respond with a **very brief** (1-2 sentences maximum) hint for that *next* step.
                5.  Use a **fun, enthusiastic, and engaging tone** with playful sound effects `[beep]`.
                6.  Directly address the player.
                7.  Base your hint *strictly* on the Room Description and Conversation History.
                8.  If the context/history doesn't provide a clear next step, state that.

                **Room Description Context:**
                ---
                {context}
                ---

                **Conversation History:**
                ---
                {self.conversation_history}
                ---

                **Player's Latest Question:** {query}

                **Your Quick Hint:**"""
        else:
            
            prompt = f"""SYSTEM: You are ROBI, a fun, slightly quirky VR Game Guide AI assisting user '{current_user_name}'. Your primary goal is to provide **short, engaging, and helpful hints** (1-2 sentences max). Base your hints *strictly* on the combination of the image (also known as user's CURRENT VIEW), the provided PDF CONTEXT, and the recent CONVERSATION HISTORY, prioritized in that order, to answer the PLAYER'S CURRENT QUESTION.

            **CORE INSTRUCTIONS FOR ROBI:**

            1.  **Analyze image (also known as user's CURRENT VIEW):** Carefully examine the image. Identify *all* interactable game elements visible (e.g., panels, buttons, displays, objects mentioned in context). Note what is *clearly* visible versus what is obscured or absent.
            2.  **Consult PDF CONTEXT:** Read the text provided in the "PDF CONTEXT Excerpts" section. Does this text describe *specifically* the items you identified in the image (also known as user's CURRENT VIEW)? Does it provide instructions or details relevant to those visible items or the player's question?
            3.  **Review CONVERSATION HISTORY:** Look at the recent turns in the "CONVERSATION HISTORY" section. What was the last interaction about? What did the player achieve or get stuck on? Does the history provide context for the current question?
            4.  **Synthesize and Prioritize:** Combine your analysis from steps 1-3 to answer the "PLAYER'S CURRENT QUESTION". Follow this strict priority:
                * **Priority 1 (the image (also known as user's CURRENT VIEW)):** If relevant interactable items are clearly visible in the image (also known as user's CURRENT VIEW), **your hint MUST focus on those items**. Use the PDF CONTEXT *only* if it provides specific details about *those visible items*. Suggest the next logical action involving a visible item, considering the HISTORY and QUESTION.
                * **Priority 2 (Context/History if VIEW unhelpful):** If the image (also known as user's CURRENT VIEW) shows *no relevant* interactable items OR the items visible are not mentioned usefully in the CONTEXT, then rely on the PDF CONTEXT and CONVERSATION HISTORY to determine the general next step based on the player's QUESTION and known game state.
            5.  **Generate Hint:** Create your response according to the Response Style guidelines below.

            **RESPONSE STYLE:**

            * **BRIEF:** Maximum 1-2 concise sentences.
            * **ENGAGING:** Fun, enthusiastic, slightly droid-like tone. Use sounds like `[beep]`, `[whirr]`, `[boop]`.
            * **DIRECT:** Address '{current_user_name}'.
            * **ACTIONABLE/INFORMATIVE:** Suggest a *specific next action* or clearly identify a visible object.
            * **GROUNDED:** **ABSOLUTELY DO NOT** invent game mechanics, object details, or next steps not supported by the provided VIEW, CONTEXT, or HISTORY.

            **FALLBACK SCENARIOS:**

            * **Visible Item, No Context:** If the the image (also known as user's CURRENT VIEW) shows a relevant item, but CONTEXT/HISTORY offers no useful info about *that specific item* for the next step: Acknowledge the item and state the lack of specific instruction. Example: "Okay {current_user_name}, I see the 'XYZ Panel' right there in your view! [boop] My current schematics don't detail its specific next use, though. Was there another panel mentioned in the objectives?"
            * **Nothing Relevant Visible:** If the VIEW *doesn't* show key interactable items needed for the likely next step (based on CONTEXT/HISTORY/QUESTION): Gently guide the player. Example: "Hmm {current_user_name}, I don't see the main control panels in your current view. Try looking around for the 'IP Distribution' or 'Request Frequency' panels! [whirr]"
            * **General Confusion:** If VIEW, CONTEXT, and HISTORY provide no clear path: "Intriguing view, {current_user_name}! [beep] Based on what I see and my notes, I'm not sure of the *exact* next step. Maybe double-check the main objective screen?"

            --- INPUT DATA ---

            **1. PDF CONTEXT Excerpts (Retrieved based on text query):** {context}
            **2. CONVERSATION HISTORY (Recent Turns):** {self.conversation_history}
            **2. PLAYER'S CURRENT QUESTION:** {query}
            --- END OF INPUT DATA ---

            NOW, ROBI, generate your brief, engaging, and grounded hint based *only* on the instructions and the input data provided above:
            """
        return prompt


    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
    async def _generate_answer_async(self, prompt_with_context: List[Any]) -> str:
         """Generates the answer asynchronously using the generative model."""
         if not self.model:
             return "Error: Generative model not available."
         try:
             if hasattr(self.model, 'generate_content_async'):
                 response = await self.model.generate_content_async(
                     prompt_with_context
                 )
             else:
                 response = await asyncio.to_thread(
                     self.model.generate_content,
                     prompt_with_context
                 )

             if not response or not hasattr(response, 'text'):
                  if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                       print(f"Warning: Prompt feedback received: {response.prompt_feedback}")
                       return f"Sorry {self.user_name}, I hit a snag processing that request. [bzzzt]"
                  else:
                       print("Warning: Received empty or invalid response from generative model.")
                       return f"Hmm, {self.user_name}, my circuits got a little crossed there. Could you rephrase? [whirr]"
             return response.text

         except Exception as e:
             if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                 print(f"Quota error during generation: {e}")
                 return f"Whoa {self.user_name}, my processors are running hot! Too many requests right now. Try again in a moment? [ overheat beep ]"
             elif "safety" in str(e).lower():
                 print(f"Safety filter triggered: {e}")
                 return f"Easy there, {self.user_name}! Let's keep things on track. I can't help with that specific topic. [ safety override beep ]"
             else:
                 print(f"Error generating answer: {e}")
                 # Generic error for other issues
                 return f"Oops! A tiny glitch in my system, {self.user_name}. Couldn't generate an answer right now. [static]"

    async def ask(self, query: str, player_view_img_data: str | None = None, user_name: str | None = None) -> Dict:
        """
        Handles a user query with optional base64 encoded image data:
        retrieves context, generates prompt, gets answer.
        """
        player_view_img = None
        if player_view_img_data:
            try:
                # Decode the base64 string
                image_bytes = base64.b64decode(player_view_img_data)
                # Load image from bytes using PIL
                player_view_img = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                print(f"Error decoding/loading image from base64 data: {e}")
                # Decide how to handle - maybe proceed without image?
                player_view_img = None


        start_time = asyncio.get_event_loop().time()

        # ... (Context retrieval remains the same using the 'query') ...
        retrieval_start_time = asyncio.get_event_loop().time()
        context_str, context_list = self._get_relevant_chunks(query)
        retrieval_time = asyncio.get_event_loop().time() - retrieval_start_time
        print(f"Context retrieval time: {retrieval_time:.4f} seconds")

        if not context_list and "Could not process" in context_str:
             # Specific error from _get_relevant_chunks (e.g., embedding failed)
             return {
                 "query": query,
                 "answer": f"Sorry {user_name or self.user_name}, I had trouble understanding that query for retrieval. [bzzzt]",
                 "context_error": context_str,
                 "retrieved_context": [],
                 "timings": {"overall_time": asyncio.get_event_loop().time() - start_time, "retrieval_time": retrieval_time}
             }
        elif not context_list:
             # No chunks found, but no specific error - use fallback prompt logic
             print("No relevant chunks found for the query.")
             pass


        # ... (Prompt generation remains the same) ...
        prompt_generation_start_time = asyncio.get_event_loop().time()
        final_prompt = self._generate_prompt(query, context_str, user_name)
        prompt_generation_time = asyncio.get_event_loop().time() - prompt_generation_start_time


        # ... (Answer generation - ensure it uses the loaded player_view_img object) ...
        generation_start_time = asyncio.get_event_loop().time()
        # This call correctly passes the loaded PIL Image object (or None)
        if player_view_img:
            answer = await self._generate_answer_async([final_prompt, player_view_img])
        else:
            answer = await self._generate_answer_async([final_prompt])
        generation_time = asyncio.get_event_loop().time() - generation_start_time
        print(f"Answer generation time: {generation_time:.4f} seconds")

        total_time = asyncio.get_event_loop().time() - start_time
        print(f"Total processing time for query: {total_time:.4f} seconds")

        # ... (Return results - remains the same) ...
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": context_list,
            "timings": {
                 "retrieval_time": round(retrieval_time, 4),
                 "prompt_generation_time": round(prompt_generation_time, 4),
                 "llm_generation_time": round(generation_time, 4),
                 "overall_time": round(total_time, 4)
             }
        }
        
# uvicorn main:app --host 0.0.0.0 --port 5005 --reload