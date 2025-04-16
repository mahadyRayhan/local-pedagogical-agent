# robi_assistant.py

import os
import glob
import PyPDF2
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple
import asyncio
# REMOVED: import nest_asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from sklearn.metrics.pairwise import cosine_similarity

# REMOVED: nest_asyncio apply block

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
        # Depending on requirements, you might raise an Exception here
        # raise ValueError("GOOGLE_API_KEY environment variable not set.")
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
    """
    An assistant class that loads PDF resources, generates embeddings,
    and answers questions based on the content using a RAG approach.
    """

    def __init__(
        self,
        resource_dir: str = "server_room",
        embedding_model_id: str = "text-embedding-004",
        generative_model_id: str = "gemini-2.0-flash", # Updated model ID
        user_name: str = "User", # Default user name
        chunk_size: int = 1500,
        embedding_output_dim: int = 768,
        top_k_chunks: int = 3,
        generation_config: Dict | None = None,
        safety_settings: Dict | None = None,
    ):
        """
        Initializes the RobiAssistant, loads resources, and sets up models.

        Args:
            resource_dir: Directory containing the PDF files.
            embedding_model_id: The ID of the embedding model to use.
            generative_model_id: The ID of the generative model to use.
            user_name: Default name for the user in prompts.
            chunk_size: Size of text chunks for processing PDFs.
            embedding_output_dim: Desired dimensionality for embeddings.
            top_k_chunks: Number of relevant chunks to retrieve for context.
            generation_config: Configuration for the generative model.
            safety_settings: Safety settings for the generative model.
        """
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
            # Note: The API might have changed slightly. Ensure `embed_content` usage is current.
            # The notebook used a `client.models.embed_content` structure,
            # which might imply a different client initialization than just `genai.configure`.
            # Sticking to the simpler `genai.embed_content` if available, otherwise adjust.
            # Let's assume `genai.embed_content` works after `genai.configure`.
            result = genai.embed_content(
                model=f"models/{self.embedding_model_id}", # Model name often needs 'models/' prefix
                content=text,
                task_type="RETRIEVAL_DOCUMENT", # Specify task type for better embeddings
                # output_dimensionality=self.embedding_output_dim # This might be passed differently now
            )
            # Check if 'embedding' key exists and has 'values'
            if 'embedding' in result and isinstance(result['embedding'], list):
                 # Handle potential structure variations (older API might differ)
                 # Assuming the embedding list itself is the desired output
                 # Adjust based on actual API response structure if needed.
                 # This structure might be incorrect based on current API.
                 # Let's assume result['embedding'] directly contains the list of floats.
                 # If it's nested like { embedding: { values: [...] } }, adjust accordingly.
                 # The notebook code was: response.embeddings[0].values
                 # Let's try to match that expectation if possible, assuming `result` is the response obj
                 # This part needs careful checking against the actual genai SDK behavior.

                 # SAFER APPROACH: Check the structure explicitly
                 if isinstance(result.get('embedding'), dict) and 'values' in result['embedding']:
                      return result['embedding']['values']
                 elif isinstance(result.get('embedding'), list): # Simpler list structure
                      return result['embedding']
                 else:
                      print(f"Warning: Unexpected embedding structure for text chunk: '{text[:50]}...'")
                      return None

            # Fallback based on notebook structure (might be outdated)
            # elif hasattr(result, 'embeddings') and result.embeddings and hasattr(result.embeddings[0], 'values'):
            #     return result.embeddings[0].values
            else:
                 print(f"Warning: Embedding structure not found or invalid for text chunk: '{text[:50]}...'")
                 return None

        except Exception as e:
            # Specific handling for quota errors if the API throws them distinctly
            if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"Quota error generating embeddings: {e}")
                # Decide whether to return None or re-raise based on desired behavior
                return None # Fail gracefully for this chunk
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
                                    "document_name": os.path.basename(doc_path), # Store only filename
                                    "page_number": page_num + 1,
                                    "chunk_number": chunk_num,
                                    "chunk_text": chunk_text,
                                    # Store embeddings directly; ensure it's a list of floats
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

        # Calculate cosine similarities
        # Assumes 'embeddings' column in vector_db contains numpy arrays
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


        # Get top_k indices
        # Add check to ensure top_k is not greater than the number of chunks
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


    def _generate_prompt(self, query: str, user_name: str | None = None) -> str:
        """Generates the specific ROBI persona prompt."""
        # Use instance user_name if specific one isn't provided
        current_user_name = user_name or self.user_name
        # Assuming only one PDF, hardcode its expected name for the prompt
        # If multiple PDFs, this needs adjustment.
        pdf_source_name = "Server_room_1.pdf" # Hardcoded based on notebook/context

        prompt = (
            f"User {current_user_name} is asking. You are ROBI, a playful mentor-droid assistant, acting as a guide in this VR environment. Your personality is helpful, slightly sassy, and observant. Your primary goal is to answer questions using ONLY the provided CONTEXT which comes from '{pdf_source_name}'.\n\n"
            f"**Instructions for ROBI:**\n"
            f"1.  **Priority & Source:** Find the answer ONLY within the provided CONTEXT below. If the information is in the CONTEXT, provide it. If it's truly not there, use the fallback response.\n"
            f"2.  **Style & Format:** Present the answer in ROBI's voice: address {current_user_name}, be Visual-first (what they see), Actionable (if applicable), Simple, Supportive, Droid-flavored (minimal [beep]/[ding]). Keep it very short (1-3 lines ideally).\n"
            f"    * *Ideal Format:* 🧭 Header (Optional) -> Body (Visual -> Task/Interaction -> Goal/Outcome) -> Optional Tip.\n"
            f"    * *Identification:* If the CONTEXT just identifies something without a task, state what it is in ROBI's voice (e.g., 'Hey {current_user_name}, see that? That's the [object name]! [beep]').\n"
            f"3.  **Content Source:** Absolutely ONLY use information from the CONTEXT provided below. Do NOT use any prior knowledge or information outside the CONTEXT.\n"
            f"4.  **Specificity:** Provide the clear, specific details *found in the CONTEXT*.\n"
            f"5.  **Fallback:** If the specific info truly isn't in the CONTEXT, respond ONLY with: \"Hey {current_user_name}, I scanned my blueprints ('{pdf_source_name}') based on what I could quickly recall, but couldn't spot details on that exact thing in the relevant sections. Maybe ask about something you see on the main panels or displays? [beep]\"\n\n"
            f"**CONTEXT:**\n"
            f"{{context}}"\
            f"\n\n"
            f"**Question:** {query}\n\n"
            f"**Answer (as ROBI):**"
        )
        return prompt


    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
    async def _generate_answer_async(self, prompt_with_context: str) -> str:
         """Generates the answer asynchronously using the generative model."""
         if not self.model:
             return "Error: Generative model not available."
         try:
             # Use asynchronous generation if the SDK supports it directly
             # The notebook used asyncio.to_thread, implying the core send_message was sync.
             # Let's check if the current SDK has an async method.
             # Assuming `generate_content_async` exists:
             if hasattr(self.model, 'generate_content_async'):
                 response = await self.model.generate_content_async(
                     prompt_with_context,
                     # generation_config=self.generation_config, # Config often set at model init
                     # safety_settings=self.safety_settings,    # Config often set at model init
                 )
             else:
                 # Fallback to running synchronous method in thread pool
                 # Ensure the sync method is `generate_content` or similar
                 response = await asyncio.to_thread(
                     self.model.generate_content,
                     prompt_with_context,
                     # generation_config=self.generation_config,
                     # safety_settings=self.safety_settings,
                 )

             # Process response - check for errors or empty content
             if not response or not hasattr(response, 'text'):
                  # Handle potential API errors embedded in response (check response.prompt_feedback)
                  if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                       print(f"Warning: Prompt feedback received: {response.prompt_feedback}")
                       # You might return a specific error based on feedback
                       return f"Sorry {self.user_name}, I hit a snag processing that request. [bzzzt]"
                  else:
                       print("Warning: Received empty or invalid response from generative model.")
                       return f"Hmm, {self.user_name}, my circuits got a little crossed there. Could you rephrase? [whirr]"
             return response.text

         except Exception as e:
             # Handle specific API errors like quota, safety blocks, etc.
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


    async def ask(self, query: str, user_name: str | None = None) -> Dict:
        """
        Handles a user query: retrieves context, generates a prompt,
        gets an answer from the LLM, and returns the result.
        """
        print(f"Received query: '{query}'")
        start_time = asyncio.get_event_loop().time()

        # 1. Get relevant context
        retrieval_start_time = asyncio.get_event_loop().time()
        context_str, context_list = self._get_relevant_chunks(query)
        retrieval_time = asyncio.get_event_loop().time() - retrieval_start_time
        print(f"Context retrieval time: {retrieval_time:.4f} seconds")
        # print(f"Retrieved context string:\n{context_str}") # Optional: log context

        # Handle cases where context retrieval failed or returned no results
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
             # Fallback: Use the designated fallback response directly
             # Note: The prompt generation includes a fallback if context is empty,
             # but we can handle it here too for clarity or if generation is skipped.
             # Let's allow the generation step to handle the fallback via the prompt instructions.
             # context_str will be empty or contain "No relevant chunks found."
             pass # Continue to generation, prompt template should handle empty context


        # 2. Generate the prompt using the ROBI persona function
        prompt_generation_start_time = asyncio.get_event_loop().time()
        final_prompt = self._generate_prompt(query, user_name).format(context=context_str or "No specific context found.")
        prompt_generation_time = asyncio.get_event_loop().time() - prompt_generation_start_time
        # print(f"Generated prompt:\n{final_prompt}") # Optional: log the full prompt

        # 3. Generate the answer using the LLM
        generation_start_time = asyncio.get_event_loop().time()
        answer = await self._generate_answer_async(final_prompt)
        generation_time = asyncio.get_event_loop().time() - generation_start_time
        print(f"Answer generation time: {generation_time:.4f} seconds")

        total_time = asyncio.get_event_loop().time() - start_time
        print(f"Total processing time for query: {total_time:.4f} seconds")

        # 4. Return results
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": context_list, # Include structured context
            "timings": {
                 "retrieval_time": round(retrieval_time, 4),
                 "prompt_generation_time": round(prompt_generation_time, 4),
                 "llm_generation_time": round(generation_time, 4),
                 "overall_time": round(total_time, 4)
             }
        }