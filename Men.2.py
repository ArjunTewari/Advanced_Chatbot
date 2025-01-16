import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
_=load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "gpt-3.5-turbo-0125")

from langchain.memory import ConversationBufferMemory

summary_memory = ConversationBufferMemory(llm = llm, max_token_limit = 500)
convo_memory = ConversationBufferMemory(llm = llm, max_token_limit = 500)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
def load_faiss_index(file_path="faiss_index"):
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
# Add an instruction to guide the assistant
system_instruction = "You are a nice, helpful assistant"

vectorstore = FAISS.from_texts([""], embedding=embeddings)

def inspect_faiss_index(vectorstore):
    """
    Inspect the texts and metadata stored in the FAISS index.
    """
    # Retrieve stored texts and their metadata
    stored_texts = vectorstore.docstore._dict  # Access stored texts
    for key, value in stored_texts.items():
        print(f"ID: {key}")
        print(f"Text: {value.page_content}")
        print(f"Metadata: {value.metadata}")
        print("-" * 50)
def retrieve_user_summary(vectorstore, user_id):
    """Retrieve the latest summary for a specific user."""
    results = vectorstore.similarity_search("", k=1, filter={"user_id": user_id, "type": "summary"})
    return results[0].page_content if results else ""

summary_instruction = "You are a concise assistant. Generate a brief summary of the conversation, and who gives contexual answers using the summary stored and the previous recent messages"
def store_final_summary(vectorstore, user_id):
    """Summarize the entire conversation in memory and store it."""
    # Retrieve the entire conversation from memory
    conversation_history = summary_memory.load_memory_variables({})["history"]

    # Use LLM to summarize the conversation
    summary_prompt = f"{summary_instruction}\n\nConversation History:\n{conversation_history}\n\nSummary:"
    final_summary = llm.invoke(summary_prompt).content.strip()

    # Store the summary in FAISS
    vectorstore.add_texts(
        [final_summary],
        metadatas=[{"user_id": user_id, "type": "summary"}]
    )
    print("\nConversation summarized and stored in FAISS.")

def summarize_and_answer(vectorstore, user_id, user_input):
    existing_summary = retrieve_user_summary(vectorstore, user_id)
    convo_memory.chat_memory.add_user_message(user_input)
    full_prompt = f"{system_instruction}\n{existing_summary}\nUser: {user_input}\n{convo_memory}\nAI:"
    response = llm.invoke(full_prompt).content
    summary_memory.chat_memory.add_user_message(user_input)
    summary_memory.chat_memory.add_ai_message(response)

    return response

def save_faiss_Index(vectorstore, file_path = "faiss_index"):
    vectorstore.save_local(file_path)
    print("Chat saved")
    print(f"FAISS index saved at: {file_path}")
    return

def clear_faiss_index(file_path="faiss_index"):
    """Clear the FAISS index by replacing it with an empty one."""
    global vectorstore
    vectorstore = FAISS.from_texts([""], embedding=embeddings)
    # save_faiss_Index(vectorstore, file_path)
    print("All data has been deleted from the FAISS database.")


try:
    vectorstore = load_faiss_index(file_path="faiss_index")
    print("Loaded FAISS index from disk.")
except Exception as e:
    print(f"No existing FAISS index found. Initializing a new one.")



user_id = input("Enter user_id :")
# prompt = st.chat_input("Say something")
while True:
    user_input = input("Say something : ")
    if user_input == "exit":
        store_final_summary(vectorstore, user_id)
        break
    if user_input == "delete data":
        clear_faiss_index()
        break
    response = summarize_and_answer(vectorstore, user_id, user_input)
    print(f"AI: {response}")

save_faiss_Index(vectorstore, file_path="faiss_index")

inspect_faiss_index(vectorstore)
