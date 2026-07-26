from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import time

# Load environment variables
load_dotenv()

# Initialize the Groq chat model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# List of questions to send to the model
questions = [
    "What is AI?",
    "What is Generative AI?",
    "What is Agentic AI?"
]

# Track total execution time
total_start = time.time()

for question in questions:

    # Start timer for the current request
    start = time.time()

    # Create conversation messages
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Answer every question in one line."),
        HumanMessage(content=question)
    ]

    # Generate response
    response = llm.invoke(messages)

    # Stop timer
    end = time.time()

    # Display response
    print("Question:", question, end="\n\n")
    print("Answer:", response.content, end="\n\n")
    print(f"Time Taken: {end - start:.2f} seconds", end="\n\n")

    # Display token usage
    print("Input Tokens:", response.usage_metadata["input_tokens"])
    print("Output Tokens:", response.usage_metadata["output_tokens"])
    print("Total Tokens:", response.usage_metadata["total_tokens"])

# Display total execution time
total_end = time.time()

print(f"Total Execution Time: {total_end - total_start:.2f} sec")