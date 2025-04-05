import openai
import os
import time

# Replace with your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_assistant():
    """Create a new assistant"""
    assistant = openai.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a helpful document assistant. Help users answer questions relating to passport renewal.",
        tools=[],
        model="gpt-4-turbo-preview"
    )
    return assistant

def create_thread():
    """Create a new thread"""
    thread = openai.beta.threads.create()
    return thread

def add_message_to_thread(thread_id, content):
    """Add a message to a thread"""
    message = openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    return message

def run_assistant(thread_id, assistant_id):
    """Run the assistant on the thread"""
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    return run

def wait_for_run_completion(thread_id, run_id):
    """Wait for a run to complete"""
    while True:
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        if run.status == "completed":
            return run
        elif run.status in ["failed", "cancelled", "expired"]:
            raise Exception(f"Run ended with status: {run.status}")
        
        print(f"Waiting for run to complete. Current status: {run.status}")
        time.sleep(2)

def get_messages(thread_id):
    """Get messages from a thread"""
    messages = openai.beta.threads.messages.list(
        thread_id=thread_id
    )
    return messages

def display_message_data(messages):
    for message in reversed(list(messages.data)):
        role = message.role
        content = message.content[0].text.value if hasattr(message.content[0], 'text') else str(message.content[0])
        print(f"{role.upper()}: {content}")

def main():
    # Create an assistant
    assistant = create_assistant()
    print(f"Created assistant with ID: {assistant.id}")

    # Create a thread
    thread = create_thread()
    print(f"Created thread with ID: {thread.id}")

    # Add a message to the thread
    message_content = "How long is a US passport valid for?"
    add_message_to_thread(thread.id, message_content)
    print(f"Added message to thread: '{message_content}'")

    # Run the assistant on the thread
    run = run_assistant(thread.id, assistant.id)
    print(f"Started run with ID: {run.id}")

    # Wait for the run to complete
    run = wait_for_run_completion(thread.id, run.id)
    print(f"Run completed with status: {run.status}")

    # Get messages
    messages = get_messages(thread.id)
    print("\nConversation:")
    
    display_message_data(messages)

    # Add a message to the thread
    message_content = "I live in Lexington, MA.  Where can I go to get my passport renewed?"
    add_message_to_thread(thread.id, message_content)
    print(f"Added message to thread: '{message_content}'")

    # Run the assistant on the thread
    run = run_assistant(thread.id, assistant.id)
    print(f"Started run with ID: {run.id}")

    # Wait for the run to complete
    run = wait_for_run_completion(thread.id, run.id)
    print(f"Run completed with status: {run.status}")

    # Get messages
    messages = get_messages(thread.id)
    print("\nConversation:")
    
    display_message_data(messages)

    # Add a message to the thread
    message_content = "I am a US citizen.  How much does it cost to renew my passport?"
    add_message_to_thread(thread.id, message_content)
    print(f"Added message to thread: '{message_content}'")

    # Run the assistant on the thread
    run = run_assistant(thread.id, assistant.id)
    print(f"Started run with ID: {run.id}")

    # Wait for the run to complete
    run = wait_for_run_completion(thread.id, run.id)
    print(f"Run completed with status: {run.status}")

    # Get messages
    messages = get_messages(thread.id)
    print("\nConversation:")
    
    display_message_data(messages)

if __name__ == "__main__":
    main()