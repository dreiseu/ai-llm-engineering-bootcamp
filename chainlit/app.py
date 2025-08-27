import chainlit as cl


@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a user inputs a message in the UI.
    It sends back an "echo" of what the user sent.

    Args:
        message: The user message.
    """
    
    # This is a simple echo bot - you can replace this with your own logic
    response = f"Echo: {message.content}"
    
    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()


@cl.on_chat_start
async def start():
    """
    This function will be called at the start of every chat session.
    """
    await cl.Message(
        content="Hello! I'm a simple echo bot. Send me a message and I'll echo it back to you!"
    ).send()