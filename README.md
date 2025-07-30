# AI/LLM Engineering Kick-off: Key Takeaways

This notebook demonstrates how to use the OpenAI API to interact with GPT-4.1-nano for prompt engineering and reasoning tasks. Below are the main concepts and takeaways:

## Key Concepts

- **OpenAI API Setup**: Learn how to securely set your API key and initialize the OpenAI client in Python.
- **ChatCompletion Model**: Send prompts to the GPT-4.1-nano model using different roles (`developer`, `assistant`, `user`).
- **Helper Functions**: Use Python functions to wrap messages and pretty-print responses for easier interaction.
- **Prompt Engineering**: Explore techniques to guide model behavior, including:
  - Role-based instructions (e.g., setting tone or context with the `developer` role)
  - Few-shot prompting using the `assistant` role to provide examples
  - Chain-of-thought prompting for step-by-step reasoning
- **Reasoning Example**: See how the model handles counting tasks and how prompt clarity affects accuracy.

## Activities

- Experiment with prompt engineering techniques to improve model responses.
- Update prompts to encourage correct reasoning and step-by-step answers.

## How to Use

1. Install required dependencies (see `pyproject.toml` and `uv.lock`).
2. Enter your OpenAI API key when prompted.
3. Run the notebook cells to interact with GPT-4.1-nano and explore prompt engineering.

## Credits

Materials adapted for PSI AI Academy. Original materials from AI Makerspace.
