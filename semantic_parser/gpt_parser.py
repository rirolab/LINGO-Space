import os
import timeout_decorator
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = lambda instruction: f"""
Imagine you are a semantic parser.
I will give you an instruction that describes a space.
Given the instruction, first extract an action from the instruction.
Then, extract the source and the target of the action.
Note that a space is composed of reference objects and relation predicates.

Instruction: put the green blocks in a blue bowl
Output: {{'action': 'move', 'source': 'green blocks', 'target': [('blue bowl', 'in')]}}

Instruction: move to the front of the table
Output: {{'action': 'move', 'source': 'robot', 'target': [('table', 'front')]}}

Instruction: put the Pepsi Wild cherry box to the left of the gold box and put the Pepsi Wild cherry box to the right of the bull figure and put the Pepsi Wild cherry box below the gold box and put the Pepsi Wild cherry box below the gold box
Output: {{'action': 'move', 'source': 'Pepsi Wild cherry box', 'target': [('gold box', 'left'), ('bull figure', 'right'), ('gold box', 'below'), ('gold box', 'below')]}}

Instruction: {instruction}
Output:
"""

@timeout_decorator.timeout(240)
def send_query(query, model="gpt-4", temparature=0.2): # model="gpt-4-0314"
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": query},],
        temperature=temparature)
    return chat_completion['choices'][0]['message']['content']

def parse(instruction):
    query = prompt(instruction)
    output = send_query(query)
    output = eval(output)
    return output
