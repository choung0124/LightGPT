from openai import OpenAI


client = OpenAI()

prompt_template_2 = """Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here
Question: {question}
Context:
{context} 
Previous Chats:
{previous_chats_str}"""

def inference(question, context, previous_chats_str):
    prompt = prompt_template_2.format(question=question, context=context, previous_chats_str=previous_chats_str)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an domain expert in everything related to light and its relations to neurological disorders such as Alzheimer's and Parkinson's"},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    return chunk.choices[0].delta.content
            


answer = inference("What is the relationship between light and Alzheimer's?", "Light has been shown to have a relationship with Alzheimer's", "Previous Chats: [User: What is the relationship between light and Alzheimer's? Context: Light has been shown to have a relationship with Alzheimer's]")  