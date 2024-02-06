import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings


class SFRMistralEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral', add_special_tokens=True)
        self.model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', torch_dtype=torch.float16).cuda()
        self.max_length = 4096

    def __call__(self, input: Documents) -> Embeddings:
        # Prepare the input documents with the required task instruction
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        input_with_instructions = [self.get_detailed_instruct(task, doc) for doc in input]

        # Tokenize the input documents
        inputs = self.tokenizer(input_with_instructions, max_length=self.max_length - 1, padding=True, truncation=True, return_tensors="pt")

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract and normalize embeddings
        embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Debugging: print the shape of embeddings
        print("Embeddings shape:", normalized_embeddings.shape)

        # Convert to list of lists if necessary
        embeddings_list = normalized_embeddings.cpu().numpy().tolist()

        # Debugging: print first few embeddings
        print("First few embeddings:", embeddings_list[:2])
        
        return embeddings_list
    
    def last_token_pool(self, last_hidden_states, attention_mask):
        # Implement the pooling mechanism as per the model creator's instructions
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        # Implement the instruction addition for each document
        return f'Instruct: {task_description}\nQuery: {query}'