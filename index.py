import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import difflib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient


tracked_file_types = ['.txt', '.md', '.py', '.java', '.js', '.html', '.css']
workspace_dir = os.path.basename(os.getcwd())

class Index:
    def __init__(self, embed_model_name='BAAI/bge-small-en-v1.5', query_model_name='CohereLabs/command-a-reasoning-08-2025'):
        self.structure = {}
        self.embed_model = SentenceTransformer(embed_model_name)
        # check if open-ai-key storage exists
        if os.path.exists('.hf_key'):
            hf_key = open('.hf_key', 'r').read().strip()
        else:
            hf_key = input('Enter your HF API key: ').strip()
            with open('.hf_key', 'w') as f:
                f.write(hf_key)
        os.environ['HF_TOKEN'] = hf_key
        self.query_model = InferenceClient(
            provider="cohere",
            api_key=os.environ["HF_TOKEN"],
        )
        self.query_model_name = query_model_name

    def build_user_index(self):
        print('Building user index...')
        self.structure[workspace_dir] = {}
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.endswith(ext) for ext in tracked_file_types):
                    file_path = os.path.relpath(os.path.join(root, file), '.')
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()
                        lines = list(filter(lambda line: line.strip(), lines))
                        print(f'Indexing {file_path} with {len(lines)} lines')
                        self.structure[workspace_dir][file_path] = self.generate_indices(lines)

    def update_file_index(self, file_path):
        print(f'Updating index for {file_path}')
        lines = open(file_path, 'r').read().splitlines()
        lines = list(filter(lambda line: line.strip(), lines)) # ignore empty lines
        rel_file_path = os.path.relpath(file_path, '.')
        file_structure = self.structure[workspace_dir][rel_file_path]
        o_lines = [atom['text'] for atom in file_structure]
        matcher = difflib.SequenceMatcher(None, o_lines, lines)
        new_file_structure = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                new_file_structure.extend(self.generate_indices(lines[j1:j2]))
            elif tag == 'delete':
                pass
            elif tag == 'insert':
                new_file_structure.extend(self.generate_indices(lines[j1:j2]))
            elif tag == 'equal':
                new_file_structure.extend(file_structure[i1:i2])
        self.structure[workspace_dir][rel_file_path] = new_file_structure

    def generate_indices(self, texts):
        text_embeddings = self.embed_model.encode(texts)
        queries = []
        for text in texts:
            query = self.query_model.chat.completions.create( model=self.query_model_name, messages=[ {"role": "user", "content": '''You are an AI agent in a RAG system where your job is to generate questions based on a paragraph of text with the idea being that when a user searches with a query, we will do RAG between the user's query and the question you generated so we can surface the relevant paragraph to the user. Now, generate a concise question that the following paragraph could answer. Make the question specific to the content of the paragraph.\n\nParagraph: {text}\n\nQuestion:'''} ], )
            queries.append(query.choices[0].message)
        query_embeddings = self.embed_model.encode(queries)
        indices = [
            {
                'text': texts[i],
                'text_embedding': text_embeddings[i],
                'query': queries[i],
                'query_embedding': query_embeddings[i]
            } for i in range(len(texts))
        ]
        return indices





