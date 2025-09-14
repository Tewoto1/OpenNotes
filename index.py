import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import difflib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient


tracked_file_types = ['.txt', '.md', '.py', '.java', '.js', '.html', '.css']
workspace_dir = os.getcwd()

class Index:
    def __init__(self, embed_model_name='BAAI/bge-small-en-v1.5', query_model_name='openai/gpt-oss-120b'):
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
            provider="cerebras",
            api_key=os.environ["HF_TOKEN"],
        )
        self.query_model_name = query_model_name
        self.embed_model_dim = self.embed_model.get_sentence_embedding_dimension()
        self.vectors = faiss.IndexFlatIP(self.embed_model_dim)
        self.meta_data = []

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
                        self.structure[workspace_dir][file_path] = self.add_indices(file_path, lines)

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
                new_file_structure.extend(self.add_indices(file_path, lines[j1:j2], line_num_offset=j1))
            elif tag == 'delete':
                self.vectors.remove_ids(np.array([atom['vec_index'] for atom in file_structure[i1:i2]]))
            elif tag == 'insert':
                new_file_structure.extend(self.add_indices(file_path, lines[j1:j2], line_num_offset=j1))
            elif tag == 'equal':
                new_file_structure.extend(file_structure[i1:i2])
                for i in range(i1, i2):
                    self.meta_data[file_structure[i]['vec_index']]['line_num'] = j1 + (i - i1)
                    self.meta_data[file_structure[i]['vec_index']+1]['line_num'] = j1 + (i - i1)
        self.structure[workspace_dir][rel_file_path] = new_file_structure

    def add_indices(self, file_path, texts, line_num_offset=0):
        text_embeddings = self.embed_model.encode(texts)
        queries = []
        for text in texts:
            query = self.query_model.chat.completions.create( model=self.query_model_name, messages=[ {"role": "user", "content": f'''You are an AI agent in a RAG system where your job is to generate questions based on a paragraph of text with the idea being that when a user searches with a query, we will do RAG between the user's query and the question you generated so we can surface the relevant paragraph to the user. Now, generate a concise question that the following paragraph could answer. Reply with just the question and nothing else. Make the question specific to the content of the paragraph.\n\nParagraph: {text}\n\nQuestion:'''} ], )
            print(query.choices[0].message.content.strip())
            queries.append(query.choices[0].message.content.strip())
        query_embeddings = self.embed_model.encode(queries)
        indices = []
        for i in range(len(texts)):
            indices.append(
                {
                    'text': texts[i],
                    'text_embedding': text_embeddings[i],
                    'query': queries[i],
                    'query_embedding': query_embeddings[i],
                    'vec_index': len(self.meta_data)
                }
            )
            self.vectors.add(np.array([text_embeddings[i]]))
            self.meta_data.append({
                'file_path': file_path,
                'type': 'text',
                'line_num': i + line_num_offset,
            })
            self.vectors.add(np.array([query_embeddings[i]]))
            self.meta_data.append({
                'file_path': file_path,
                'type': 'query',
                'line_num': i + line_num_offset,
            })
        return indices

    def query(self, msg):
        q_embed = self.embed_model.encode([msg])
        D, I = self.vectors.search(q_embed, 1) # type: ignore
        idx = I[0][0]
        print(self.meta_data[idx]['type'])
        lines = open(os.path.join(workspace_dir, self.meta_data[idx]['file_path']), 'r').read().splitlines()
        lines = list(filter(lambda line: line.strip(), lines)) # ignore empty lines
        print(lines[self.meta_data[idx]['line_num']])






