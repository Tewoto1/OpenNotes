import faiss, numpy as np, textwrap, os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# I assume for now that we will later import a database structure
# from a separate file, like a dictionary indexing the notes and content
notes = {
    'name1': [
        "The following notes outline preliminary research into the potential for fungal life forms to impact the composition and structure of Martian regolith.",
        "This research is purely speculative and based on known terrestrial fungal properties and the chemical composition of Martian soil as determined by various rover missions."
    ],
    'name2': [
        "Fungi are highly adaptable organisms known for surviving in extreme environments on Earth, including radiation-heavy and nutrient-poor conditions.",
        "Martian regolith is rich in iron oxides and perchlorates, which are toxic to most known life forms.",
        "The extremely low temperatures and thin atmosphere of Mars present a significant challenge. However, some fungi can enter dormant states or produce cryoprotective compounds.",
        "The lack of liquid water on the surface is a major hurdle."
    ],
    'name3': [
        "**Bioweathering:** Fungi secrete organic acids to break down rocks and absorb nutrients.",
        "**Soil Aggregation:** Mycelial networks could bind regolith particles together, increasing soil porosity and potentially stabilizing slopes.",
        "**Resource Extraction:** Fungi could be engineered to metabolize specific Martian minerals, such as extracting iron from hematite.",
        "**Perchlorate Reduction:** Perchlorates are a major contaminant in Martian soil."
    ],
    'name4': [
        "What specific fungal species, or genetic modifications thereof, would be best suited for this environment?",
        "What is the long-term impact of fungal bioweathering on the structural integrity of the Martian surface?",
        "How would fungal growth be sustained in a low-nutrient, low-water environment?",
        "Could the introduction of terrestrial fungi to Mars have unforeseen and negative ecological consequences?"
    ]
}
# Notes eventually of form where each value of a dictionary is a list of strings
# of the paragraphs in the notes

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

notes_embeddings = {}
for name, paragraphs in notes.items():
    embeddings = embed_model.encode(paragraphs)
    notes_embeddings[name] = embeddings

# indexxing the notes embeddings
name_to_index = {}
index_to_name = {}
all_embeddings = []
counter = 0
for name, embeddings in notes_embeddings.items():
    name_to_index[name] = (counter, counter + len(embeddings))
    for emb in embeddings:
        all_embeddings.append(emb)
        index_to_name[counter] = name
        counter += 1

random_name = index_to_name[0]
dimension = notes_embeddings[random_name][0].shape[0]
index = faiss.IndexFlatIP(dimension)
n = len(all_embeddings)
index.add(n=2, x=np.array(all_embeddings))

def retrieve_notes_from_semantics(query, k=3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k) # type: ignore
    results = []
    for idx in I[0]:
        name = index_to_name[idx]
        para_idx = idx - name_to_index[name][0]
        paragraph = notes[name][para_idx]
        results.append((name, paragraph))
    return results

# So this part is up to just embedding the notes themselves
# Next step we have to generate the questions for each paragraph and then embed those
# so that when provided a user query we can find the most relevant question and then return
# the notes corresponding to that question

QUERY_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
gen = AutoModelForCausalLM.from_pretrained(
    QUERY_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

def generate_questions(paragraph, num_questions=3):
    prompt = f'''Generate a concise question that the following 
                paragraph could answer. Make the question specific to the content 
                of the paragraph.\n\nParagraph: {paragraph}\n\nQuestions:'''
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    outputs = gen.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.7,
        temperature=0.95,
        num_return_sequences=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    decoded_output = tok.batch_decode(outputs, skip_special_tokens=True)[0]
    question = decoded_output.split("Questions:")[-1].strip()
    return question[:num_questions]
# Testing the question generation
# def test_generate_questions():
#     for para in notes['name1']:
#         print(generate_questions(para))

# Now we generate questions for each paragraph in the notes,
# and store them in a new dictionary mapping questions to (note_name, paragraph_index)
questions_dict = {}
for name, paragraphs in notes.items():
    for idx, para in enumerate(paragraphs):
        question = generate_questions(para)
        questions_dict[question] = (name, idx)

# Now we embed each of the generated questions and index them
questions = list(questions_dict.keys())
questions_embeddings = embed_model.encode(questions)
questions_index = faiss.IndexFlatIP(dimension)
questions_index.add(n=len(questions_embeddings), x=np.array(questions_embeddings))

def retrieve_notes_from_query(query, k=3):
    query_embedding = embed_model.encode([query])
    D, I = questions_index.search(query_embedding, k) # type: ignore
    results = []
    for idx in I[0]:
        question = questions[idx]
        name, para_idx = questions_dict[question]
        paragraph = notes[name][para_idx]
        results.append((question, name, paragraph))
    return results
# Testing the full retrieval system
# def test_retrieve_notes_from_query():
#     query = "How can fungi help with soil on Mars?"
#     results = retrieve_notes_from_query(query, k=3)
#     for question, name, para in results:
#         print(f"Question: {question}\nFrom Note: {name}\nParagraph: {para}\n")