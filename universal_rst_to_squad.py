import os
import json
import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from collections import defaultdict

NLTK_DATA_DIR = "/Users/nurbekkhujaev/GradPro/python-docs-qa-generator/nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk_resources():
    resources = [
        ("stopwords", "corpora/stopwords"),
        ("maxent_ne_chunker", "chunkers/maxent_ne_chunker"),
        ("words", "corpora/words"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
    ]
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, download_dir=NLTK_DATA_DIR)

ensure_nltk_resources()

QUESTION_TEMPLATES = [
    "What is mentioned about {}?",
    "Can you explain {}?",
    "How does {} work?",
    "What are the key features of {}?",
    "Why is {} important?",
    "What is the role of {} in Python?",
    "How is {} used in programming?",
    "What challenges are associated with {}?",
    "What are the benefits of {}?",
    "What makes {} unique?",
]

BAD_KEY_TERMS = {
    "the", "this", "that", "these", "those", "also", "such", "when", "while", "then",
    "except", "formerly", "note", "noted", "import", "return", "added", "using", "used", "use", "will", "shall"
}

def clean_rst_text(raw_text):
    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(("..", "::", ">>>", "===")):
            continue
        if not line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def custom_sentence_tokenizer(text):
    abbrevs = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 'etc.', 'i.e.', 'e.g.']
    for abbrev in abbrevs:
        text = text.replace(abbrev, abbrev.replace('.', '@@@'))
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('@@@', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]

def extract_sentences(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read().strip()
    cleaned_text = clean_rst_text(raw_text)
    return custom_sentence_tokenizer(cleaned_text)

def extract_key_terms(text):
    try:
        words = word_tokenize(text)
        words = [w for w in words if w.isalnum()]
        try:
            stop_words = set(stopwords.words("english"))
            filtered = [w for w in words if w.lower() not in stop_words]
        except:
            filtered = [w for w in words if w.lower() not in BAD_KEY_TERMS]

        chunked = ne_chunk(nltk.pos_tag(filtered))
        terms = []
        for subtree in chunked:
            if isinstance(subtree, Tree):
                term = " ".join([token for token, _ in subtree.leaves()])
                terms.append(term)
        if not terms:
            terms = [word for word, pos in nltk.pos_tag(filtered) if pos.startswith("NN")]
        return terms or [filtered[0]] if filtered else ['Python']
    except Exception:
        words = text.split()
        return [words[0]] if words else ['Python']

def generate_qa_pairs(sentences, max_pairs=100, variations_per_sentence=3):
    context_map = defaultdict(list)
    qa_id = 1

    for i, sentence in enumerate(sentences):
        if qa_id > max_pairs:
            break

        key_terms = extract_key_terms(sentence)
        key_terms = [kt for kt in key_terms if kt.lower() not in BAD_KEY_TERMS and len(kt) > 2]
        if not key_terms:
            continue

        key_term = random.choice(key_terms)
        question_templates = random.sample(QUESTION_TEMPLATES, min(variations_per_sentence, len(QUESTION_TEMPLATES)))

        # Build context window
        start = max(0, i - 2)
        end = min(len(sentences), i + 3)
        context = " ".join(sentences[start:end])

        answer_start = context.find(sentence)
        if answer_start == -1:
            continue

        for template in question_templates:
            if qa_id > max_pairs:
                break

            question = template.format(key_term)
            context_map[context].append({
                "id": f"qa_{qa_id}",
                "question": question,
                "answers": [{"text": sentence, "answer_start": answer_start}],
                "is_impossible": False
            })
            qa_id += 1

    return context_map

def format_squad_json(context_map, title):
    return {
        "version": "1.0",
        "data": [{
            "title": title,
            "paragraphs": [{"context": ctx, "qas": qas} for ctx, qas in context_map.items()]
        }]
    }

def save_to_json(data, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved: {path}")
    except Exception as e:
        print(f"❌ Error saving {path}: {e}")

def convert_rst_file(rst_path, output_dir, max_pairs=100):
    try:
        title = os.path.splitext(os.path.basename(rst_path))[0]
        sentences = extract_sentences(rst_path)
        context_map = generate_qa_pairs(sentences, max_pairs=max_pairs, variations_per_sentence=4)
        if not context_map:
            print(f"⚠️ No valid Q&A pairs generated for {title}")
            return []

        squad_data = format_squad_json(context_map, title)
        output_path = os.path.join(output_dir, f"{title}.json")
        save_to_json(squad_data, output_path)
        return squad_data["data"]
    except Exception as e:
        print(f"❌ Failed for {rst_path}: {e}")
        return []

def process_all_rst_files(input_dir, output_dir, merge_output=False, max_pairs=100):
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".rst"):
            full_path = os.path.join(input_dir, filename)
            print(f"📘 Processing: {filename}")
            module_data = convert_rst_file(full_path, output_dir, max_pairs=max_pairs)
            all_data.extend(module_data)

    if merge_output:
        merged_output = {
            "version": "1.0",
            "data": all_data
        }
        save_to_json(merged_output, os.path.join(output_dir, "python_stdlib_merged.json"))

if __name__ == "__main__":
    INPUT_RST_FOLDER = "/Users/nurbekkhujaev/GradPro/python-docs-qa-generator/python_rst_docs"
    OUTPUT_JSON_FOLDER = "/Users/nurbekkhujaev/GradPro/python-docs-qa-generator/qa_output"
    os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)
    process_all_rst_files(INPUT_RST_FOLDER, OUTPUT_JSON_FOLDER, merge_output=True, max_pairs=100)
