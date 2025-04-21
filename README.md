# Python Documentation QA Dataset Generator

This project automates the generation of SQuAD-style question-answer (QA) datasets from Python standard library documentation written in reStructuredText (`.rst`) format. It was developed as part of a graduation thesis to support the fine-tuning of large language models (LLMs) for question answering over programming and API documentation.

---

## 🚀 Features

- Parses `.rst` files containing structured Python documentation
- Extracts `.. function::`, `.. method::`, and `.. class::` blocks
- Automatically generates 3–4 natural language QA pairs per item
- Formats output into [SQuAD v1.0](https://rajpurkar.github.io/SQuAD-explorer/) JSON format
- Generates markdown logs with per-module QA statistics and coverage

---

## 📂 Directory Structure

```
project-root/
├── enhanced_rst_to_squad.py         # Main script
├── python_rst_docs/                 # Folder for .rst documentation files
├── qa_output/                       # Output folder for JSON files and logs
```

---

## 📦 Requirements
- Python 3.7+
- No external dependencies (only standard libraries: `os`, `re`, `json`, `random`, `pathlib`)

Optional for sentence refinement:
- `nltk` for sentence/token filtering (planned in future version)

---

## 🛠️ How to Use

### 1. Prepare Your `.rst` Files
Place `.rst` documentation files into the `python_rst_docs/` folder. For example:
- `math.rst`
- `io.rst`
- `string.rst`

### 2. Run the Generator
```bash
python enhanced_rst_to_squad.py
```

### 3. Output Files
- JSON files are saved in `qa_output/`
- Each module produces:
  - `module.train.json`
  - `module.val.json`
  - `logs/module_report.md`

---

## ✅ Supported Directives
- `.. function::` — standalone functions
- `.. method::` — class methods
- `.. class::` — class-level blocks

Support for `.. data::`, `.. attribute::`, and `.. exception::` is planned in a future update.

---
```
