import json
from dataclasses import dataclass
from typing import List

LETTER_OPTIONS = ["A", "B", "C", "D"]

@dataclass
class MCQAItem:
    question: str
    options: List[str]
    answer: str
    trait: str
    answer_full_writing: str = ""  # For free-form evaluation

def load_mcqa(path: str) -> List[MCQAItem]:
    """Load MCQA items from JSON file."""
    data = []
    with open(path, "r") as f:
        raw = json.load(f)
    for ex in raw:
        data.append(MCQAItem(
            question = ex["question"],
            options = ex["options"],
            answer = ex["answer"],
            trait = ex["trait"].lower().strip(),
            answer_full_writing = ex.get("answer_full_writing", ""),
        ))
    return data

def extract_choice(text: str) -> str:
    """
    Extract answer choice from model response.
    Looks for patterns like "A)", "B)", "C)", "D)" first,
    then falls back to just the letter.
    Returns the choice with parenthesis (e.g., "A)") or None.
    """
    # First, look for letter with parenthesis
    for c in ["A)", "B)", "C)", "D)"]:
        if c in text:
            return c
    # Fallback: look for just the letter and add parenthesis
    for c in ["A", "B", "C", "D"]:
        if c in text:
            return f"{c})"
    return None

def build_prompt_chat(tokenizer, question: str, options: List[str]) -> str:
    """
    Build a chat-formatted prompt using the tokenizer's chat template.
    This matches the approach from part 1 for MCQA.
    """
    content = question.strip() + "\n" + "\n".join(options) + "\nAnswer:"
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that answers multiple choice questions. Your answer should be in the format of A), B), C), D). Do not include any other text in your answer."
        },
        {
            "role": "user", 
            "content": content
        }
    ]
    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return chat_str

def build_prompt_freeform(tokenizer, question: str) -> str:
    """
    Build a free-form chat prompt (question only, no options).
    This matches the approach from part 1 for free-form generation.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions directly and concisely."
        },
        {
            "role": "user",
            "content": question.strip()
        }
    ]
    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return chat_str

def check_free_form_answer(response: str, correct_answer: str) -> bool:
    """
    Check if the model's free-form response matches the correct answer.
    Uses lowercase matching for better accuracy.
    """
    response_lower = response.strip().lower()
    correct_lower = correct_answer.strip().lower()
    return correct_lower in response_lower
