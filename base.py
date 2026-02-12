# app.py
import streamlit as st
import joblib
import numpy as np
from rapidfuzz import process, fuzz, distance
from wordfreq import word_frequency, top_n_list
from rapidfuzz.distance import Levenshtein
from transformers import T5ForConditionalGeneration, T5Tokenizer
from spellchecker import SpellChecker
import torch

# -----------------------------
# Load Models and Data
# -----------------------------
spell_model = joblib.load("spell_model_xgb.pkl")
vocab = joblib.load("vocab.pkl")
vectorizer = joblib.load("tfidf.pkl")
spell = SpellChecker()

common_vocab = set(top_n_list("en", 5000))
vocab = list(set(vocab) & common_vocab)

model_name = "vennify/t5-base-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(model_name)
grammar_model = T5ForConditionalGeneration.from_pretrained(model_name)

# -----------------------------
# Helper Functions
# -----------------------------
def features(wrong, candidate, vectorizer):
    prefix = len([1 for a, b in zip(wrong, candidate) if a == b])
    suffix = len([1 for a, b in zip(wrong[::-1], candidate[::-1]) if a == b])

    v1 = vectorizer.transform([wrong])
    v2 = vectorizer.transform([candidate])
    tfidf_sim = float((v1 @ v2.T).toarray()[0][0])

    set1, set2 = set(wrong), set(candidate)
    jaccard_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    return [
        fuzz.ratio(wrong, candidate) / 100,
        distance.Levenshtein.distance(wrong, candidate),
        abs(len(wrong) - len(candidate)),
        prefix,
        suffix,
        word_frequency(candidate, "en"),
        tfidf_sim,
        jaccard_sim
    ]


def autocorrect_word(word):
    if word not in spell.unknown([word]):
        return word
    
    candidates = [c for c, _, _ in process.extract(word, vocab, limit=10)]
    candidates = [c for c in candidates if distance.Levenshtein.distance(word, c) <= 3]
    if not candidates:
        return word

    X_test = [features(word, c, vectorizer) for c in candidates]
    probs = spell_model.predict_proba(np.array(X_test))[:, 1]
    best = candidates[probs.argmax()]

    if probs.max() < 0.3:
        return word
    if Levenshtein.distance(word, best) > 3:
        return word

    return best


def autocorrect_sentence_spelling(sentence):
    return " ".join(autocorrect_word(w) for w in sentence.split())


def correct_grammar(text: str) -> str:
    input_text = "gec: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = grammar_model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def correct_pipeline(sentence: str):
    spelling_fixed = autocorrect_sentence_spelling(sentence)
    final_corrected = correct_grammar(spelling_fixed)
    return final_corrected

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Machine Learning Based Autocorrect System")
st.write("Enter a sentence below to correct .")

user_input = st.text_area("Your Sentence", "")

if st.button("Correct"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence!")
    else:
        with st.spinner("Processing..."):
            corrected = correct_pipeline(user_input)
        st.success("Correction Complete!")
        st.subheader("Corrected Sentence:")
        st.write(corrected)

st.write("---")
