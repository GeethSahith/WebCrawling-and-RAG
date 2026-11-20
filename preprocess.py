# Cleaned and chunked credit card data for RAG implementation
import json
import os
import re

def clean_text(text):
    text = re.sub(r'(Cash Back|Points Rewards|Travel Rewards).+?Credit Cards,?\s*', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

# Merged card fields
def combine_card_fields(card):
    parts = []
    card_name = clean_text(card.get('card_name', ''))
    if card_name:
        parts.append(f"Card: {card_name}")
    if card.get('category'):
        parts.append(f"Category: {clean_text(card['category'])}")
    if card.get('features'):
        parts.append(f"Features: {clean_text(card['features'])}")
    if card.get('benefits'):
        parts.append(f"Benefits: {clean_text(card['benefits'])}")
    if card.get('rates_fees'):
        parts.append(f"Rates: {clean_text(card['rates_fees'])}")
    return ' '.join(parts)
# splitted the text into 500 word chunks
def chunk_text(text, card_name, category, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            "text": ' '.join(chunk_words),
            "metadata": {"card_name": card_name, "category": category, "source": "bankofamerica_credit_cards"}
        })
    return chunks if chunks else [{"text": text, "metadata": {"card_name": card_name, "category": category, "source": "bankofamerica_credit_cards"}}]

def preprocess_credit_cards(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_cards = json.load(f)
    
    all_chunks = []
    for card in raw_cards:
        card_name = clean_text(card.get('card_name', ''))
        category = clean_text(card.get('category', ''))
        
        if not card_name:
            continue
        
        combined_text = combine_card_fields(card)
        if len(combined_text) < 50:
            continue
        
        card_chunks = chunk_text(combined_text, card_name, category)
        all_chunks.extend(card_chunks)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Preprocessed {len(all_chunks)} chunks")
# Saved in the cleaned json data/chunks 
preprocess_credit_cards("data/raw/credit_cards.json", "data/chunks/card_chunks.json")
