# Crawled the Bank of America Website using BeutifulSoup
import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import html

BASE_URL = "https://www.bankofamerica.com"
CREDIT_CARDS_URL = f"{BASE_URL}/credit-cards/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Downloaded the HTML content from the URL
def fetch_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text
    except:
        return None

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_card_urls(soup):
    urls = set()
    card_list_div = soup.select_one('.card-list[data-jcr]')
    if card_list_div:
        jcr_data = html.unescape(card_list_div.get('data-jcr', ''))
        if jcr_data:
            try:
                card_configs = json.loads(jcr_data)
                for key, config in card_configs.items():
                    if isinstance(config, dict):
                        learn_more = config.get('learnMore', {})
                        if isinstance(learn_more, dict):
                            path = learn_more.get('path', '')
                            if path:
                                urls.add(f"{BASE_URL}/credit-cards/{path}")
            except:
                pass
    return list(urls)
# Scraped individual card page
def scrape_card_page(url):
    html_content = fetch_page(url)
    if not html_content:
        return None
    
    soup = BeautifulSoup(html_content, 'html.parser')
    card_data = {"card_name": "", "features": [], "benefits": []}
    
    title_elem = soup.select_one('h1, .product-name')
    if title_elem:
        card_data["card_name"] = clean_text(title_elem.get_text())
    
    if not card_data["card_name"]:
        meta_title = soup.select_one('meta[property="og:title"]')
        if meta_title:
            card_data["card_name"] = clean_text(meta_title.get('content', ''))
    
    features = []
    benefits = []
    
    all_paragraphs = soup.select('p, li')
    for p in all_paragraphs:
        text = clean_text(p.get_text())
        if not text or len(text) < 20 or len(text) > 300:
            continue
        
        text_lower = text.lower()
        skip_keywords = ['compare credit', 'apply now', 'learn more', 'cookie', 'privacy', 'log in']
        if any(kw in text_lower for kw in skip_keywords):
            continue
        
        if any(kw in text_lower for kw in ['cash back', 'points', 'apr', 'fee', 'earn', 'bonus', 'rewards']):
            if text not in features:
                features.append(text)
        
        if any(kw in text_lower for kw in ['benefit', 'protection', 'insurance', 'preferred rewards']):
            if text not in benefits:
                benefits.append(text)
    
    card_data["features"] = features[:15]
    card_data["benefits"] = benefits[:10]
    
    return card_data

# Got the category from /credit_cards 
def get_category(card_name):
    name_lower = card_name.lower()
    if "cash" in name_lower and "back" in name_lower:
        return "Cash Back"
    elif "travel" in name_lower:
        return "Travel Rewards"
    elif "airline" in name_lower or "alaska" in name_lower:
        return "Airline Rewards"
    elif "student" in name_lower:
        return "Cards for Students"
    elif "secured" in name_lower:
        return "Build Credit"
    return "Featured"


def scrape_all_cards():
    print("Fetching main page...")
    html_content = fetch_page(CREDIT_CARDS_URL)
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    product_urls = extract_card_urls(soup)
    print(f"Found {len(product_urls)} cards to scrape")
    
    cards = []
    seen_names = set()
    
    for i, url in enumerate(product_urls):
        print(f"[{i+1}/{len(product_urls)}] Scraping card...")
        card_data = scrape_card_page(url)
        
        if card_data and card_data.get("card_name"):
            name = card_data["card_name"]
            if name in seen_names:
                continue
            
            seen_names.add(name)
            card_data["category"] = get_category(name)
            card_data["rates_fees"] = ""
            cards.append(card_data)
            print(f"  {name}")
        
        time.sleep(0.5)
    
    return cards

# Ran the Scraper and formatted the data
def main():
    print("Bank of America Credit Cards Scraper")
    os.makedirs("data/raw", exist_ok=True)
    
    cards = scrape_all_cards()
    
    output_cards = []
    for card in cards:
        output_cards.append({
            "card_name": card.get("card_name", ""),
            "category": card.get("category", "Featured"),
            "features": ", ".join(card.get("features", [])),
            "benefits": ", ".join(card.get("benefits", [])),
            "rates_fees": card.get("rates_fees", ""),
        })
    
    output_path = "data/raw/credit_cards1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_cards, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal cards: {len(output_cards)}")

main()
