# Bank of America Credit Cards RAG System

## Project Structure

crawler.py
preprocess.py
index.py
rag_api.py
app.py
requirements.txt

## Setup
### 1. Install Dependencies

```bash
python -m venv myenv
.\myenv\Scripts\activate.ps1

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create .env file in project root with:
OPENAI_API_KEY=openai_api_key

## System Workflow

### Step 1: Run the Crawler to crawl the website
python crawler.py

Saves to: data/raw/credit_cards1.json
### Step 2: Preprocess Data
python preprocess.py

Saves to: data/chunks/card_chunks.json
### Step 3: Build Vector Store
python index.py

vector_store/index.faiss - FAISS vector index
vector_store/metadata.pkl - Card metadata (pickled list)

### Step 5: Run Streamlit
streamlit run app.py


## Example Queries
What does Air France KLM Visa Signature card offer?
What are the features of the Atmos Rewards Summit card?