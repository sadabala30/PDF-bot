# PDF Bot - Learning Journal

## Phase 1 - LLM API Fundamentals

### What is an API key?
- Proves you're allowed to access Claude programmatically
- Never share it, never push to GitHub

---

### What is API calling?
- Manually: You type in Claude website → Claude responds
- Via API: Your Python code sends message → Claude responds → code uses it
- Your code can now think using Claude's brain automatically

---

### Project Setup Commands
```bash
mkdir pdf-bot          # create project folder
cd pdf-bot             # go into folder
python -m venv venv    # create isolated Python environment for this project
venv\Scripts\activate  # activate it — you'll see (venv) appear
pip install anthropic  # install Claude's Python library
code app.py            # create and open Python file
```

---

### Why virtual environment?
- Keeps this project's libraries separate from everything else on your PC
- Always activate before working: `venv\Scripts\activate`

---

### First API Call — app.py
```python
import anthropic                        # load the library

client = anthropic.Anthropic(api_key="your-key")  # connect using your key

message = client.messages.create(       # send message to Claude
    model="claude-haiku-4-5-20251001",  # cheapest/fastest model
    max_tokens=1024,                    # max response length
    messages=[
        {"role": "user", "content": "your question here"}
    ]
)

print(message.content[0].text)        

---

## Conversation Memory

### Problem
Claude has no memory by default.
Every API call is completely fresh — it forgets everything.

### Why?
Our code was only sending the current message each time:
```python
messages=[{"role": "user", "content": user_input}]
```

### Fix — conversation_history list
Keep a list that grows with every message sent and received.
Send the entire list to Claude every time.

```python
conversation_history = []  # stores full conversation

# every loop:
conversation_history.append({"role": "user", "content": user_input})
# after claude replies:
conversation_history.append({"role": "assistant", "content": reply})

# send full history to claude:
messages=conversation_history
```

### What Claude actually receives by message 2:
```python
[
    {"role": "user", "content": "my name is sai"},
    {"role": "assistant", "content": "Nice to meet you Sai!"},
    {"role": "user", "content": "whats my name"}
]
```

### Key takeaway
Claude has no memory by itself.
You send the full history manually every time.
This is how every chatbot in the world works.

System prompts — what are they?
Right now Claude has no personality, no role, no rules. It just answers anything generically. eg: system="You are a helpful assistant who only answers questions about machine learning...",
A system prompt lets you tell Claude who it is before the conversation starts.

### What is it?
A secret instruction given to Claude before the conversation starts.
User never sees it. Claude treats it as its rulebook.

### Where it sits in the API call:
```python
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system="Your instructions here",  # <-- system prompt
    messages=conversation_history
)
```
### Why it matters for PDF bot:
We'll use it to tell Claude:
"Only answer from the document. If answer isn't there, say you don't know."
This stops Claude from making things up.

import fitz              # pymupdf library

doc = fitz.open("ml.pdf")  # opens the PDF
len(doc)                   # counts pages → 615

page = doc[page_num]       # grabs a specific page
text = page.get_text()     # extracts all text from that page
That's it. PDF → plain text in 4 lines.

615 pages is way too much to send to Claude all at once. Remember tokens? Sending 615 pages would cost a fortune and exceed Claude's limit.
The solution is chunking — split the text into small pieces and only send the relevant piece to Claude.
That's exactly what we build next.

chunking:
Think of it like this:
Someone asks "What is gradient descent?"

You don't send all 615 pages to Claude
You find the 3-4 chunks that talk about gradient descent
You send only those to Claude

Cheaper. Faster. More accurate.

full_text = ""
for page in doc:
    full_text += page.get_text()
Loops through all 615 pages and joins all text into one giant string.

pythonfor i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)
This is the core. Let's say chunk_size=500:

i=0 → words 0 to 499 → chunk 1
i=500 → words 500 to 999 → chunk 2
i=1000 → words 1000 to 1499 → chunk 3
...and so on until all words are chunked ie buckets

range(0, total_words, 500) = jump in steps of 500.
ie: 615 pages → one giant text → split into 523 chunks of ~500 words each
Each chunk is small enough to send to Claude

One problem with this approach though:
What if a concept like "gradient descent" starts at word 498 and ends at word 650?
bucket 1 → "...gradient des"   ← cut off!
bucket 2 → "cent is a method" ← missing context!
The chunk got split right in the middle of the explanation.

The fix is called overlap — each chunk shares some words with the next one:
bucket 1 → words 0-499
bucket 2 → words 400-899   ← starts 100 words back
bucket 3 → words 800-1299  ← starts 100 words back
That way no concept gets cut off at the edges.

now, We just need to find the RIGHT chunk for each question ie embeddings
Option 1 — Simple word matching
Option 2 — Embeddings:
Convert every chunk into a list of numbers that captures its meaning.
"gradient descent"              → [0.2, 0.8, 0.1, 0.9, ...]
"optimization that minimizes"   → [0.21, 0.79, 0.11, 0.88, ...]  ← similar numbers!
"recipe for chocolate cake"     → [0.9, 0.1, 0.8, 0.2, ...]      ← very different numbers
Similar meaning = similar numbers.
ie cosine similarity
cosine similarity = (A · B) / (||A|| × ||B||)
A vector database does it for you across all 523 chunks instantly. ie chromadb
chooses eg Top 3 means the 3 chunks with the highest cosine similarity score to the question.

So the pipeline in one line:
text → embedding model (eigenvectors involved here) → vector → cosine similarity search → top chunks
Eigenvectors = how meaningful numbers are created.
Cosine similarity = how we find the closest ones.

RAG stands for:
Retrieval: Find relevant chunks, Augmented: Add them to the prompt, Generation: Claude generates the answer

RAG pipeline:
USER ASKS: "What is gradient descent?"         ↓
        STEP 1 — Embed the question
        "What is gradient descent?" → [0.2, 0.8, 0.1, ...]
        Who does it:sentence-transformers library does it. Specifically a pre-trained model called all-MiniLM-L6-v2.
        What that model is: A neural network trained on millions of sentences. It learned to convert any text into 384 numbers that capture meaning.
                    ↓
        STEP 2 — Search ChromaDB
        Compare against all 523 chunk embeddings
        Find top 3 most similar chunks
                    ↓
        STEP 3 — Build prompt
        "Here is the context: {chunk1} {chunk2} {chunk3}
         Answer this question: What is gradient descent?"
                    ↓
        STEP 4 — Send to Claude
        Claude reads only those 3 chunks and answers
                    ↓
        STEP 5 — Show answer to user

Why RAG is powerful:
Without RAG → Claude answers from its training data → can hallucinate, make things up. ie Claude has never seen your PDF. It only knows what it was trained on — books, websites, articles from the internet up to its training cutoff.
It sounds confident. It might even be partially right. But it's not from YOUR document.

With RAG → Claude answers from YOUR document → grounded, accurate, specific.
Every serious AI product uses this pattern. ChatGPT plugins, Notion AI, Google NotebookLM — all RAG under the hood.

how does pre-trained model called all-MiniLM-L6-v2. works?
input: "What is gradient descent?"
            ↓
Split into tokens (small word pieces)
["What", "is", "gradient", "des", "##cent", "?"]
            ↓
Each token goes through neural network layers
(transformers — attention mechanism)
            ↓
All token vectors get averaged/pooled
            ↓
One vector of 384 numbers
[0.2, 0.8, 0.1, 0.9, 0.3, ... x384]

The model was trained so that similar meanings produce similar vectors. It learned this from reading millions of sentence pairs.

In code it's literally 2 lines:
pythonfrom sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("What is gradient descent?")

Claude alone    = smart human, no phone, answers from memory only

Claude + RAG    = same smart human, but with a phone
                  can look up your specific document instantly
                  answers based on what they just read

"What is gradient descent?" → 384 numbers
        ↓
ChromaDB compared against all 523 chunk embeddings
        ↓
Found top 3 most similar chunks using cosine similarity
        ↓
Returned actual text from your ML textbook

The chunks talk about objective functions, regularization, neural network weights — all related to optimization and gradient descent. It didn't just match the words "gradient descent" — it understood the meaning and found related content.
That's the power of embeddings over simple word search.


