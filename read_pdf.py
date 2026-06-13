import fitz  # this is pymupdf

# open the pdf
doc = fitz.open("ml.pdf")

print(f"Total pages: {len(doc)}")

# read first 2 pages only for now
for page_num in range(2):
    page = doc[page_num]
    text = page.get_text()
    print(f"\n--- Page {page_num + 1} ---")
    print(text)