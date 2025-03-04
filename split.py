from langchain_community.document_loaders import RecursiveUrlLoader

# Load the document
loader = RecursiveUrlLoader(
    "https://docs.python.org/3/tutorial/controlflow.html",
    max_depth=1,
)

html_docs = loader.load()

from langchain_community.document_transformers import MarkdownifyTransformer
import re

# Custom transformer that extends the MarkdownifyTransformer to strip trailing newlines from code blocks
class CustomMarkdownifyTransformer(MarkdownifyTransformer):
    def transform_documents(self, documents):
        transformed_docs = super().transform_documents(documents)
        for doc in transformed_docs:
            if hasattr(doc, 'page_content'):
                doc.page_content = self._strip_trailing_newline_from_code_blocks(doc.page_content)
        return transformed_docs

    def _strip_trailing_newline_from_code_blocks(self, text):
        # Regex to find code blocks and ensure they end with a newline before the closing backticks
        code_block_pattern = re.compile(r'(```\w*\n[\s\S]*?)```')
        return code_block_pattern.sub(lambda match: match.group(1).rstrip() + '\n```', text)

# Transform the document to Markdown format
md = CustomMarkdownifyTransformer()
md_docs = md.transform_documents(html_docs)
for i in range(0, len(md_docs)):
    md_docs[i].page_content=md_docs[i].page_content.replace("\n\n```","\n```python")

# Save the Markdown document
with open('md_docs.md', 'w') as f:
    f.write(md_docs[0].page_content)
    
#################################################################################
    
from langchain.schema import Document

sicbh_include_headers_in_content = False
sicbh_filter_headers = ["Table of Contents", "This Page", "Navigation"]
sicbh_show_unwanted_chunks_metadata = False

# Function to divide the Markdown documents into chunks based on headers
def split_into_chunks_by_headers(md_docs):
    chunks = []
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
    code_block_pattern = re.compile(r'^\s*```')
    in_code_block = False

    for doc in md_docs:
        if hasattr(doc, 'page_content'):
            lines = doc.page_content.split('\n')
        else:
            raise AttributeError("Document object has no 'page_content' attribute")

        current_chunk = {'metadata': {}, 'content': ''}
        current_headers = {}
        prev_header_level = 0

        for line in lines:            
            if code_block_pattern.match(line):
                in_code_block = not in_code_block

            if not in_code_block:
                match = header_pattern.match(line)
                if match:
                    # If there is content in the current chunk, add it to the chunks list
                    if current_chunk['content']:
                        current_chunk['content'] = current_chunk['content'].strip()
                        chunks.append(current_chunk)
                        current_chunk = {'metadata': {}, 'content': ''}

                    # Extract the header level and text
                    header_level = len(match.group(1))
                    header_text = match.group(2)

                    # Clean the header text
                    header_text = re.sub(r'\\', '', header_text)
                    header_text = re.sub(r'\[¶\]\(.*?\)', '', header_text).strip()

                    # Update the current headers
                    header_key = f'Header {header_level}'
                    if header_level > prev_header_level:
                        current_headers[header_key] = header_text
                    else:
                        del current_headers[f'Header {prev_header_level}']
                        current_headers[header_key] = header_text

                    # Add the header line to metadata
                    current_chunk['metadata'] = current_headers.copy()

                    # Optionally add the cleaned header text to content
                    if sicbh_include_headers_in_content:
                        current_chunk['content'] += f"{match.group(1)} {header_text}\n"

                    # Update the previous header level
                    prev_header_level = header_level
                else:
                    current_chunk['content'] += line + '\n'
            else:
                current_chunk['content'] += line + '\n'

        # Add the last chunk to the chunks list
        if current_chunk['content']:
            current_chunk['content'] = current_chunk['content'].strip()
            chunks.append(current_chunk)

    # Convert the chunks to Document objects, filtering out unwanted chunks
    documents = []
    unwanted_chunks = []
    for chunk in chunks:
        metadata = chunk['metadata']
        if metadata and not any(any(unwanted in value for unwanted in sicbh_filter_headers) for value in metadata.values()):
            documents.append(Document(page_content=chunk['content'], metadata=chunk['metadata']))
        else:
            unwanted_chunks.append(chunk['metadata'])

    # Optionally print the unwanted chunks metadata
    if sicbh_show_unwanted_chunks_metadata and unwanted_chunks:
        print(f"Unwanted chunks metadata:")
        for chunk in unwanted_chunks:
            print(chunk)
        print()

    return documents

# Split the Markdown documents into chunks based on headers
chunks_by_headers = split_into_chunks_by_headers(md_docs)
print(f"Number of chunks by headers: {len(chunks_by_headers)}")

# Save the chunked documents to a file
chunks_by_headers_docs_file_content = ""
for doc in chunks_by_headers:
    token_count = len(doc.page_content)
    chunks_by_headers_docs_file_content += f"\nToken count: {token_count}\n"
    chunks_by_headers_docs_file_content += f"Metadata: {doc.metadata}\n"
    chunks_by_headers_docs_file_content += f"Content:\n{doc.page_content}\n\n"
    chunks_by_headers_docs_file_content += f"-------------\n"

with open('chunked_by_header_docs.txt', 'w') as f:
    f.write(chunks_by_headers_docs_file_content)

#################################################################################

import re
from langchain.schema import Document

scbt_text_follow_code_block = False

# Function to split a chunk into smaller parts based on token count
def split_chunk_by_tokens(content,  max_tokens):
    # Split content into code blocks and paragraphs
    parts = re.split(r'(\n```.*?\n.*?\n```)', content, flags=re.DOTALL)
    final_parts = []
    for part in parts:
        if part.startswith('\n```'):
            final_parts.append(part.strip())   
        else:
            #sub_parts = re.split(r'\.\s+|\n', part)             
            #sub_parts = part.split('\n')      
            #final_parts.extend([sub.strip() for sub in sub_parts if sub.strip()])            
            sub_parts = re.split(r'(\.\s+)', part)  # '. ' 기준으로 문장 분리하되 구분자 유지
            merged_parts = []
            for i in range(0, len(sub_parts) - 1, 2):
                merged_parts.append(sub_parts[i] + sub_parts[i + 1])
            if len(sub_parts) % 2 == 1:
                merged_parts.append(sub_parts[-1])
            final_parts.extend([sub.strip() for sub in merged_parts if sub.strip()])

    # Remove newlines from the start and end of each part
    parts = final_parts

    # Calculate total tokens
    total_tokens = sum(len(part) for part in parts)
    target_tokens_per_chunk = total_tokens // (total_tokens // max_tokens + 1)

    # Initialize variables
    chunks = []
    current_chunk = ""
    current_token_count = 0

    # Iterate over the parts and merge them if needed
    i = 0
    while i < len(parts):
        part = parts[i]

        # Merge parts if the current part ends with ":" or "```" (if enabled) and has a following part
        while (part.endswith(":") or (scbt_text_follow_code_block and part.endswith("```"))) and i + 1 < len(parts):
            part += "\n\n" + parts[i + 1]
            i += 1  # Skip the next part as it has been merged

        # Calculate the token count of the part        
        part_token_count = len(part)

        # Split the part into smaller parts if it exceeds the target token count
        if current_token_count + part_token_count > target_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = part
            current_token_count = part_token_count
        else:
            current_chunk += "\n\n" + part if current_chunk else part
            current_token_count += part_token_count

        i += 1

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to divide the Markdown documents into chunks based on token count
def split_into_chunks_by_tokens(chunks, max_tokens):
    split_chunks = []

    for chunk in chunks:
        token_count = len(chunk.page_content)
        if token_count > max_tokens:
            split_texts = split_chunk_by_tokens(chunk.page_content,  max_tokens)
            for text in split_texts:
                split_chunks.append(Document(page_content=text, metadata=chunk.metadata))
        else:
            split_chunks.append(chunk)

    return split_chunks

# Initialize the tokenizer
chunk_max_tokens = 500

# Split the chunks into smaller parts based on token count
chunked_docs = split_into_chunks_by_tokens(chunks_by_headers, chunk_max_tokens)
print(f"Number of chunks by tokens: {len(chunked_docs)}")

# Save the chunked documents to a file
chunked_docs_file_content = ""
tcs=[]
for doc in chunked_docs:
    token_count = len(doc.page_content)
    tcs.append(token_count)
    chunked_docs_file_content += f"\nToken count: {token_count}\n"
    chunked_docs_file_content += f"Metadata: {doc.metadata}\n"
    chunked_docs_file_content += f"Content:\n{doc.page_content}\n\n"
    chunked_docs_file_content += f"-------------\n"

with open('chunked_docs.txt', 'w') as f:
    f.write(chunked_docs_file_content)
print('MAX:',max(tcs))
print('MIN:',min(tcs))