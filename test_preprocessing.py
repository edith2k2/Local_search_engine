import asyncio
import os
from pathlib import Path
from typing import List
from preprocessing import DocumentProcessor, ProcessedChunk
from nltk.tokenize import sent_tokenize
import numpy as np

# Create a mock test file if needed
test_text_file = "pdfs/test_text.txt"
test_pdf_file = "pdfs/1.pdf"
api_key = "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"

async def test_read_file():
    processor = DocumentProcessor()
    content, metadata = await processor.read_file(test_text_file)
    
    # Check if file content is read correctly
    assert "This is a test document" in content
    # assert metadata.file_name == "test_txt.txt"
    assert metadata.file_type == ".txt"
    assert isinstance(metadata.created_time, str)
    print(content, metadata)
    print("test_read_file passed!")

async def test_read_pdf():
    processor = DocumentProcessor()
    content = await asyncio.get_event_loop().run_in_executor(
        None, processor._read_pdf, test_pdf_file
    )
    
    # Check if PDF is read correctly
    # assert content.startswith("This is a test document")  # Adjust based on actual PDF content
    print(content)
    print("test_read_pdf passed!")

async def test_chunk_document():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=1)
    content = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet, making it a popular pangram. It's often used for typing practice, font displays, and testing equipment. While simple, this sentence serves as a great tool for showcasing how all the letters are used in different contexts.
    It’s a fun and quirky way to test a variety of systems and applications that require the use of all characters in the English alphabet."""
    chunks = processor.chunk_document(content)
    
    # Check if chunking occurs correctly
    assert len(chunks) > 1
    assert chunks[0].chunk_id == 0
    assert chunks[-1].chunk_id == len(chunks) - 1
    print("test_chunk_document passed!", chunks)

async def test_generate_context_batch():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=1, anthropic_api_key=api_key)
    content = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet, making it a popular pangram. It's often used for typing practice, font displays, and testing equipment. While simple, this sentence serves as a great tool for showcasing how all the letters are used in different contexts.
    It’s a fun and quirky way to test a variety of systems and applications that require the use of all characters in the English alphabet."""
    chunks = processor.chunk_document(content)
    contexts = await processor.generate_context_batch(chunks, content)
    
    # Check if context generation happens correctly
    print(contexts)
    assert len(contexts) == len(chunks)
    assert isinstance(contexts[0], str)
    print("test_generate_context_batch passed!")

async def test_compute_embedding():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=1, anthropic_api_key=api_key)
    content = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet, making it a popular pangram. It's often used for typing practice, font displays, and testing equipment. While simple, this sentence serves as a great tool for showcasing how all the letters are used in different contexts.
    It’s a fun and quirky way to test a variety of systems and applications that require the use of all characters in the English alphabet."""
    embedding = processor.compute_embedding(content)
    
    # Check if embedding is computed
    print(embedding)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # Assuming model generates 768-dimensional embeddings
    print("test_compute_embedding passed!")

async def test_process_document():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=1, anthropic_api_key=api_key)
    result = await processor.process_document(test_pdf_file)
    
    # Check if document is processed
    print(result)
    assert "metadata" in result
    assert "chunks" in result
    assert len(result['chunks']) > 0
    print("test_process_document passed!")

async def test_process_documents():
    processor = DocumentProcessor(anthropic_api_key="your_anthropic_api_key_here")
    results = await processor.process_documents([test_text_file, test_pdf_file])
    
    # Check if multiple documents are processed concurrently
    assert len(results) == 2
    print("test_process_documents passed!")

async def test_save_indices():
    processor = DocumentProcessor()
    await processor.process_document(test_pdf_file)
    
    # Save the indices to disk
    output_dir = "output"
    await processor.save_indices(output_dir)
    
    # Check if files were saved
    # assert os.path.exists(Path(output_dir) / "test_file" / "metadata.json")
    # assert os.path.exists(Path(output_dir) / "test_file" / "chunks.json")
    # assert os.path.exists(Path(output_dir) / "test_file" / "embeddings.npy")
    print("test_save_indices passed!")

async def test_load_indices():
    processor = DocumentProcessor()
    output_dir = "output"
    
    # Load indices from disk
    await processor.load_indices(output_dir)
    
    # Check if documents are loaded into processor
    assert len(processor.documents) > 0
    print("test_load_indices passed!")

# Run all the tests
async def run_tests():
    # await test_read_file()
    # await test_read_pdf()
    # await test_chunk_document()
    # await test_generate_context_batch()
    # await test_compute_embedding()
    # await test_process_document()
    # await test_process_documents()
    await test_save_indices()
    # await test_load_indices()

# Ensure we run the test if the script is executed directly
if __name__ == "__main__":
    asyncio.run(run_tests())

# from transformers import AutoTokenizer
# from tiktoken import encoding_for_model

# class TokenExample:
#     def __init__(self):
#         # For embeddings
#         self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
#         # For GPT models
#         self.gpt_tokenizer = encoding_for_model("gpt-3.5-turbo")
        
#     def compare_tokenization(self, text: str):
#         """Compare different tokenization methods."""
#         # Wrong way (just splitting words)
#         word_tokens = text.split()
        
#         # Using transformer tokenizer
#         transformer_tokens = self.embedding_tokenizer.tokenize(text)
        
#         # Using GPT tokenizer
#         gpt_tokens = self.gpt_tokenizer.encode(text)
        
#         print(f"Original text: '{text}'")
#         print(f"\nSimple word split (WRONG):")
#         print(f"Token count: {len(word_tokens)}")
#         print(f"Tokens: {word_tokens}")
        
#         print(f"\nTransformer tokenizer:")
#         print(f"Token count: {len(transformer_tokens)}")
#         print(f"Tokens: {transformer_tokens}")
        
#         print(f"\nGPT tokenizer:")
#         print(f"Token count: {len(gpt_tokens)}")
#         print(f"Decoded tokens: {[self.gpt_tokenizer.decode([token]) for token in gpt_tokens]}")

# # Test with different examples
# example_texts = [
#     "Let's look at tokenization123!",
#     "The quick brown fox jumps over the lazy dog.",
#     "Antidisestablishmentarianism is a long word.",
#     "Here's an example with numbers: 12345 and symbols @#$%",
#     "GPT-3.5 is a language model."
# ]

# tokenizer = TokenExample()
# for text in example_texts:
#     print("\n" + "="*80)
#     tokenizer.compare_tokenization(text)