# ========================================
# CELL 1: Imports and Setup
# ========================================

import os
import warnings
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("✅ Imports completed successfully!")

# ========================================
# CELL 2: Initialize Georgian mGPT Model
# ========================================

# Initialize the mGPT-1.3B-georgian model
model_name = "ai-forever/mGPT-1.3B-georgian"

print(f"Loading {model_name}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for compatibility
    low_cpu_mem_usage=True,
    device_map=None  # Keep on CPU for stability
)

print("✅ Georgian mGPT model loaded successfully!")

# ========================================
# CELL 3: Simple Embeddings Class
# ========================================

class SimpleEmbeddings:
    """Simple embeddings using transformers without sentence-transformers dependency"""
    
    def __init__(self):
        # Use a simple multilingual model for embeddings
        from transformers import AutoModel, AutoTokenizer
        
        self.embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts):
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded_input = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.numpy()
    
    def embed_documents(self, texts):
        """LangChain compatible method"""
        return self.encode(texts).tolist()
        
    def embed_query(self, text):
        """LangChain compatible method"""
        return self.encode([text])[0].tolist()

# Initialize embeddings
embeddings = SimpleEmbeddings()
print("✅ Embeddings initialized!")

# ========================================
# CELL 4: Load and Process Documents
# ========================================

# Load PDF - Update this path to your actual PDF file
pdf_path = "Georgia_codex.pdf"  # Update this path

if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ PDF loaded: {len(documents)} pages")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"✅ Documents split into {len(splits)} chunks")
else:
    print(f"❌ PDF file not found: {pdf_path}")
    print("Please update the pdf_path variable with the correct path to your PDF file")
    # Create dummy documents for testing
    from langchain.schema import Document
    splits = [
        Document(page_content="საქართველო არის ქვეყანა კავკასიაში. ეს არის ძალიან ლამაზი ადგილი.", metadata={"page": 1}),
        Document(page_content="თბილისი არის საქართველოს დედაქალაქი. ეს არის ძველი და ისტორიული ქალაქი.", metadata={"page": 2})
    ]
    print("✅ Using dummy documents for testing")

# ========================================
# CELL 5: Create Vector Store
# ========================================

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db_georgian"
)

print("✅ Vector store created!")

# ========================================
# CELL 6: Text Generation Function
# ========================================

def generate_georgian_response(prompt, max_new_tokens=200):
    """Generate response using the Georgian mGPT model"""
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    
    return response

print("✅ Text generation function ready!")

# ========================================
# CELL 7: RAG Pipeline Setup
# ========================================

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def rag_pipeline(question):
    """Complete RAG pipeline"""
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create Georgian prompt
    prompt = f"""თქვენ ხართ დამხმარე ასისტენტი, რომელიც პასუხობს კითხვებს მოწოდებული კონტექსტის საფუძველზე.

კონტექსტი:
{context}

კითხვა: {question}

გთხოვთ, გასცეთ ზუსტი და სასარგებლო პასუხი ქართულ ენაზე. თუ კონტექსტში არ არის საკმარისი ინფორმაცია, გთხოვთ, ეს აღნიშნოთ.

პასუხი:"""
    
    # Generate response
    response = generate_georgian_response(prompt, max_new_tokens=150)
    
    return response

print("✅ RAG pipeline setup complete!")

# ========================================
# CELL 8: Test the System
# ========================================

# Test questions
test_questions = [
    "რა არის საქართველო?",
    "What is Georgia?",
    "თბილისის შესახებ რა იცით?",
    "რა ინფორმაცია არის ამ დოკუმენტში?"
]

print("🧪 Testing RAG system:")
print("=" * 50)

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. კითხვა: {question}")
    print("-" * 30)
    
    try:
        response = rag_pipeline(question)
        print(f"პასუხი: {response}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 50)

# ========================================
# CELL 9: Interactive Query Function
# ========================================

def ask_question(question):
    """Simple function to ask questions"""
    try:
        return rag_pipeline(question)
    except Exception as e:
        return f"Error: {str(e)}"

print("✅ Interactive function ready!")
print("\nExample usage:")
print("answer = ask_question('თქვენი კითხვა აქ')")
print("print(answer)")

# ========================================
# CELL 10: Quick Test
# ========================================

# Quick test
test_question = "რა არის საქართველო?"
test_answer = ask_question(test_question)

print(f"\n🔍 Quick test:")
print(f"კითხვა: {test_question}")
print(f"პასუხი: {test_answer}")

print("\n🎉 Georgian mGPT RAG system is ready!")
print("You can now ask questions using: ask_question('your question here')")
