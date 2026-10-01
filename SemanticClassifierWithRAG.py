# =============================================================================
# Generative AI Assignment (Assignment 2)
# =============================================================================
#
# - Sections contain **TODO** markers where you must implement functions/classes.
# - At the bottom, uncomment **Section F** and run to validate your code.
# - The final submission should have **Section F** commented.
# - PLEASE NAME YOUR PYTHON FILE AS ROLLNO.py (last 5 digits of your roll number)
# - Example - If your roll no is 42069 then python file should be named as 42069.py
# - Please submit the python file in the assignment

# =============================================================================
# Imports & Utilities
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss  # requires faiss-cpu
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from sentence_transformers import SentenceTransformer

hf_logging.set_verbosity_error()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# =============================================================================
# Section A — Embeddings + FAISS
# =============================================================================

def compute_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 16,
) -> np.ndarray:
    """
    STUDENT TASK:
    - Use SentenceTransformer to encode texts.
    - Return a numpy float32 array of shape (len(texts), dim).
    """    
    #Implementation
    SenTransModel = SentenceTransformer(model_name_or_path=model_name,
                                        device=DEVICE,
                                        )
    enc_txt = SenTransModel.encode(texts,
                                     convert_to_numpy=True,
                                     batch_size=batch_size,
                                     show_progress_bar=True).astype('float32')

    return enc_txt


def build_faiss_index(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    STUDENT TASK:
    - Normalize embeddings to unit length.
    - Create a faiss.IndexFlatIP and add normalized embeddings.
    - Return (index, normalized_embeddings).
    """
    #Implementation
    norm_emb = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_emb[norm_emb==0.0] = 1.0
    normalized_embeddings = embeddings / norm_emb
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index,normalized_embeddings

# =============================================================================
# Section B — Top-k and Top-p sampling
# =============================================================================

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -1e10,
) -> torch.Tensor:
    """
    STUDENT TASK:
    - Apply top-k and/or top-p filtering to logits.
    - Return filtered logits.
    """
    #Implementation
    logits = logits.clone()

    if top_k>0:
      top_k = min(max(1,top_k),logits.size(-1))
      topk_to_keep = torch.topk(logits,top_k)
      min_allowed = topk_to_keep[0][...,-1,None]
      logits[logits < min_allowed] = filter_value

    if top_p>0.0:
      top_p = min(top_p,1.0)
      logits_sorted, indices_sorted = torch.sort(logits,descending=True)
      cumulative_probs = torch.cumsum(F.softmax(logits_sorted,dim=-1),dim=-1)

      cutoff = cumulative_probs > top_p
      cutoff[...,0] = 0
      indices_2_remove = indices_sorted[cutoff]
      if indices_2_remove.numel() > 0:
        logits[indices_2_remove] = filter_value

    return logits


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Helper: sample an index from logits with temperature scaling."""
    probs = F.softmax(logits / max(1e-8, temperature), dim=-1)
    return int(torch.multinomial(probs, 1))

# =============================================================================
# Section C — Generation with LiquidAI/LFM2-700M
# =============================================================================

MODEL_NAME = "LiquidAI/LFM2-700M"

def generate_with_sampling(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 64,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> str:
    """
    STUDENT TASK:
    - Autoregressively generate tokens using top-k/top-p filtering.
    - Return decoded text (including prompt).
    """
    #Implementation
    # model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(DEVICE)
    gen_input = inputs.clone()
    try:
      eos_token_id = tokenizer.eos_token_id
    except:
      eos_token_id = None

    with torch.no_grad():
      for i in range(max_new_tokens):
        output = model(gen_input)
        next_logits = output.logits[:, -1, :].squeeze(0)

        filtered_logits = top_k_top_p_filtering(next_logits,top_k=top_k,top_p=top_p)

        next_token_id = sample_next_token(filtered_logits,temperature=temperature)
        next_token = torch.tensor([[next_token_id]],device=DEVICE)
        gen_input = torch.cat([gen_input,next_token],dim=-1)

        if eos_token_id is not None and next_token_id == eos_token_id:
          break

    output_text = tokenizer.decode(gen_input[0], skip_special_tokens=True)
    return output_text


# =============================================================================
# Section D — Retrieval-Augmented Generation (RAG)
# =============================================================================

def rag_retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    doc_texts: List[str],
    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 3,
) -> List[Tuple[int, float, str]]:
    """
    STUDENT TASK:
    - Encode query with SentenceTransformer.
    - Normalize and search index.
    - Return list of (doc_id, score, doc_text).
    """
    #Implementation
    enc_query = compute_embeddings([query],model_name=emb_model_name)
    norm_query = np.linalg.norm(enc_query, axis=1, keepdims=True)
    norm_query[norm_query==0.0] = 1.0 
    normalized_query = enc_query / norm_query
    normalized_query = normalized_query.astype('float32')

    distances, indices = index.search(normalized_query,k=k)
    return [(int(i),float(d),doc_texts[int(i)]) for i,d in zip(indices[0],distances[0])]


def rag_generate_answer(
    query: str,
    retrieved: List[Tuple[int, float, str]],
    generator_model,
    tokenizer,
    max_new_tokens: int = 128,
) -> str:
    """
    STUDENT TASK:
    - Build a prompt by combining retrieved docs + query.
    - Generate answer using generator model.
    """
    #Implementation

    prompt = 'Context: '
    for i,d,dt in retrieved:
      prompt += dt
      prompt += '\n'
    prompt += 'Query: '+query

    
    if generator_model is None or tokenizer is None:
        return prompt
        
    generated_answer = generate_with_sampling(prompt,generator_model,tokenizer,max_new_tokens=max_new_tokens)
    return generated_answer

# =============================================================================
# Section E — Sentiment Classifier
# =============================================================================

class SentimentClassifier(nn.Module):
    """
    STUDENT TASK:
    - Implement forward pass to map embeddings -> class logits.
    """
    def __init__(self, encoder_dim: int = 384, num_labels: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_dim // 2, num_labels),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        #Implementation
        logits = self.head(embeddings)
        return logits


def build_sentiment_pipeline(
    encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_labels: int = 3,
):
    """
    STUDENT TASK:
    - Build (encoder, classifier).
    - Return both.
    """
    #Implementation
    encoder = SentenceTransformer(encoder_model_name)
    encoder_dim = encoder.get_sentence_embedding_dimension()
    classifier = SentimentClassifier(encoder_dim=encoder_dim,num_labels=num_labels)
    return encoder,classifier


############################################################
#TESTING CODE
############################################################

#=============================================================================
#Section F — Validate submission
#Uncomment the below commented code and run to see if your solutions work.
#These checks use dummy models so they don’t require heavy downloads.
#The final submission shouldn't have this code uncommented
#=============================================================================

# import traceback, sentence_transformers

# def print_result(name, ok, msg=""):
#     status = "PASS" if ok else "FAIL"
#     print(f"{status}  {name}  -- {msg}")

# # 1) compute_embeddings
# try:
#     emb = compute_embeddings(["hello world", "this is a test"], batch_size=2)
#     ok = isinstance(emb, np.ndarray) and emb.shape[0]==2 and emb.dtype==np.float32
#     print_result("compute_embeddings", ok, f"shape={getattr(emb,'shape',None)}, dtype={getattr(emb,'dtype',None)}")
# except Exception as e:
#     print_result("compute_embeddings", False, traceback.format_exc())

# # 2) build_faiss_index
# try:
#     dummy = np.random.randn(4, 64).astype(np.float32)
#     idx, norm_emb = build_faiss_index(dummy)
#     ok = (hasattr(idx, "ntotal") and isinstance(norm_emb, np.ndarray) and norm_emb.shape==dummy.shape)
#     norms = np.linalg.norm(norm_emb, axis=1)
#     if ok and not np.allclose(norms, 1.0, atol=1e-4):
#         ok = False
#         msg = f"norms not ~1 (min {norms.min():.4f})"
#     else:
#         msg = "OK"
#     print_result("build_faiss_index", ok, msg)
# except Exception:
#     print_result("build_faiss_index", False, traceback.format_exc())

# # 3) top_k_top_p_filtering
# try:
#     logits = torch.randn(200)
#     filtered = top_k_top_p_filtering(logits.clone(), top_k=5, top_p=1.0)
#     ok = isinstance(filtered, torch.Tensor) and (filtered > -1e9).sum().item() <= 5
#     print_result("top_k_top_p_filtering", ok, f"kept={(filtered > -1e9).sum().item()}")
# except Exception:
#     print_result("top_k_top_p_filtering", False, traceback.format_exc())


# # 4) generate_with_sampling (dummy model/tokenizer)
# try:
#     class DummyTokenizer:
#         def __call__(self, text, return_tensors='pt', truncation=True):
#             return {'input_ids': torch.tensor([[0]], dtype=torch.long)}
#         def decode(self, ids, skip_special_tokens=True):
#             return "decoded"

#     class DummyModel:
#         def __init__(self):
#             self._p = nn.Parameter(torch.randn(1))
#         def to(self, device): pass
#         def parameters(self): yield self._p
#         def eval(self): pass
#         def __call__(self, input_ids):
#             b, s = input_ids.shape
#             logits = torch.zeros((b, s, 8))
#             logits[:, :, 1] = 10.0
#             return type("O", (), {"logits": logits})

#     out = generate_with_sampling("Hello", DummyModel(), DummyTokenizer(), max_new_tokens=3)
#     ok = isinstance(out, str)
#     print_result("generate_with_sampling", ok, f"returned type {type(out)}")
# except Exception:
#     print_result("generate_with_sampling", False, traceback.format_exc())

# # 5) rag_retrieve (monkeypatch ST)
# try:
#     docs = ["a","b","c","d"]
#     emb_docs = np.random.randn(len(docs), 384).astype(np.float32)
#     idx, norm_emb = build_faiss_index(emb_docs)
#     class DummyST:
#         def __init__(self, *a, **kw): pass
#         def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
#             return norm_emb[2].reshape(1, -1)

#     sentence_transformers.SentenceTransformer = DummyST
#     retrieved = rag_retrieve("query", idx, docs, k=2)
#     ok = isinstance(retrieved, list) and len(retrieved) == 2
#     print_result("rag_retrieve", ok, f"retrieved={retrieved[:2]}")
# except Exception:
#     print_result("rag_retrieve", False, traceback.format_exc())

# # 6) rag_generate_answer
# try:
#     out = rag_generate_answer("who is x?", [(0, 0.9, "doc a")], None, None, max_new_tokens=5)
#     ok = isinstance(out, str)
#     print_result("rag_generate_answer", ok, f"returned type {type(out)}")
# except Exception:
#     print_result("rag_generate_answer", False, traceback.format_exc())

# # 7) SentimentClassifier forward + grads
# try:
#     model = SentimentClassifier(encoder_dim=16, num_labels=3)
#     x = torch.randn(3, 16)
#     out = model(x)
#     ok = isinstance(out, torch.Tensor) and out.shape[0] == 3
#     if ok:
#         labels = torch.randint(0, 3, (3,), dtype=torch.long)
#         loss = nn.CrossEntropyLoss()(out, labels)
#         loss.backward()
#         grads = [p.grad for p in model.parameters() if p.grad is not None]
#         ok = ok and len(grads) > 0
#     print_result("SentimentClassifier", ok, f"out_shape={getattr(out,'shape',None)}")
# except Exception:
#     print_result("SentimentClassifier", False, traceback.format_exc())

