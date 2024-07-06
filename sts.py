# STEP1
from sentence_transformers import SentenceTransformer

# STEP2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP3
sentences1 = "너무 졸려요"
sentences2 = "자고 싶어요"
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# STEP4
# embeddings = model.encode(sentences)
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)
print(embeddings1.shape)
# [3, 384]

# STEP5
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])