import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

texts = [
    "This is a test article about technology and artificial intelligence.",
    "The weather is expected to remain sunny throughout the weekend.",
]


# --------------------------------------------------
# SentenceTransformer implementation
# --------------------------------------------------

print("Loading SentenceTransformer...")

sentence_model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16}
)

st_embeddings = sentence_model.encode(
    texts,
    show_progress_bar=False
)

print("SentenceTransformer embeddings:")
print(st_embeddings.shape)


# --------------------------------------------------
# Transformers implementation
# --------------------------------------------------

print("\nLoading Transformers model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

transformer_model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

transformer_model.eval()

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = transformer_model(**inputs)

token_embeddings = outputs.last_hidden_state
attention_mask = inputs["attention_mask"]

mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

sum_embeddings = torch.sum(token_embeddings * mask, dim=1)

sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

mean_embeddings = sum_embeddings / sum_mask

transformer_embeddings = F.normalize(
    mean_embeddings,
    p=2,
    dim=1
)

transformer_embeddings = transformer_embeddings.cpu().numpy()

print("Transformers embeddings:")
print(transformer_embeddings.shape)


# --------------------------------------------------
# Compare
# --------------------------------------------------

difference = abs(st_embeddings - transformer_embeddings)

print("\nMaximum absolute difference:")
print(difference.max())

print("\nMean absolute difference:")
print(difference.mean())