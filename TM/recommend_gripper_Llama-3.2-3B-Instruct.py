import torch
import os
import json
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.optim as optim
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 모델 및 디바이스 설정
model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUGGINGFACE_TOKEN = "test Token"

# Fine-Tuned Llama 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HUGGINGFACE_TOKEN
).to(device)

# 토크나이저에 pad_token 추가
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_model.resize_token_embeddings(len(tokenizer))

# FAISS 데이터베이스 경로 설정
db_path = "faiss_gripper_db"

# 2. FAISS 데이터베이스 생성 (필요 시)
if not os.path.exists(db_path):
    print("🔹 FAISS 데이터베이스가 존재하지 않음. 새로 생성 중...")
    # Gripper 정보 정의
    documents = [
        "A Suction Gripper is best for soft, round objects like apples. It uses vacuum suction.",
        "A Parallel Jaw Gripper is best for flat, rigid objects like boxes. It uses mechanical clamping.",
        "A Magnetic Gripper is best for metallic objects like rods. It uses magnetism.",
        "A Suction Gripper works well on lightweight, deformable objects like foam balls.",
        "Parallel Jaw Grippers are ideal for securely grasping paper stacks without damaging them.",
        "Magnetic Grippers are commonly used in industrial settings to manipulate metal sheets and rods."
    ]

    # Hugging Face 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS 벡터 데이터베이스 생성 및 저장
    vector_db = FAISS.from_texts(documents, embedding_model)
    vector_db.save_local(db_path)
    print("✅ FAISS 데이터베이스 생성 완료!")
else:
    # 기존 FAISS 데이터베이스 로드
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS 데이터베이스 로드 완료!")

# 3. 데이터셋 로드
dataset_path = "GripperImages/gripper_dataset.json"

# JSON 파일 로드 및 형식 확인
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

if "image_text_pairs" not in dataset_json:
    raise KeyError("❌ 'image_text_pairs' key가 JSON 파일에 없습니다.")

data = dataset_json["image_text_pairs"]
if not isinstance(data, list):
    raise ValueError(f"❌ 데이터가 리스트 형식이 아닙니다: {type(data)}")


# 4. 데이터셋 클래스 정의 (RAG 기반으로 검색한 정보 추가)
class GripperDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length  # 모델 입력 길이 맞추기

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"❌ 잘못된 인덱스 접근: {idx} (데이터 길이: {len(self.data)})")

        item = self.data[idx]

        if "text" not in item or "label" not in item:
            raise KeyError(f"❌ 데이터에 'text' 또는 'label' 키가 없습니다: {item}")

        # RAG 검색: 사용자의 입력과 가장 유사한 문서 검색
        retrieved_docs = vector_db.similarity_search(item["text"], k=1)
        context = retrieved_docs[0].page_content if retrieved_docs else ""

        # 입력 텍스트: 검색된 문서를 포함하여 학습
        input_text = f"Context: {context}\n\n{item['text']}"
        output_text = f"{item['label']}"

        # 토큰 변환 (입력과 정답 모두 동일한 max_length 적용)
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        label_ids = tokenizer(
            output_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        labels = label_ids["input_ids"].squeeze(0)

        # 🔹 Padding 토큰은 손실 계산에서 제외하기 위해 ignore_index=-100 설정
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids["input_ids"].squeeze(0),
            "labels": labels
        }


# 데이터셋 및 DataLoader 설정
dataset = GripperDataset(data, tokenizer, max_length=256)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 손실 함수 및 최적화 기법 정의
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(llama_model.parameters(), lr=2e-5)

# 학습 루프
num_epochs = 3
llama_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = llama_model(
            input_ids=inputs["input_ids"], 
            labels=inputs["labels"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# 모델 저장
save_path = "./fine_tuned_llama_text"
llama_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Fine-Tuning 완료 및 모델 저장!")
