import os
import json
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_model_and_tokenizer(model_name: str, token: str, device: torch.device):
    """
    주어진 모델 이름과 Hugging Face 토큰을 사용해 모델과 토크나이저를 로드합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=token
    ).to(device)
    return model, tokenizer


def prepare_tokenizer(tokenizer, model):
    """
    토크나이저에 pad_token이 없으면 추가하고, 모델의 토큰 임베딩 크기를 조정합니다.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


def load_or_create_faiss_db(db_path: str):
    """
    FAISS 벡터 데이터베이스가 존재하면 로드하고, 없으면 주어진 Gripper 문서들을 사용해 새로 생성합니다.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(db_path):
        print("🔹 FAISS 데이터베이스가 존재하지 않음. 새로 생성 중...")
        documents = [
            "A Suction Gripper is best for soft, round objects like apples. It uses vacuum suction.",
            "A Parallel Jaw Gripper is best for flat, rigid objects like boxes. It uses mechanical clamping.",
            "A Magnetic Gripper is best for metallic objects like rods. It uses magnetism.",
            "A Suction Gripper works well on lightweight, deformable objects like foam balls.",
            "Parallel Jaw Grippers are ideal for securely grasping paper stacks without damaging them.",
            "Magnetic Grippers are commonly used in industrial settings to manipulate metal sheets and rods."
        ]
        vector_db = FAISS.from_texts(documents, embedding_model)
        vector_db.save_local(db_path)
        print("✅ FAISS 데이터베이스 생성 완료!")
    else:
        vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS 데이터베이스 로드 완료!")
    return vector_db


def load_dataset(dataset_path: str):
    """
    JSON 파일에서 데이터셋을 로드합니다. 'image_text_pairs' 키의 존재와 리스트 형식을 확인합니다.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    if "image_text_pairs" not in dataset_json:
        raise KeyError("❌ 'image_text_pairs' key가 JSON 파일에 없습니다.")

    data = dataset_json["image_text_pairs"]
    if not isinstance(data, list):
        raise ValueError(f"❌ 데이터가 리스트 형식이 아닙니다: {type(data)}")
    return data


class GripperDataset(Dataset):
    """
    Gripper 이미지와 텍스트 쌍 데이터셋 클래스.
    RAG 기반 검색으로 FAISS 데이터베이스에서 관련 문서를 가져와 입력에 추가합니다.
    """
    def __init__(self, data, tokenizer, vector_db, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"❌ 잘못된 인덱스 접근: {idx} (데이터 길이: {len(self.data)})")

        item = self.data[idx]
        if "text" not in item or "label" not in item:
            raise KeyError(f"❌ 데이터에 'text' 또는 'label' 키가 없습니다: {item}")

        # RAG 검색: 사용자의 입력과 가장 유사한 문서를 검색
        retrieved_docs = self.vector_db.similarity_search(item["text"], k=1)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        input_text = f"Context: {context}\n\n{item['text']}"
        output_text = f"{item['label']}"

        # 입력과 정답 토큰화 (최대 길이, padding, truncation 적용)
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        label_ids = self.tokenizer(
            output_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        labels = label_ids["input_ids"].squeeze(0)
        # 손실 계산 시 패딩 토큰은 무시
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids["input_ids"].squeeze(0),
            "labels": labels
        }


class LlamaFineTuner:
    """
    Llama 모델의 파인튜닝을 위한 클래스.
    학습 루프, 옵티마이저 초기화, 모델 저장 등의 기능을 포함합니다.
    """
    def __init__(self, model, tokenizer, dataloader, device, lr=2e-5, num_epochs=3):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                # 모든 텐서를 디바이스로 이동
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"]
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.4f}")

    def save_model(self, save_path: str):
        """
        파인튜닝된 모델과 토크나이저를 지정한 경로에 저장합니다.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("✅ Fine-Tuning 완료 및 모델 저장!")


def main():
    # 1. 모델 및 디바이스 설정
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HUGGINGFACE_TOKEN = "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"

    model, tokenizer = load_model_and_tokenizer(model_name, HUGGINGFACE_TOKEN, device)
    prepare_tokenizer(tokenizer, model)

    # 2. FAISS 데이터베이스 생성 또는 로드
    db_path = "faiss_gripper_db"
    vector_db = load_or_create_faiss_db(db_path)

    # 3. 데이터셋 로드
    dataset_path = "GripperImages/gripper_dataset.json"
    data = load_dataset(dataset_path)

    # 4. 데이터셋 클래스 및 DataLoader 생성
    gripper_dataset = GripperDataset(data, tokenizer, vector_db, max_length=256)
    dataloader = DataLoader(gripper_dataset, batch_size=1, shuffle=True)

    # 5. 모델 파인튜닝 수행
    fine_tuner = LlamaFineTuner(model, tokenizer, dataloader, device, lr=2e-5, num_epochs=3)
    fine_tuner.train()

    # 6. 파인튜닝된 모델 저장
    save_path = "./fine_tuned_llama_text"
    fine_tuner.save_model(save_path)


if __name__ == "__main__":
    main()
