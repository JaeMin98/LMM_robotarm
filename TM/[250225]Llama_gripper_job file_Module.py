import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class ModelManager:
    @staticmethod
    def load_model_and_tokenizer(model_name: str, token: str, device: torch.device):
        """
        주어진 모델 이름과 HF 토큰을 사용해 모델과 토크나이저를 로드합니다.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=token
        ).to(device)
        return model, tokenizer

    @staticmethod
    def prepare_tokenizer(tokenizer, model):
        """
        토크나이저에 pad_token이 없으면 추가하고, 모델의 토큰 임베딩 크기를 조정합니다.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))


class FAISSManager:
    @staticmethod
    def setup_faiss_db(db_path: str) -> FAISS:
        """
        지정된 경로에 FAISS 데이터베이스가 없으면 Job File 템플릿 데이터를 기반으로 생성하고,
        존재하면 로드합니다.
        """
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists(db_path):
            print("🔹 FAISS 데이터베이스가 존재하지 않음. 새로 생성 중...")
            # Job File 템플릿 데이터 정의
            jobfile_templates = [
                # H2017 - Suction Gripper
                {
                    "model": "H2017",
                    "gripper": "Suction Gripper",
                    "description": "Suction Gripper를 장착하는 Job File.",
                    "jobfile": json.dumps({
                        "task": "Attach Suction Gripper",
                        "steps": [
                            {"action": "MOVEJ", "S": 50, "A": 3, "T": 2, "coordinates": [10, 20, 30, 0, 0, 0]},
                            {"action": "ATTACH", "gripper": "Suction Gripper"},
                            {"action": "MOVEJ", "S": 50, "A": 3, "T": 2, "coordinates": [0, 0, 0, 0, 0, 0]}
                        ]
                    }, indent=2)
                },
                # H2017 - Parallel Jaw Gripper
                {
                    "model": "H2017",
                    "gripper": "Parallel Jaw Gripper",
                    "description": "Parallel Jaw Gripper를 장착하는 Job File.",
                    "jobfile": json.dumps({
                        "task": "Attach Parallel Jaw Gripper",
                        "steps": [
                            {"action": "MOVEL", "S": 30, "A": 2, "T": 1, "coordinates": [15, 25, 35, 0, 0, 0]},
                            {"action": "ATTACH", "gripper": "Parallel Jaw Gripper"},
                            {"action": "MOVEL", "S": 30, "A": 2, "T": 1, "coordinates": [0, 0, 0, 0, 0, 0]}
                        ]
                    }, indent=2)
                },
                # H2017 - Magnetic Gripper
                {
                    "model": "H2017",
                    "gripper": "Magnetic Gripper",
                    "description": "Magnetic Gripper를 장착하는 Job File.",
                    "jobfile": json.dumps({
                        "task": "Attach Magnetic Gripper",
                        "steps": [
                            {"action": "MOVEP", "S": 40, "A": 2, "T": 1, "coordinates": [12, 18, 28, 0, 0, 0]},
                            {"action": "ATTACH", "gripper": "Magnetic Gripper"},
                            {"action": "MOVEP", "S": 40, "A": 2, "T": 1, "coordinates": [0, 0, 0, 0, 0, 0]}
                        ]
                    }, indent=2)
                },
                # HS220 - Suction Gripper
                {
                    "model": "HS220",
                    "gripper": "Suction Gripper",
                    "description": "Suction Gripper를 장착하는 Job File.",
                    "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
            S1   MOVE P,S=60%,A=3,T=1  (10.5, 20.3, -5.7, -327.5, 40.2, 241.0)A
            S2   ATTACH Suction Gripper
            S3   MOVE P,S=60%,A=3,T=1  (0, 0, 0, 0, 0, 0)A
            END
                    """
                },
                # HS220 - Parallel Jaw Gripper
                {
                    "model": "HS220",
                    "gripper": "Parallel Jaw Gripper",
                    "description": "Parallel Jaw Gripper를 장착하는 Job File.",
                    "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
            S1   MOVE P,S=60%,A=3,T=1  (14.3, 22.8, -6.2, -320.7, 38.5, 237.2)A
            S2   ATTACH Parallel Jaw Gripper
            S3   MOVE P,S=60%,A=3,T=1  (0, 0, 0, 0, 0, 0)A
            END
                    """
                },
                # HS220 - Magnetic Gripper
                {
                    "model": "HS220",
                    "gripper": "Magnetic Gripper",
                    "description": "Magnetic Gripper를 장착하는 Job File.",
                    "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
            S1   MOVE P,S=60%,A=3,T=1  (18.0, 30.1, -8.5, -312.4, 35.7, 232.5)A
            S2   ATTACH Magnetic Gripper
            S3   MOVE P,S=60%,A=3,T=1  (0, 0, 0, 0, 0, 0)A
            END
                    """
                }
            ]

            # Job File 템플릿 데이터를 JSON 파일로 저장
            with open("robot_jobfile_templates.json", "w", encoding="utf-8") as f:
                json.dump({"jobfile_templates": jobfile_templates}, f, indent=4)

            # 각 템플릿을 하나의 문서로 결합
            documents = [
                f"{entry['model']}: {entry['description']}\n{entry['jobfile']}"
                for entry in jobfile_templates
            ]
            vector_db = FAISS.from_texts(documents, embedding_model)
            vector_db.save_local(db_path)
            print("✅ FAISS 데이터베이스 생성 완료!")
        else:
            vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
            print("✅ FAISS 데이터베이스 로드 완료!")
        return vector_db


class JobFileDataset(Dataset):
    """
    파인튜닝용 Job File 데이터셋.
    각 데이터 샘플에 대해 FAISS를 활용하여 관련 문맥(context)을 검색하고,
    이를 입력 텍스트에 포함시킵니다.
    """
    def __init__(self, data, tokenizer, vector_db, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        # FAISS 검색을 활용하여 관련 문맥 검색
        retrieved_docs = self.vector_db.similarity_search(item["text"], k=1)
        context = retrieved_docs[0].page_content if retrieved_docs else ""

        # 입력 텍스트 구성: 검색된 FAISS 문맥을 Context로 활용
        input_text = f"Context: {context}\n\n{item['text']}"
        output_text = item["label"]

        # 토크나이징 (padding, truncation, max_length 적용)
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
        # 패딩 토큰은 손실 계산 시 무시 (-100)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids["input_ids"].squeeze(0),
            "labels": labels
        }


class LlamaJobFileFineTuner:
    """
    파인튜닝을 위한 클래스.
    모델 학습, 옵티마이저 설정, 손실 계산 및 모델 저장 기능을 포함합니다.
    """
    def __init__(self, model, tokenizer, dataloader, device, lr: float = 2e-5, num_epochs: int = 5):
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
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=inputs["input_ids"], labels=inputs["labels"])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.4f}")

    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("✅ Fine-Tuning 완료 및 모델 저장!")


def load_dataset(dataset_path: str) -> list:
    """
    주어진 JSON 파일에서 'jobfile_pairs' 키를 기반으로 데이터셋을 로드합니다.
    """
    if not os.path.exists(dataset_path):
        print("❌ 데이터셋 파일이 없습니다. 먼저 데이터를 생성하세요.")
        exit()
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    data = dataset_json["jobfile_pairs"]
    return data


def main():
    # 1. 모델 및 디바이스 설정
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    HUGGINGFACE_TOKEN = "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"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = ModelManager.load_model_and_tokenizer(model_name, HUGGINGFACE_TOKEN, device)
    ModelManager.prepare_tokenizer(tokenizer, model)

    # 2. FAISS 데이터베이스 생성 또는 로드 (Job File 템플릿 기반)
    db_path = "faiss_robot_gripper_jobfile_db"
    vector_db = FAISSManager.setup_faiss_db(db_path)

    # 3. 파인튜닝을 위한 데이터셋 로드
    dataset_path = "robot_jobfile_dataset.json"
    data = load_dataset(dataset_path)

    # 4. Dataset 및 DataLoader 구성
    dataset = JobFileDataset(data, tokenizer, vector_db, max_length=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 5. 학습 루프 수행
    finetuner = LlamaJobFileFineTuner(model, tokenizer, dataloader, device, lr=2e-5, num_epochs=5)
    finetuner.train()

    # 6. 파인튜닝된 모델 저장
    save_path = "./fine_tuned_llama_gripper_jobfile"
    finetuner.save_model(save_path)


if __name__ == "__main__":
    main()
