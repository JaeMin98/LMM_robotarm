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
                    "gripper": "Magnetic Gripper",
                    "jobfile": """
                    MOVEJ S=60 A=100 T=1 Coordinates=[-0.2, -17.54, 138.34, -0.33, -0.1, -0.43]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.254, -312.028, 665.51, 90.28, 179.04, 89.8]
                    MOVEJ S=60 A=100 T=1 Coordinates=[-23.71, 17.21, 114.01, -33.88, -45.42, -64.36]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.265, -312.076, 668.308, 90.28, 179.04, 89.8]
                    MOVEJ S=60 A=100 T=1 Coordinates=[163.31, -0.51, 106.48, -61.27, -24.78, -30.66]
                    MOVEL S=250 A=1000 T=1 Coordinates=[945.196, -317.002, 999.878, 89.91, 179.11, 89]
                    END
                    """
                },

                {
                    "model": "H2017", 
                    "gripper": "Two-Finger Gripper",
                    "jobfile": """
                    MOVEJ S=60 A=100 T=1 Coordinates=[-0.01, -17.61, 138.58, 0, 0, 0.01]
                    MOVEJ S=60 A=100 T=1 Coordinates=[-11.42, 3.58, 131.84, -14.88, -46.13, -79.4]
                    MOVEL S=250 A=1000 T=1 Coordinates=[916.81, -103.143, 664.979, 89.5, 179.79, 89.74]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.442, -103.325, 664.862, 89.5, 179.79, 89.74]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.464, -103.34, 668.954, 89.5, 179.79, 89.74]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.157, -103.362, 664.981, 89.5, 179.79, 89.75]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.491, -103.338, 673.316, 89.5, 179.79, 89.74]
                    MOVEL S=250 A=1000 T=1 Coordinates=[940.266, -104.774, 1061.342, 89.5, 179.79, 89.74]
                    MOVEJ S=60 A=100 T=1 Coordinates=[185.17, -9.32, 110.09, -38.56, -11.94, -51.89]
                    END
                    """

                },

                {
                    "model": "H2017", 
                    "gripper": "Three-Finger Gripper",
                    "jobfile": """
                    MOVEJ S=60 A=100 T=1 Coordinates=[-0.01, -17.61, 138.58, 0, 0, 0.01]
                    MOVEJ S=60 A=100 T=1 Coordinates=[9.23, 5.42, 128.97, 14.78, -44.71, -100.6]
                    MOVEL S=250 A=1000 T=1 Coordinates=[918.012, 105.878, 670.193, 89.14, -179.9, 89.37]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.261, 105.574, 669.946, 89.14, -179.9, 89.36]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.288, 105.576, 672.703, 89.14, -179.9, 89.36]
                    MOVEL S=250 A=1000 T=1 Coordinates=[938.599, 105.599, 687.721, 89.14, -179.9, 89.36]
                    MOVEL S=250 A=1000 T=1 Coordinates=[943.782, 106.013, 981.967, 89.14, -179.9, 89.36]
                    MOVEJ S=60 A=100 T=1 Coordinates=[187.21, -4.51, 112.85, 27.02, -21.97, -116.06]
                    END
                    """

                },
                {
                    "model": "H2017", 
                    "gripper": "Suction Gripper",
                    "jobfile": """
                    MOVEJ S=60 A=100 T=1 Coordinates=[-0.01, -17.61, 138.58, 0, 0, 0.01]
                    MOVEJ S=60 A=100 T=1 Coordinates=[26.27, 7.24, 128.18, 34.67, -51.34, -113.78]
                    MOVEL S=250 A=1000 T=1 Coordinates=[919.237, 288.371, 664.059, 89.75, 180, 90.47]
                    MOVEL S=250 A=1000 T=1 Coordinates=[940.267, 288.281, 664.254, 89.75, 180, 90.48]
                    MOVEL S=250 A=1000 T=1 Coordinates=[940.225, 288.279, 669.061, 89.75, 180, 90.48]
                    MOVEL S=250 A=1000 T=1 Coordinates=[937.619, 288.295, 981.52, 89.75, -180, 90.48]
                    MOVEJ S=60 A=100 T=1 Coordinates=[192.77, -4.27, 113.73, 51.26, -24.62, -138.77]
                    END
                    """

                },
                # HS220 - Suction Gripper
                {
                    "model": "HS220",
                    "gripper": "Magnetic Gripper",
                    "jobfile": """
                Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
                    S1   MOVE P,S=60%,A=3,T=1  (21.7, 55.2, -16.5, -315.0, 31.2, 223.7)A
                    S2   MOVE P,S=60%,A=3,T=1  (19.3, 57.1, -17.4, -313.2, 32.1, 221.5)A
                    S3   MOVE P,S=60%,A=3,T=1  (16.0, 60.0, -18.8, -310.6, 33.5, 218.7)A
                    S4   MOVE P,S=60%,A=3,T=1  (12.7, 62.9, -20.2, -307.9, 34.9, 215.8)A
                    S5   MOVE P,S=60%,A=3,T=1  (9.5, 65.8, -21.6, -305.2, 36.3, 212.9)A
                    S6   MOVE P,S=60%,A=3,T=1  (6.2, 68.7, -23.0, -302.5, 37.7, 210.0)A
                    S7   MOVE P,S=60%,A=3,T=1  (3.0, 71.6, -24.4, -299.8, 39.1, 207.1)A
                    S8   MOVE P,S=60%,A=3,T=1  (0.0, 74.5, -25.8, -297.0, 40.5, 204.2)A
                    S9   MOVE P,S=60%,A=3,T=1  (-3.0, 77.4, -27.2, -294.2, 41.9, 201.3)A
                END

                    """
                },
                {
                    "model": "HS220",
                    "gripper": "Two-Finger Gripper",
                    "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
                S1   MOVE P,S=60%,A=3,T=1  (15.098,76.218,-23.05,-327.989,40.853,241.733)A
                S2   MOVE P,S=60%,A=3,T=1  (13.064,78.627,-23.826,-325.036,41.168,238.127)A
                S3   MOVE P,S=60%,A=3,T=1  (11.035,81.026,-24.591,-322.088,41.478,234.515)A
                S4   MOVE P,S=60%,A=3,T=1  (-7.277,102.669,-31.525,-295.541,44.318,202.025)A
                S5   MOVE P,S=60%,A=3,T=1  (-11.347,107.474,-33.065,-289.635,44.945,194.815)A
                S6   MOVE P,S=60%,A=3,T=1  (-15.407,112.289,-34.611,-283.735,45.575,187.589)A
                S7   MOVE P,S=60%,A=3,T=1  (-23.548,121.902,-37.689,-271.944,46.833,173.152)A
                S8   MOVE P,S=60%,A=3,T=1  (-29.649,129.12,-40.0,-263.098,47.776,162.327)A
                S9   MOVE P,S=60%,A=3,T=1  (-35.754,136.326,-42.315,-254.238,48.726,151.494)A
                S10   MOVE P,S=60%,A=3,T=1  (-37.793,138.727,-43.085,-251.291,49.032,147.891)A
                S11   MOVE P,S=60%,A=3,T=1  (-42.687,134.176,-41.265,-245.428,46.692,143.848)A
                S12   MOVE P,S=60%,A=3,T=1  (-47.577,129.625,-39.456,-239.56,44.357,139.794)A
                S13   MOVE P,S=60%,A=3,T=1  (-52.477,125.072,-37.627,-233.697,42.027,135.756)A
                S14   MOVE P,S=60%,A=3,T=1  (-59.811,118.234,-34.9,-224.903,38.515,129.689)A
                S15   MOVE P,S=60%,A=3,T=1  (-64.704,113.679,-33.081,-219.037,36.184,125.64)A
                S16   MOVE P,S=60%,A=3,T=1  (-69.599,109.127,-31.267,-213.176,33.845,121.603)A
                S17   MOVE P,S=60%,A=3,T=1  (-74.497,104.571,-29.445,-207.31,31.514,117.55)A
                S18   MOVE P,S=60%,A=3,T=1  (-79.392,100.017,-27.632,-201.449,29.173,113.505)A
            END
                    """
                },
                {
                    "model": "HS220",
                    "gripper": "Three-Finger Gripper",
                    "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
                S1   MOVE P,S=60%,A=3,T=1  (14.712,77.815,-22.347,-326.731,41.235,240.375)A
                S2   MOVE P,S=60%,A=3,T=1  (12.789,80.354,-23.089,-324.102,41.554,236.912)A
                S3   MOVE P,S=60%,A=3,T=1  (10.867,82.883,-23.83,-321.471,41.874,233.442)A
                S4   MOVE P,S=60%,A=3,T=1  (-6.123,101.283,-30.997,-296.745,44.121,204.342)A
                S5   MOVE P,S=60%,A=3,T=1  (-10.215,106.239,-32.542,-290.855,44.753,197.156)A
                S6   MOVE P,S=60%,A=3,T=1  (-14.303,111.191,-34.089,-284.963,45.387,189.969)A
                S7   MOVE P,S=60%,A=3,T=1  (-22.482,120.984,-37.175,-273.163,46.651,175.604)A
                S8   MOVE P,S=60%,A=3,T=1  (-28.623,128.262,-39.506,-264.321,47.598,164.894)A
                S9   MOVE P,S=60%,A=3,T=1  (-34.754,135.525,-41.841,-255.465,48.552,154.186)A
                S10   MOVE P,S=60%,A=3,T=1  (-37.812,138.923,-42.925,-251.521,49.034,147.681)A
                S11   MOVE P,S=60%,A=3,T=1  (-41.952,135.112,-41.112,-246.215,46.995,144.515)A
                S12   MOVE P,S=60%,A=3,T=1  (-46.096,131.301,-39.304,-240.907,44.956,141.347)A
                S13   MOVE P,S=60%,A=3,T=1  (-50.236,127.489,-37.492,-235.597,42.916,138.18)A
                S14   MOVE P,S=60%,A=3,T=1  (-57.542,120.667,-34.783,-226.815,39.412,132.159)A
                S15   MOVE P,S=60%,A=3,T=1  (-62.428,116.124,-32.972,-220.962,37.091,128.121)A
                S16   MOVE P,S=60%,A=3,T=1  (-67.321,111.583,-31.162,-215.107,34.764,124.081)A
                S17   MOVE P,S=60%,A=3,T=1  (-72.218,107.038,-29.351,-209.25,32.439,120.043)A
                S18   MOVE P,S=60%,A=3,T=1  (-77.113,102.497,-27.539,-203.391,30.112,116.003)A
            END
                    """
                },
                {
                    "model": "HS220",
                    "gripper": "Suction Gripper",
                    "jobfile": """
                Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
                    S1   MOVE P,S=60%,A=3,T=1  (18.5, 48.7, -13.7, -308.9, 28.9, 218.2)A
                    S2   MOVE P,S=60%,A=3,T=1  (16.0, 50.5, -14.5, -307.1, 29.7, 216.3)A
                    S3   MOVE P,S=60%,A=3,T=1  (12.5, 53.2, -15.8, -304.5, 31.1, 213.5)A
                    S4   MOVE P,S=60%,A=3,T=1  (9.0, 55.9, -17.1, -301.8, 32.5, 210.6)A
                    S5   MOVE P,S=60%,A=3,T=1  (5.5, 58.6, -18.4, -299.1, 33.9, 207.7)A
                    S6   MOVE P,S=60%,A=3,T=1  (2.0, 61.3, -19.7, -296.4, 35.3, 204.8)A
                    S7   MOVE P,S=60%,A=3,T=1  (-1.5, 64.0, -21.0, -293.6, 36.7, 201.9)A
                    S8   MOVE P,S=60%,A=3,T=1  (-5.0, 66.7, -22.3, -290.8, 38.1, 199.0)A
                    S9   MOVE P,S=60%,A=3,T=1  (-8.5, 69.4, -23.6, -288.0, 39.5, 196.1)A
                END
                    """
                }
            ]

            # Job File 템플릿 데이터를 JSON 파일로 저장
            with open("robot_jobfile_templates.json", "w", encoding="utf-8") as f:
                json.dump({"jobfile_templates": jobfile_templates}, f, indent=4)

            # 각 템플릿을 하나의 문서로 결합
            documents = [
                f"{entry['model']}: {entry['gripper']}\n{entry['jobfile']}"
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
    def __init__(self, model, tokenizer, dataloader, device, lr: float = 2e-5, num_epochs: int = 20):
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
    finetuner = LlamaJobFileFineTuner(model, tokenizer, dataloader, device, lr=2e-5, num_epochs=20)
    finetuner.train()

    # 6. 파인튜닝된 모델 저장
    save_path = "./fine_tuned_llama_gripper_jobfile_faiss"
    finetuner.save_model(save_path)


if __name__ == "__main__":
    main()
