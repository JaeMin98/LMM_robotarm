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

# ✅ 1. 모델 및 디바이스 설정
model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUGGINGFACE_TOKEN = "test_token"

# ✅ Fine-Tuned Llama 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HUGGINGFACE_TOKEN
).to(device)

# ✅ 토크나이저에 pad_token 추가 (필수)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_model.resize_token_embeddings(len(tokenizer))

# ✅ FAISS 데이터베이스 경로 설정
db_path = "faiss_robot_jobfile_db"

# ✅ 2. FAISS 데이터베이스 생성 또는 업데이트
if not os.path.exists(db_path):
    print("🔹 FAISS 데이터베이스가 존재하지 않음. 새로 생성 중...")

    # ✅ Job File 템플릿 데이터 로드
    jobfile_templates = [
    # ✅ H2017 Job File 예제
        {
            "model": "H2017",
            "description": "Doosan Robotics 협동로봇 H2017의 Task Editor 기반 JSON Job File 포맷",
            "jobfile": json.dumps({
                "application": "task-editor",
                "data": {
                    "project": {
                        "name": "Task_Example",
                        "model": "H2017",
                        "fileParameter": "robot-param-01",
                        "createDate": "2025.02.17 16:00:05",
                        "updateDate": "2025.02.17 16:50:02"
                    },
                    "taskList": [
                        {
                            "id": "123456",
                            "type": "move_periodic",
                            "name": "Move Periodic",
                            "properties": {
                                "coordinates": 0,
                                "operatingMode": "sync",
                                "operationCondition": "amplitude",
                                "repeatNumber": 1,
                                "amplitude": { "x": 10, "y": 20, "z": 30, "a": 0, "b": 0, "c": 0 },
                                "period": { "x": 5, "y": 5, "z": 5, "a": 0, "b": 0, "c": 0 }
                            }
                        }
                    ]
                }
            }, indent=2)
        },
        # ✅ HS220 Job File 예제
        {
            "model": "HS220",
            "description": "Yaskawa HS220 산업용 로봇의 텍스트 기반 Job File 포맷",
            "jobfile": """
            Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0
            S1   MOVE P,S=60%,A=3,T=1  (15.098,76.218,-23.05,-327.989,40.853,241.733)A
            S2   MOVE P,S=60%,A=3,T=1  (13.064,78.627,-23.826,-325.036,41.168,238.127)A
            S3   MOVE P,S=60%,A=3,T=1  (11.035,81.026,-24.591,-322.088,41.478,234.515)A
            S4   MOVE P,S=60%,A=3,T=1  (8.995,83.434,-25.361,-319.142,41.795,230.911)A
            END
            """
        }
    ]

    # ✅ JSON 파일로 저장
    with open("robot_jobfile_templates.json", "w", encoding="utf-8") as f:
        json.dump({"jobfile_templates": jobfile_templates}, f, indent=4)

    # ✅ 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ 벡터 데이터베이스 생성
    documents = [f"{entry['model']}: {entry['description']}\nExample: {entry['jobfile']}" for entry in jobfile_templates]
    vector_db = FAISS.from_texts(documents, embedding_model)
    vector_db.save_local(db_path)
    print("✅ FAISS 데이터베이스 생성 완료!")
else:
    # ✅ 기존 FAISS 데이터베이스 로드
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS 데이터베이스 로드 완료!")

# ✅ 3. 데이터셋 로드 및 처리
dataset_path = "robot_jobfile_dataset.json"

if not os.path.exists(dataset_path):
    # ✅ 샘플 데이터 생성
    jobfile_dataset = {
        "jobfile_pairs": [
            {
                "text": "Generate a job file for H2017 robotic arm with basic movement commands.",
                "label": "The best job file format for H2017 is: {'application':'task-editor','data':{'project':{'name':'Task_H2017', 'model':'H2017'}, 'taskList':[...]}}"
            },
            {
                "text": "Create a job file for HS220 robotic arm for movement sequence.",
                "label": "The best job file format for HS220 is: Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 6  AuxAxis: 0 S1 MOVE P,S=60%,A=3,T=1 (15.098,76.218,-23.05,-327.989,40.853,241.733)A ..."
            }
        ]
    }
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(jobfile_dataset, f, indent=4)

# ✅ JSON 파일 로드
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

data = dataset_json["jobfile_pairs"]

# ✅ 4. 데이터셋 클래스 정의
class JobFileDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ✅ RAG 검색: 가장 유사한 문서 검색
        retrieved_docs = vector_db.similarity_search(item["text"], k=1)
        context = retrieved_docs[0].page_content if retrieved_docs else ""

        # ✅ 입력 텍스트 구성
        input_text = f"Context: {context}\n\n{item['text']}"
        output_text = f"{item['label']}"

        input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        label_ids = tokenizer(output_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)

        labels = label_ids["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100  # 🔹 Padding 무시

        return {"input_ids": input_ids["input_ids"].squeeze(0), "labels": labels}

# ✅ 데이터셋 및 DataLoader 설정
dataset = JobFileDataset(data, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ✅ 손실 함수 및 최적화 기법 정의
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(llama_model.parameters(), lr=2e-5)

# ✅ 체크포인트 저장 함수
def save_checkpoint(epoch):
    checkpoint_path = f"./checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': llama_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss
    }, checkpoint_path)
    print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")

# ✅ 학습 루프
num_epochs = 5
llama_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = llama_model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    save_checkpoint(epoch)

# ✅ 모델 저장
save_path = "./fine_tuned_llama_jobfile"
llama_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Fine-Tuning 완료 및 모델 저장!")
