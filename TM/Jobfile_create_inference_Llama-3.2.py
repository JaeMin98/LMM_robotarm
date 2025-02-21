import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ 모델 및 디바이스 설정
model_path = "./fine_tuned_llama_jobfile"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Fine-Tuned 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# ✅ FAISS 데이터베이스 로드
db_path = "faiss_robot_jobfile_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

print("✅ 모델 및 FAISS 데이터베이스 로드 완료!")

# ✅ 1. 사용자 입력
robot_model = input("📝 Enter the robot model (e.g., H2017, HS220): ").strip()
command_description = input("📝 Describe the movement sequence: ").strip()

# ✅ 2. FAISS RAG 검색 (유사한 Job File 검색)
query = f"{robot_model}: {command_description}"
retrieved_docs = vector_db.similarity_search(query, k=1)

if retrieved_docs:
    context = retrieved_docs[0].page_content
else:
    context = "No relevant job file found. Provide the best possible output."

# ✅ 3. 모델 입력 구성
combined_prompt = f"Context: {context}\n\nGenerate a job file for {robot_model} with the following command: {command_description}"

# ✅ 4. 토크나이징
inputs = tokenizer(
    combined_prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)

# ✅ 5. 모델 추론 (Job File 생성)
print("\n🔹 Generating Job File...\n")

with torch.no_grad():
    output_tokens = llama_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

jobfile_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ✅ 6. 결과 출력
print("\n🔹 **Generated Job File:**")
print(jobfile_response)
