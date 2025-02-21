import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ 1. 모델 및 디바이스 설정
model_path = "./fine_tuned_llama_text"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Fine-Tuned LLaMA 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

print("✅ Fine-Tuned LLaMA 모델 로드 완료!")

# ✅ FAISS 데이터베이스 로드
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_gripper_db", embedding_model, allow_dangerous_deserialization=True)
print("✅ FAISS 데이터베이스 로드 완료!")

# ✅ 2. 사용자 입력 받기
text_input = input("📝 Enter a description of the object: ").strip()

# ✅ 3. FAISS를 이용해 가장 관련 있는 문서 검색
retrieved_docs = vector_db.similarity_search(text_input, k=1)
context = retrieved_docs[0].page_content if retrieved_docs else ""

# ✅ 4. 모델 입력 구성 (검색된 문맥 추가)
combined_prompt = f"Context: {context}\n\n{text_input}"

# ✅ 5. 토크나이징
inputs = tokenizer(
    combined_prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
).to(device)

# ✅ 6. 모델 추론
print("🔹 Processing text input...")

with torch.no_grad():
    output_tokens = llama_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,  # ✅ 충분한 문장 길이 확보
        do_sample=False,  # ✅ 안정적인 결과 생성
        pad_token_id=tokenizer.pad_token_id  
    )

gripper_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(gripper_response)
# ✅ 7. 후처리: 추천된 그리퍼 추출
valid_grippers = ["Suction Gripper", "Parallel Jaw Gripper", "Magnetic Gripper"]
pattern = "|".join(map(re.escape, valid_grippers))

match = re.search(pattern, gripper_response)
recommended_gripper = match.group(0) if match else "Unknown"

# ✅ 8. 최종 출력
print("\n🔹 **LLaMA Model Prediction:**")
print(f"📝 Object Description: {text_input}")
print(f"✅ Recommended Gripper: {recommended_gripper}")
print(f"💡 Explanation: {gripper_response}")

# ✅ 9. FAISS 검색 결과 확인
print("\n🔹 **Retrieved Context from FAISS:**")
print(context)
