import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class LlamaGripperPredictor:
    """
    Fine-Tuned LLaMA 모델과 FAISS 벡터 데이터베이스를 이용해
    객체 설명에 대해 그리퍼를 추천하는 기능을 수행하는 클래스.
    """
    def __init__(self, model_path: str, faiss_db_path: str, device: str = None):
        # 디바이스 설정
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.faiss_db_path = faiss_db_path

        # 모델 및 토크나이저 로드
        print("✅ Fine-Tuned LLaMA 모델 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        print("✅ Fine-Tuned LLaMA 모델 로드 완료!")

        # FAISS 데이터베이스 로드
        print("✅ FAISS 데이터베이스 로드 중...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = FAISS.load_local(self.faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS 데이터베이스 로드 완료!")

        # 유효한 그리퍼 목록 및 정규식 패턴 생성
        self.valid_grippers = ["Suction Gripper", "Parallel Jaw Gripper", "Magnetic Gripper"]
        self.gripper_pattern = "|".join(map(re.escape, self.valid_grippers))

    def retrieve_context(self, description: str) -> str:
        retrieved_docs = self.vector_db.similarity_search(description, k=1)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        return context

    def generate_response(self, description: str, context: str, max_length: int = 256, max_new_tokens: int = 100) -> str:
        combined_prompt = f"Context: {context}\n\n{description}"
        inputs = self.tokenizer(
            combined_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        print("🔹 Processing text input...")

        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response

    def extract_recommended_gripper(self, response: str) -> str:
        """
        모델의 응답에서 추천된 그리퍼를 정규식으로 추출합니다.
        """
        match = re.search(self.gripper_pattern, response)
        return match.group(0) if match else "Unknown"

    def predict(self, description: str) -> dict:
        """
        전체 예측 과정을 수행하여 추천 그리퍼, 모델 응답, 검색 문맥을 반환합니다.
        """
        context = self.retrieve_context(description)
        response = self.generate_response(description, context)
        recommended_gripper = self.extract_recommended_gripper(response)
        return {
            "description": description,
            "recommended_gripper": recommended_gripper,
            "explanation": response,
            "context": context
        }


class IOHandler:
    """
    사용자 입력과 출력 처리를 담당하는 클래스.
    """
    @staticmethod
    def get_user_input(prompt: str = "📝 Enter a description of the object: ") -> str:
        """
        사용자로부터 객체 설명을 입력 받기
        """
        return input(prompt).strip()

    @staticmethod
    def print_prediction(result: dict):
        """
        예측 결과를 출력
        """
        print("\n🔹 **LLaMA Model Prediction:**")
        print(f"📝 Object Description: {result['description']}")
        print(f"✅ Recommended Gripper: {result['recommended_gripper']}")
        print(f"💡 Explanation: {result['explanation']}")
        print("\n🔹 **Retrieved Context from FAISS:**")
        print(result['context'])


def main():
    # 설정: 모델 경로, FAISS 데이터베이스 경로, 디바이스 설정
    model_path = "./fine_tuned_llama_text"
    faiss_db_path = "faiss_gripper_db"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 예측기 객체 생성
    predictor = LlamaGripperPredictor(model_path, faiss_db_path, device)

    # 사용자 입력 처리
    user_description = IOHandler.get_user_input()

    # 예측 수행
    result = predictor.predict(user_description)

    # 결과 출력
    IOHandler.print_prediction(result)


if __name__ == "__main__":
    main()
