import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Tuple


class JobFileGenerator:
    """
    모델, 토크나이저 및 FAISS 데이터베이스를 로드하고,
    사용자 입력에 따른 Job File을 생성하는 클래스.
    이번에는 학습(추론) 없이 FAISS DB에 있는 내용을 그대로 사용합니다.
    """
    def __init__(self, model_path: str, db_path: str, device: str = None) -> None:
        self.model_path = model_path
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 및 토크나이저 로드 (여기서는 참고용으로만 로드)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # FAISS 데이터베이스 로드
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists(db_path):
            print("❌ FAISS 데이터베이스를 찾을 수 없습니다. 먼저 데이터베이스를 생성하세요.")
            exit()
        self.vector_db = FAISS.load_local(db_path, self.embedding_model, allow_dangerous_deserialization=True)
        print("✅ 모델 및 FAISS 데이터베이스 로드 완료!")

        # 입력값 검증을 위한 기준 설정
        self.robot_models = ["H2017", "HS220"]
        self.gripper_mapping = {
            "A": "Magnetic Gripper",
            "B": "Two-Finger Gripper",
            "C": "Three-Finger Gripper",
            "D": "Suction Gripper"
        }

    def validate_inputs(self, robot_model: str, gripper_type: str) -> str:
        """
        사용자 입력값(로봇 모델, 그리퍼 타입)을 검증하고,
        그리퍼 타입에 대응하는 이름을 반환합니다.
        """
        if robot_model not in self.robot_models:
            raise ValueError(f"❌ Error: Invalid robot model '{robot_model}'. Please choose 'H2017' or 'HS220'.")
        if gripper_type not in self.gripper_mapping:
            raise ValueError(f"❌ Error: Invalid gripper type '{gripper_type}'. Please choose 'A', 'B', 'C' or 'D'.")
        return self.gripper_mapping[gripper_type]

    def retrieve_context(self, robot_model: str, gripper_name: str) -> str:
        """
        FAISS 데이터베이스를 활용하여 입력값과 유사한 Job File 문서를 검색합니다.
        검색된 내용이 있다면 그대로 반환합니다.
        """
        query = f"{robot_model}: {gripper_name} Gripper Installation"
        retrieved_docs = self.vector_db.similarity_search(query, k=1)
        if retrieved_docs:
            return retrieved_docs[0].page_content
        else:
            return "No relevant job file found."

    def generate_job_file(
        self, 
        robot_model: str, 
        gripper_type: str, 
        max_new_tokens: int = 50,  
        temperature: float = 0.4,
        top_p: float = 0.7, 
        max_length: int = 512
    ) -> Tuple[str, str]:
        """
        FAISS 검색 결과를 참고하여 Job File을 생성합니다.
        모델 추론은 진행하지 않고, FAISS에서 검색된 내용을 그대로 반환합니다.
        """
        # 입력값 검증
        gripper_name = self.validate_inputs(robot_model, gripper_type)
        # FAISS 검색으로 관련 문맥 검색
        context = self.retrieve_context(robot_model, gripper_name)
        # FAISS DB에 있는 내용을 그대로 사용
        jobfile_response = context.strip()
        return gripper_name, jobfile_response

    def determine_file_extension(self, robot_model: str) -> str:
        """
        로봇 모델에 따라 Job File 확장자를 결정합니다.
        """
        return "drl" if robot_model == "H2017" else "job"


class IOHandler:
    """
    사용자 입력과 출력, 파일 저장 등의 I/O를 담당하는 클래스.
    """
    @staticmethod
    def get_user_input() -> Tuple[str, str]:
        robot_model = input("🤖 Enter the robot model (H2017, HS220): ").strip().upper()
        gripper_type = input("🛠️ Enter the gripper type: ").strip().upper()
        return robot_model, gripper_type

    @staticmethod
    def save_job_file(content: str, robot_model: str, gripper_type: str, file_extension: str) -> str:
        job_file_path = f"./robot_job_{robot_model}_{gripper_type}.{file_extension}"
        with open(job_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return job_file_path

    @staticmethod
    def display_results(robot_model: str, gripper_name: str, job_file_path: str, jobfile_response: str) -> None:
        print("\n🔹 **Generated Job File:**")
        print(f"🤖 Robot Model: {robot_model}")
        print(f"🛠️ Gripper Type: {gripper_name}")
        print(f"📂 Job File Saved: {job_file_path}")
        print(f"\n📜 **Generated Job File Content:**\n{jobfile_response}")


def main() -> None:
    model_path = "./fine_tuned_llama_gripper_jobfile_faiss"
    db_path = "faiss_robot_gripper_jobfile_db"

    generator = JobFileGenerator(model_path, db_path)

    robot_model, gripper_type = IOHandler.get_user_input()

    try:
        gripper_name, jobfile_response = generator.generate_job_file(robot_model, gripper_type)
    except ValueError as e:
        print(e)
        return

    file_extension = generator.determine_file_extension(robot_model)
    job_file_path = IOHandler.save_job_file(jobfile_response, robot_model, gripper_type, file_extension)

    IOHandler.display_results(robot_model, gripper_name, job_file_path, jobfile_response)


if __name__ == "__main__":
    main()
