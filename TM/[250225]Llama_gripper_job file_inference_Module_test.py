import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


class JobFileGenerator:
    """
    FAISS 없이, 모델이 학습한 데이터를 기반으로 Job File을 생성하는 클래스.
    사용자가 입력한 로봇 모델과 그리퍼를 바탕으로 Job File을 새롭게 생성함.
    """
    def __init__(self, model_path: str, device: str = None) -> None:
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Fine-Tuned 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        print("✅ 모델 로드 완료! (FAISS 없이 학습 데이터 기반 생성)")

    def generate_prompt(self, robot_model: str, gripper_name: str) -> str:
        """
        FAISS 검색 없이, 모델이 학습한 데이터를 기반으로 Job File을 생성하도록 유도하는 프롬프트를 생성함.
        """
        prompt = f"""
        Generate a Job File.

        Robot Model: {robot_model}
        Gripper Type: {gripper_name}

        Job File Format:
        {{
            "task": "Attach {gripper_name}",
            "steps": [
                {{"action": "MOVEJ", "S": 50, "A": 3, "T": 2, "coordinates": [10, 20, 30, 0, 0, 0]}},
                {{"action": "ATTACH", "gripper": "{gripper_name}"}},
                {{"action": "MOVEJ", "S": 50, "A": 3, "T": 2, "coordinates": [0, 0, 0, 0, 0, 0]}}
            ]
        }}
        Generate a new Job File in the same format based on your training data.
        """
        return prompt.strip()

    def generate_job_file(
        self,
        robot_model: str,
        gripper_name: str,
        max_new_tokens: int = 10,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_length: int = 512
    ) -> Tuple[str, str]:
        """
        FAISS를 사용하지 않고, 모델이 학습한 데이터를 기반으로 Job File을 생성함.
        """
        # 🔹 프롬프트 생성
        prompt = self.generate_prompt(robot_model, gripper_name)

        # 🔹 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        print("\n🔹 Generating Job File...\n")

        # 🔹 모델 추론
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 확률적 샘플링 비활성화
                temperature=temperature,  # 창의성 줄이기
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        jobfile_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return gripper_name, jobfile_response.strip()

    def determine_file_extension(self, robot_model: str) -> str:
        """
        로봇 모델에 따라 Job File 확장자를 결정함.
        """
        return "drl" if "H" in robot_model else "job"


class IOHandler:
    """
    사용자 입력과 출력, 파일 저장 등의 I/O를 담당하는 클래스.
    """
    @staticmethod
    def get_user_input() -> Tuple[str, str]:
        robot_model = input("🤖 Enter the robot model: ").strip().upper()
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
    model_path = "./fine_tuned_llama_gripper_jobfile"

    generator = JobFileGenerator(model_path)

    robot_model, gripper_type = IOHandler.get_user_input()

    gripper_name, jobfile_response = generator.generate_job_file(robot_model, gripper_type)

    file_extension = generator.determine_file_extension(robot_model)
    job_file_path = IOHandler.save_job_file(jobfile_response, robot_model, gripper_type, file_extension)

    IOHandler.display_results(robot_model, gripper_name, job_file_path, jobfile_response)


if __name__ == "__main__":
    main()
