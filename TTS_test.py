from gtts import gTTS
import os
import tempfile
import subprocess
from pydub import AudioSegment

def change_speed(sound, speed=1.0):
    """
    pydub AudioSegment의 재생 속도를 변경하는 함수
    :param sound: AudioSegment 객체
    :param speed: 속도 배율 (1.0이면 정상, 1.5면 1.5배 빠르게, 0.8이면 0.8배 느리게)
    :return: 속도가 변경된 AudioSegment 객체
    """
    # 프레임 레이트를 speed 배율만큼 변경한 후 원래 프레임 레이트로 보정
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed)})
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

class TextToSpeechUbuntu:
    def __init__(self, lang='ko', speed=1.0):
        """
        :param lang: TTS에 사용할 언어 (예: 'ko'는 한국어)
        :param speed: 재생 속도 배율 (1.0이면 정상 속도)
        """
        self.lang = lang
        self.speed = speed

    def speak(self, text):
        """
        주어진 텍스트를 음성으로 변환한 후, 재생 속도를 조절하여 Ubuntu에서 재생함.
        :param text: 읽어줄 텍스트 문자열
        """
        # gTTS를 사용하여 음성 데이터 생성 (기본 속도)
        tts = gTTS(text=text, lang=self.lang)
        
        # 임시 mp3 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        
        tts.save(temp_path)
        
        # 재생 속도 조절 (speed가 1.0이 아니면)
        if self.speed != 1.0:
            # pydub를 사용해 음성 파일 로드
            sound = AudioSegment.from_file(temp_path, format="mp3")
            # 속도 변경
            sound = change_speed(sound, self.speed)
            # 변경된 음성을 새로운 임시 파일에 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp2:
                adjusted_path = fp2.name
            sound.export(adjusted_path, format="mp3")
            os.remove(temp_path)  # 원본 임시 파일 삭제
            temp_path = adjusted_path
        
        # mpg123를 이용해 음성 파일 재생
        subprocess.call(["mpg123", temp_path])
        
        # 재생 후 임시 파일 삭제
        os.remove(temp_path)

# 예제 사용법
if __name__ == "__main__":
    # speed 매개변수: 1.0 (정상), 0.8 (느리게), 1.2 (빠르게) 등
    tts = TextToSpeechUbuntu(lang="ko", speed=1.1)
    tts.speak("안녕하세요. 이 코드는 Ubuntu에서 실행 가능한 텍스트 음성 변환 예제입니다. 재생 속도도 조절할 수 있습니다.")
