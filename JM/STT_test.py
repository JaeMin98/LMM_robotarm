import os
import time
import pyaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper

class VoiceTranscriber:
    def __init__(self, model_name="large", rate=16000, channels=1, chunk=1024,
                 silence_threshold=-50, silence_duration_ms=4000, min_meaningful_length_ms=300,
                 initial_timeout_ms=5000):
        print("Whisper 모델을 로드 중입니다...")
        self.model = whisper.load_model(model_name)
        print("Whisper 모델 로드 완료.")
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_meaningful_length_ms = min_meaningful_length_ms
        self.initial_timeout_ms = initial_timeout_ms  # 5초 내에 음성이 감지되어야 함
        self.audio_interface = pyaudio.PyAudio()

    @staticmethod
    def trim_audio(audio_segment, silence_thresh=-50, chunk_size=10):
        """
        audio_segment에서 앞뒤의 침묵 구간을 제거합니다.
        """
        nonsilent = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
        if nonsilent:
            start = nonsilent[0][0]
            end = nonsilent[-1][1]
            return audio_segment[start:end]
        return audio_segment

    def record_and_transcribe(self):
        print("녹음을 시작합니다. 엔터를 누른 후 바로 녹음되며, 5초 이내에 음성이 감지되면 녹음이 진행됩니다.")
        stream = self.audio_interface.open(format=pyaudio.paInt16,
                                           channels=self.channels,
                                           rate=self.rate,
                                           input=True,
                                           frames_per_buffer=self.chunk)
        frames = []
        sentence_audio = None
        start_time = time.time()

        try:
            while True:
                data = stream.read(self.chunk)
                frames.append(data)
                
                # 누적된 데이터를 AudioSegment로 변환
                audio_data = b"".join(frames)
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=self.audio_interface.get_sample_size(pyaudio.paInt16),
                    frame_rate=self.rate,
                    channels=self.channels
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                # 초기 5초 동안 음성이 전혀 감지되지 않으면 종료
                nonsilent = detect_nonsilent(audio_segment, min_silence_len=10, silence_thresh=self.silence_threshold)
                if not nonsilent and elapsed_ms > self.initial_timeout_ms:
                    print("5초 이내에 말이 감지되지 않았습니다.")
                    return None

                # 녹음 중 전체 길이가 침묵 감지 시간 이상이면 마지막 1초 구간 검사
                if len(audio_segment) >= self.silence_duration_ms:
                    last_segment = audio_segment[-self.silence_duration_ms:]
                    # 마지막 1초가 침묵이면 말이 끝난 것으로 판단
                    if last_segment.dBFS < self.silence_threshold:
                        if nonsilent:
                            sentence_audio = self.trim_audio(audio_segment, silence_thresh=self.silence_threshold)
                        break
        except KeyboardInterrupt:
            print("녹음이 중단되었습니다.")
            return None
        finally:
            stream.stop_stream()
            stream.close()

        if sentence_audio and len(sentence_audio) >= self.min_meaningful_length_ms:
            filename = "sentence.mp3"
            sentence_audio.export(filename, format="mp3")
            print(f"음성 파일 저장됨: {filename}")
            print("Whisper로 텍스트 변환 중...")
            result = self.model.transcribe(filename, fp16=False)
            text = result["text"]
            print("변환된 텍스트:")
            print(text)
            return text
        else:
            print("의미 있는 음성 구간이 감지되지 않았습니다.")
            return None

    def close(self):
        self.audio_interface.terminate()

def main():
    vt = VoiceTranscriber()
    print("엔터를 누르면 녹음을 시작합니다. 종료하려면 'q'를 입력하세요.")
    try:
        while True:
            user_input = input(">> ")
            if user_input.lower() == 'q':
                break
            vt.record_and_transcribe()
    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        vt.close()

if __name__ == '__main__':
    main()
