import pyaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import time

def trim_audio(audio_segment, silence_thresh=-50, chunk_size=10):
    """
    audio_segment에서 앞뒤의 침묵 구간을 제거합니다.
    - silence_thresh: 침묵으로 판단할 dBFS 값
    - chunk_size: 검사할 단위 (밀리초)
    """
    nonsilent = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if nonsilent:
        start = nonsilent[0][0]
        end = nonsilent[-1][1]
        return audio_segment[start:end]
    return audio_segment

# 마이크 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000        # 샘플링 레이트 (필요에 따라 조절)
CHUNK = 1024        # 한 번에 읽어올 프레임 수

# 침묵으로 간주할 조건 (1초 이상의 침묵)
SILENCE_THRESHOLD = -50      # dBFS (조절 가능)
SILENCE_DURATION_MS = 1000   # 1000ms = 1초

# 의미 있는 음성 구간의 최소 길이 (예: 500ms 이상)
MIN_MEANINGFUL_LENGTH_MS = 5000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("녹음을 시작합니다. 문장을 말해주세요. (종료하려면 Ctrl+C를 누르세요)")

frames = []         # 현재 문장의 오디오 데이터를 저장할 버퍼
sentence_count = 0  # 저장된 문장 번호

try:
    while True:
        # 마이크로부터 CHUNK 만큼 읽기
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 현재까지의 데이터를 AudioSegment로 변환 (WAV 형식으로 해석)
        audio_data = b"".join(frames)
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=p.get_sample_size(FORMAT),
            frame_rate=RATE,
            channels=CHANNELS
        )
        
        # 전체 길이가 SILENCE_DURATION_MS 이상일 때,
        if len(audio_segment) >= SILENCE_DURATION_MS:
            # 마지막 1초 구간 추출
            last_segment = audio_segment[-SILENCE_DURATION_MS:]
            # dBFS가 설정한 임계값보다 낮으면(즉, 조용하면) 문장이 끝난 것으로 판단
            if last_segment.dBFS < SILENCE_THRESHOLD:
                # 앞뒤의 불필요한 침묵 제거
                trimmed_audio = trim_audio(audio_segment, silence_thresh=SILENCE_THRESHOLD)
                # 의미 있는 길이 이상일 경우에만 저장 (예: 500ms 이상)
                if len(trimmed_audio) >= MIN_MEANINGFUL_LENGTH_MS:
                    sentence_count += 1
                    filename = f"sentence_{sentence_count}.mp3"
                    trimmed_audio.export(filename, format="mp3")
                    print(f"저장됨: {filename}")
                else:
                    print("의미 없는 구간으로 판단되어 저장하지 않음.")
                # 다음 문장을 위해 버퍼 초기화
                frames = []
except KeyboardInterrupt:
    print("\n녹음 종료.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
