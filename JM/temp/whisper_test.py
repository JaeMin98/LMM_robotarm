import whisper
 
model = whisper.load_model("large")
result = model.transcribe("korean-sample.mp3",fp16=False)
text=result["text"]

print(text)