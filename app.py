from flask import Flask, request, jsonify
import whisperx
import gc
import os
import time

app = Flask(__name__)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Extract audio file path from the POST request data
        file = request.files['file']
        temp_file = f"C:/Users/paperspace/AppData/Local/Programs/Python/Python310/Lib/site-packages/whisperx/{file.filename}"
        file.save(temp_file)
        
        audio_file = "test.mp3"
        
        time.sleep(5)

        # Load audio data
        audio = whisperx.load_audio(audio_file)
        
        # Initialize the diarization model with your Hugging Face authentication token
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device="cuda")
        
        # Load the Whisper model
        model = whisperx.load_model("large-v2", "cuda", compute_type="float16")
        
        # Transcribe audio
        result = model.transcribe(audio, batch_size=16)
        
        # Load alignment model based on the detected language
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        
        # Align the transcriptions
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)
        
        # Perform diarization
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Process results into formatted text
        formatted_text = format_transcription(result)
        
        # Output formatted text (consider saving to file or another output method)
        for line in formatted_text:
            print(line)

        # Return success message
        return jsonify({"message": "Processing complete", "transcription": formatted_text})
    
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)}), 500

def format_transcription(transcript_data):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f'{hours:02d}:{minutes:02d}:{seconds:06.3f}'
    
    def find_speaker(segment):
        if 'speaker' in segment:
            return segment['speaker']
        if 'words' in segment and len(segment['words']) > 0 and 'speaker' in segment['words'][0]:
            return segment['words'][0]['speaker']
        return 'UNKNOWN'
    
    formatted_lines = []
    for segment in transcript_data['segments']:
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        speaker = find_speaker(segment)
        text = segment['text'].strip()
        formatted_line = f'[{start_time}-{end_time}] {speaker}: {text}'
        formatted_lines.append(formatted_line)
    
    return formatted_lines

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')