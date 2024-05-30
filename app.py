from flask import Flask, request, jsonify
import os
import assemblyai as aai
import logging
from flask_cors import CORS
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.DEBUG)

#api key assembleyai
aai.settings.api_key = "907fa77d58f24551aeaad473b3164a5e"

# Load the BERT model and tokenizer
model_path = '/home/prax/Desktop/finalYearProject/bert.hdfs-20240429T115419Z-001/bert.hdfs'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to process video audio and transcribe to text using AssemblyAI
def process_video_and_compute_labels(video_file):
    try:
        # Ensure 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        video_path = os.path.join("uploads", video_file.filename)
        video_file.save(video_path)
        
        # Initialize the transcriber
        transcriber = aai.Transcriber()

        # Transcribe the audio file
        transcript = transcriber.transcribe(video_path)


        # Predict using the text classification model
        input_text = transcript.text
        predicted_labels = predict_labels_from_text(input_text)

        return input_text, predicted_labels

    except Exception as e:
        print(f"Error processing video audio: {e}")
        return None, None

# Function to predict labels based on input text using BERT model
def predict_labels_from_text(input_text):
    user_input = [input_text]
    user_encodings = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")
    user_dataset = TensorDataset(user_encodings['input_ids'], user_encodings['attention_mask'])
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)

    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Filter labels without a predicted value of 1
    result = [label for i, label in enumerate(labels) if predicted_labels[0][i] == 1]
    return result

# Combined endpoint to process video, transcribe audio, and predict labels
@app.route('/process_text_from_video', methods=['POST'])
def process_text_from_video():
    # Check if the 'video' file is present in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'})

    video_file = request.files['video']

    # Check if the file name is empty
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'})

    # Process the video to transcribe audio to text and predict labels
    input_text, predicted_labels = process_video_and_compute_labels(video_file)

    if input_text is None or predicted_labels is None:
        return jsonify({'error': 'Failed to process video'})

    # Return the transcribed text and predicted labels as response
    return jsonify({'transcription': input_text, 'predicted_labels': predicted_labels})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
