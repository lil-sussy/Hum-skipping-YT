import server.services.humaware_classifictation as humaware_classifictation
import os

# Initialize classifier
classifier = humaware_classifictation.AudioClassifier()

# Process audio file
input = os.path.abspath('../examples/conference_sample/audio_sample_16bit48kHz.wav')
results = classifier.process_audio_file(input)

# Save results to JSON
outputjson = os.path.abspath('../examples/conference_sample/output.json')
classifier.save_json(results, outputjson)
