import server.services.humaware_classifictation as humaware_classifictation


# Initialize classifier
classifier = humaware_classifictation.AudioClassifier()

# Process audio file
results = classifier.process_audio_file('examples/conference_sample/audio_sample16bit48kHz.wav')

# Save results to JSON
classifier.save_json(results, 'examples/conference_sample/output.json')