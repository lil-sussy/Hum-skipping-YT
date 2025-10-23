import services.segment_audio_humaware as segment_audio_humaware
import argparse
import json
import os
from config import *
import services.timeline_utils as timeline_utils
    
def five_sampling_classifier(video_id, wav_path):
    hop_ratio = HOP_RATIO
    classifications = []
    for frame_sec in CONFIG_CHOSEN_FRAME_LENGTHS_SEC:
        hop_sec = frame_sec / hop_ratio
        args = argparse.Namespace()
        args.audio = wav_path
        args.model_path = CONFIG_MODEL_PATH
        args.model_sr = 16000
        args.frame_sec = frame_sec
        args.hop_sec = hop_sec
        classification_path = os.path.expanduser(CONFIG_CLASSIFICATIONS_CACHE + video_id + "/classification_"+str(int(frame_sec*100))+"ms.json")
        os.makedirs(classification_path, exist_ok=True)
        
        classification = segment_audio_humaware.execute(args)
        classifications.append(classification)
        
        # Cache individual classification for unimplemented cache retrieval
        with open(classification_path, "w", encoding="utf-8") as f:
            json.dump(classification, f, indent=2)
        print(f"[five_sampling_classifier] Wrote output to {classification_path}")
    
    return merge_classifications(classifications)


def merge_classifications(classifications, audio_duration):
    all_segments = []
    
    for classification in classifications:
        segments = classification.get("segments", [])
        for seg in segments:
            all_segments.append(seg)

    flattened_segments = []
    for segment in all_segments:
        # Convert to milliseconds and clip to audio duration
        start_ms = int(segment.get("start", 0) * 1000)
        end_ms = int(segment.get("end", 0) * 1000)
        label = segment.get("label", "unknown")

        start_ms = max(0, start_ms)
        end_ms = min(audio_duration, end_ms)

        if start_ms < end_ms: # Only add valid, non-empty segments
            flattened_segments.append((start_ms, end_ms, label))

    # Sort all segments by their start time
    flattened_segments.sort() # Sorts by first element (start_ms) by default
    
    final_timeline_with_gaps = timeline_utils.merge_time_segments(flattened_segments, ["silence", "hum"])
    final_timeline = timeline_utils.fill_timeline_gaps(final_timeline_with_gaps, "speech")
    return final_timeline
