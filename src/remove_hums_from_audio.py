import generate_segment
import reduce_audio_test
import argparse
import json
import os
from reduce_audio_test import reduce_audio_with_multiple_classfications_and_export


def main():
    p = argparse.ArgumentParser(description="Process audio file and output labeled timestamps as JSON.")
    p.add_argument("--audio", required=True, help="Input audio file (wav, flac, mp3, ...)")
    p.add_argument("--model_path", default=None, help="Path to TorchScript model or HF model repo id")
    p.add_argument("--out", default="cringeOUT.wav", help="Output audio file")
    p.add_argument("--model_sr", type=int, default=16000, help="Target sample rate for model")
    p.add_argument("--merge_tol", type=float, default=0.15, help="Merge tolerance in seconds")
    args = p.parse_args()
    
    args.frame_sec = 0.5
    args.hop_sec = 0.25
    half_sec = generate_segment.execute(args)
    args.frame_sec = 1
    args.hop_sec = 0.3
    sec = generate_segment.execute(args)
    args.frame_sec = 2
    args.hop_sec = 0.5
    two_sec = generate_segment.execute(args)
    args.frame_sec = 4
    args.hop_sec = 1
    four_sec = generate_segment.execute(args)
    
    reduce_audio_with_multiple_classfications_and_export([half_sec, sec, two_sec, four_sec], input_path=args.audio, output_path=args.out)

    
def main_save_cache():
    p = argparse.ArgumentParser(description="Process audio file and output labeled timestamps as JSON.")
    p.add_argument("--audio", required=True, help="Input audio file (wav, flac, mp3, ...)")
    p.add_argument("--example", required=True, help="Parent folder of the example")
    p.add_argument("--model_path", default=None, help="Path to TorchScript model or HF model repo id")
    p.add_argument("--out", default="cringeOUT.wav", help="Output audio file")
    p.add_argument("--model_sr", type=int, default=16000, help="Target sample rate for model")
    p.add_argument("--merge_tol", type=float, default=0.15, help="Merge tolerance in seconds")
    args = p.parse_args()
    
    hop_ratio = 15
    args.frame_sec = 0.3
    args.hop_sec = args.frame_sec / hop_ratio
    third_sec = generate_segment.execute(args)
    third_sec_path = args.example + "third_sec_classification.json"
    with open(third_sec_path, "w", encoding="utf-8") as f:
        json.dump(third_sec, f, indent=2)
    print(f"[main] Wrote output to {third_sec_path}")
    args.frame_sec = 0.5
    args.hop_sec = args.frame_sec / hop_ratio
    half_sec = generate_segment.execute(args)
    half_sec_path = args.example + "half_sec_classification.json"
    with open(half_sec_path, "w", encoding="utf-8") as f:
        json.dump(half_sec, f, indent=2)
    print(f"[main] Wrote output to {half_sec_path}")
    args.frame_sec = 1
    args.hop_sec = args.frame_sec / hop_ratio
    sec = generate_segment.execute(args)
    sec_path = args.example + "sec_classification.json"
    with open(sec_path, "w", encoding="utf-8") as f:
        json.dump(sec, f, indent=2)
    print(f"[main] Wrote output to {sec_path}")
    args.frame_sec = 2
    args.hop_sec = args.frame_sec / hop_ratio
    two_sec = generate_segment.execute(args)
    two_sec_path = args.example + "two_sec_classification.json"
    with open(two_sec_path, "w", encoding="utf-8") as f:
        json.dump(two_sec, f, indent=2)
    print(f"[main] Wrote output to {two_sec_path}")
    args.frame_sec = 4
    args.hop_sec = args.frame_sec / hop_ratio
    four_sec = generate_segment.execute(args)
    four_sec_path = args.example + "four_sec_classification.json"
    with open(four_sec_path, "w", encoding="utf-8") as f:
        json.dump(four_sec, f, indent=2)
    print(f"[main] Wrote output to {four_sec_path}")
    

def main_from_cache():
    p = argparse.ArgumentParser(description="Process audio file and output labeled timestamps as JSON.")
    p.add_argument("--audio", required=True, help="Input audio file (wav, flac, mp3, ...)")
    p.add_argument("--example", required=True, help="Parent folder of the example")
    p.add_argument("--model_path", default=None, help="Path to TorchScript model or HF model repo id")
    p.add_argument("--out", default="cringeOUT.wav", help="Output audio file")
    p.add_argument("--model_sr", type=int, default=16000, help="Target sample rate for model")
    p.add_argument("--merge_tol", type=float, default=0.15, help="Merge tolerance in seconds")
    args = p.parse_args()
    
    classifications = {
    }
    for path in ["third_sec", "half_sec", "sec", "two_sec", "four_sec"]:
        classification_path = args.example + path + "_classification.json"
        if not os.path.exists(classification_path):
            print(f"Error: Json file '{classification_path}' not found.")
            return

        try:
            jsonfile = open(classification_path)
            classifications[path] = json.load(jsonfile)
        except Exception as e:
            print(f"Error loading json file {classification_path}: {e}")
            return
            
    reduce_audio_with_multiple_classfications_and_export([classifications["half_sec"], classifications["sec"], classifications["two_sec"], classifications["four_sec"]], input_path=args.audio, output_path=args.out)



if __name__ == "__main__":
    main_from_cache()