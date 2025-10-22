import json
from pydub import AudioSegment

import argparse
import json
import math
import os
from typing import List, Dict, Tuple, Optional
from timeline_utils import merge_time_segments, fill_timeline_gaps
import numpy as np
import soundfile as sf

def reduce_file_with_classification_and_export(json_path: str, output_path: str):
    """
    Reads an audio file and a JSON classification, then creates a new audio file
    that skips segments labeled as 'silence' or 'hum'.

    Args:
        json_path (str): The path to the JSON classification file.
        output_path (str): The path to save the processed audio file.
    """
    try:
        with open(json_path, 'r') as f:
            classification_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    audio_file_name = classification_data.get("audio_file")
    if not audio_file_name:
        print("Error: 'audio_file' key not found in the JSON data.")
        return

    audio_file_path = "examples/"+audio_file_name
    try:
        # Load the entire audio file
        original_audio = AudioSegment.from_file(audio_file_path)
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return
    return reduce_file_with_classification_and_export(classification_data, output_path, original_audio)


def reduce_audio_with_classfication_and_export(time_segments: list[tuple[int, int, str]], output_path: str, original_audio):

    original_audio_duration_ms = len(original_audio)

    # List to store effective "keep" intervals (start_ms, end_ms)
    keep_intervals = []

    # Sort segments by start time for easier processing
    segments = [{"start": seg[0]/1000, "end": seg[1]/1000, "label": seg[2], } for seg in time_segments]

    # Process segments to determine what to keep
    # A simple approach: add all 'speech' intervals. Then, for 'hum'/'silence',
    # subtract them from any overlapping 'speech' intervals.
    # This can be tricky with complex overlaps.
    # A more robust approach: build a timeline of "active" labels.

    # Let's use a simpler, common approach for classification processing:
    # 1. Start with an assumption that everything is "not keep".
    # 2. Add all "speech" segments to a list of "to_keep" intervals.
    # 3. For "hum" and "silence" segments, we mark them as "to_remove".
    #    This needs a way to modify the "to_keep" intervals.

    # A good strategy is to use a list of non-overlapping "keep" intervals.
    # For each segment:
    # If it's "speech", add it to a temporary list.
    # If it's "hum" or "silence", add it to a "remove" list.

    # This can be handled more elegantly by building a "merged" list of
    # segments where 'hum' and 'silence' take precedence for removal.
    # For now, let's assume "speech" segments are always added, and we want
    # to *cut out* any `hum` or `silence` that falls within *any* current segment.

    # Let's iterate through the segments and build a new list of intervals
    # that should be *included* in the final audio.
    # The simplest way to handle overlap is to process in order of priority:
    # `speech` first, then remove `hum` and `silence`.

    # Create a boolean array/list representing the audio timeline,
    # True means 'keep', False means 'remove'
    # Initialize all to True (assume keep unless specified to remove)
    # This might be memory intensive for very long audio files.
    # A better way is to manage intervals.

    # Let's try to maintain a list of (start_ms, end_ms) tuples for segments to keep.
    # Initialize it as empty.
    # For each 'speech' segment, add it.
    # For each 'hum'/'silence' segment, we need to subtract it from existing
    # 'keep' segments. This requires interval arithmetic.

    # Simpler approach: Create a list of all time points where a segment
    # starts or ends. Then iterate through these intervals.

    # Collect all unique event points (start/end of segments)
    event_points = set()
    for seg in segments:
        event_points.add(int(seg["start"] * 1000))
        event_points.add(int(seg["end"] * 1000))
    event_points = sorted(list(event_points))

    # Determine the label for each sub-interval
    # We will prioritize "hum" and "silence" for removal
    # If a point has both speech and hum, hum should win if we're "skipping hum".
    #
    # The most robust way for priorities:
    # 1. Initialize an empty timeline (e.g., list of (start, end, label)).
    # 2. Insert all segments, giving priority to "hum" and "silence" when overlapping.
    #    This means if a "speech" segment exists, and a "hum" segment overlaps,
    #    the overlapping part of the "speech" should be re-labeled as "hum".

    # Let's build a list of (start_ms, end_ms, label) for a consolidated timeline
    timeline = []

    # Sort segments by start time
    segments.sort(key=lambda s: s["start"])

    # This loop builds a timeline, resolving overlaps.
    # Priority: hum/silence > speech
    for segment in segments:
        seg_start_ms = int(segment["start"] * 1000)
        seg_end_ms = int(segment["end"] * 1000)
        seg_label = segment["label"]

        if not timeline:
            timeline.append([seg_start_ms, seg_end_ms, seg_label])
            continue

        new_timeline = []
        added_current_segment = False

        for i, (tl_start, tl_end, tl_label) in enumerate(timeline):
            # No overlap, or current segment is after timeline segment
            if seg_start_ms >= tl_end:
                new_timeline.append([tl_start, tl_end, tl_label])
            # No overlap, or current segment is before timeline segment
            elif seg_end_ms <= tl_start:
                if not added_current_segment:
                    new_timeline.append([seg_start_ms, seg_end_ms, seg_label])
                    added_current_segment = True
                new_timeline.append([tl_start, tl_end, tl_label])
            # There is an overlap or one contains the other
            else:
                # Calculate the parts before, during, and after the overlap
                # Part before current segment
                if tl_start < seg_start_ms:
                    new_timeline.append([tl_start, seg_start_ms, tl_label])

                # Overlapping part
                overlap_start = max(tl_start, seg_start_ms)
                overlap_end = min(tl_end, seg_end_ms)

                if overlap_start < overlap_end:
                    # Decide which label wins: "hum"/"silence" always win over "speech" for removal
                    winning_label = seg_label
                    if tl_label in ["hum", "silence"] and seg_label not in ["hum", "silence"]:
                         winning_label = tl_label # existing remove segment takes precedence
                    elif seg_label in ["hum", "silence"] and tl_label not in ["hum", "silence"]:
                         winning_label = seg_label # new remove segment takes precedence
                    # If both are keep/remove, the new one replaces
                    elif tl_label not in ["hum", "silence"] and seg_label not in ["hum", "silence"]:
                         winning_label = seg_label # both speech, newer one can override or merge
                    elif tl_label in ["hum", "silence"] and seg_label in ["hum", "silence"]:
                         winning_label = seg_label # both remove, newer one can override or merge

                    # This is the tricky part with `pydub +=`.
                    # For *this* approach, we are consolidating the timeline first.
                    new_timeline.append([overlap_start, overlap_end, winning_label])
                    added_current_segment = True # Mark that this segment has been handled in overlap

                # Part after current segment
                if tl_end > seg_end_ms:
                    new_timeline.append([seg_end_ms, tl_end, tl_label])

        # If the current segment was not added (e.g., completely after all existing segments)
        if not added_current_segment:
            if not new_timeline or seg_start_ms >= new_timeline[-1][1]:
                new_timeline.append([seg_start_ms, seg_end_ms, seg_label])
            else: # Insert in correct sorted position if it belongs in the middle
                inserted = False
                for i in range(len(new_timeline)):
                    if seg_end_ms <= new_timeline[i][0]:
                        new_timeline.insert(i, [seg_start_ms, seg_end_ms, seg_label])
                        inserted = True
                        break
                if not inserted:
                    new_timeline.append([seg_start_ms, seg_end_ms, seg_label])


        # Merge adjacent segments with the same label (optional, but good for cleanup)
        merged_timeline = []
        if new_timeline:
            new_timeline.sort(key=lambda x: x[0]) # Ensure sorted by start time
            merged_timeline.append(new_timeline[0])
            for j in range(1, len(new_timeline)):
                current_tl_seg = new_timeline[j]
                last_merged_seg = merged_timeline[-1]

                # If current segment is adjacent or overlaps and has same label
                if current_tl_seg[2] == last_merged_seg[2] and current_tl_seg[0] <= last_merged_seg[1]:
                    merged_timeline[-1][1] = max(last_merged_seg[1], current_tl_seg[1])
                else:
                    merged_timeline.append(current_tl_seg)
        timeline = merged_timeline

    # Now, `timeline` contains a consolidated list of non-overlapping segments with their effective labels.
    # Filter out the "hum" and "silence" segments to get the final "keep" intervals.
    final_keep_intervals = []
    for tl_start, tl_end, tl_label in timeline:
        if tl_label not in ["hum", "silence"]:
            final_keep_intervals.append((tl_start, tl_end))

    # Concatenate the final "keep" intervals
    processed_audio = AudioSegment.empty()
    for start_ms, end_ms in final_keep_intervals:
        if start_ms < end_ms: # Ensure valid interval
            # Ensure the segment is within the bounds of the original audio
            actual_start_ms = max(0, start_ms)
            actual_end_ms = min(original_audio_duration_ms, end_ms)

            if actual_start_ms < actual_end_ms:
                segment_audio = original_audio[actual_start_ms:actual_end_ms]
                processed_audio += segment_audio
                print(f"Including segment from {actual_start_ms/1000:.2f}s "
                      f"to {actual_end_ms/1000:.2f}s (label: Keep)")
            else:
                print(f"Warning: Calculated 'keep' interval "
                      f"({start_ms/1000:.2f}s - {end_ms/1000:.2f}s) "
                      f"resulted in an invalid or empty slice.")

    if processed_audio.duration_seconds > 0:
        try:
            processed_audio.export(output_path, format="wav")
            print(f"\nProcessed audio saved to {output_path}")
            print(f"Original audio duration: {original_audio_duration_ms/1000:.2f} seconds")
            print(f"Processed audio duration: {processed_audio.duration_seconds:.2f} seconds")
        except Exception as e:
            print(f"Error exporting processed audio to {output_path}: {e}")
    else:
        print("\nNo valid segments found to include in the output audio.")




def reduce_audio_with_multiple_classfications_and_export(classifications: list[dict], input_path:str, output_path: str):
    """
    Reads an audio file and multiple JSON classifications, then creates a new
    audio file by selectively including segments not classified as 'hum' or 'silence'
    in any of the input classifications, handling overlaps.

    Args:
        json_paths (list[str]): A list of paths to the JSON classification files.
        output_path (str): The path to save the processed audio file.
    """
    all_segments = []
    
    for classification in classifications:
        segments = classification.get("segments", [])
        for seg in segments:
            # Add a source indicator if useful for debugging
            # seg['source_json'] = json_path
            all_segments.append(seg)

    if not os.path.exists(input_path):
        print(f"Error: Audio file '{input_path}' not found.")
        return

    try:
        original_audio = AudioSegment.from_file(input_path)
    except Exception as e:
        print(f"Error loading audio file {input_path}: {e}")
        return
    original_audio_duration_ms = len(original_audio)
    
    flattened_segments = []
    for segment in all_segments:
        # for seg in segments_list:
            # Convert to milliseconds and clip to audio duration
            start_ms = int(segment.get("start", 0) * 1000)
            end_ms = int(segment.get("end", 0) * 1000)
            label = segment.get("label", "unknown")

            start_ms = max(0, start_ms)
            end_ms = min(original_audio_duration_ms, end_ms)

            if start_ms < end_ms: # Only add valid, non-empty segments
                flattened_segments.append((start_ms, end_ms, label))

    # Sort all segments by their start time
    flattened_segments.sort() # Sorts by first element (start_ms) by default
    
    final_timeline_with_gaps = merge_time_segments(flattened_segments, ["silence", "hum"])
    final_timeline = fill_timeline_gaps(final_timeline_with_gaps, "speech")
    print("[reduce_audio_with_multiple_classfications_and_export]: final timeline constituted, proceeding to producing audio")
    return reduce_audio_with_classfication_and_export(final_timeline, output_path, original_audio)


    # # 2. Build a consolidated timeline with priority for 'hum' and 'silence'
    # # Sort all segments by start time
    # all_segments.sort(key=lambda s: s.get("start", 0))

    # # This timeline will store (start_ms, end_ms, label) tuples.
    # # It will be kept sorted and non-overlapping.
    # timeline = []

    # for segment in all_segments:
    #     seg_start_ms = int(segment.get("start", 0) * 1000)
    #     seg_end_ms = int(segment.get("end", 0) * 1000)
    #     seg_label = segment.get("label", "unknown")

    #     # Ensure segment is valid and within original audio bounds
    #     seg_start_ms = max(0, seg_start_ms)
    #     seg_end_ms = min(original_audio_duration_ms, seg_end_ms)

    #     if seg_start_ms >= seg_end_ms:
    #         # print(f"Skipping invalid segment: {segment}") # Debugging
    #         continue

    #     # Add the current segment to the timeline, resolving overlaps
    #     new_timeline_entries = []
    #     current_segment_parts = [(seg_start_ms, seg_end_ms, seg_label)]

    #     # Iterate through existing timeline entries and resolve overlaps
    #     for tl_start, tl_end, tl_label in timeline:
    #         next_current_segment_parts = []
    #         for current_part_start, current_part_end, current_part_label in current_segment_parts:
    #             # No overlap with existing timeline entry
    #             if current_part_end <= tl_start or current_part_start >= tl_end:
    #                 next_current_segment_parts.append((current_part_start, current_part_end, current_part_label))
    #                 continue

    #             # Overlap exists: split current_part into up to 3 pieces
    #             # 1. Part before existing timeline entry (no overlap)
    #             if current_part_start < tl_start:
    #                 next_current_segment_parts.append((current_part_start, tl_start, current_part_label))

    #             # 2. Overlapping part
    #             overlap_start = max(current_part_start, tl_start)
    #             overlap_end = min(current_part_end, tl_end)

    #             if overlap_start < overlap_end:
    #                 # Priority for removal: 'hum'/'silence' override other labels
    #                 winning_label = current_part_label # Default: new segment label wins
    #                 if tl_label in ["hum", "silence"] and current_part_label not in ["hum", "silence"]:
    #                     winning_label = tl_label # Existing 'hum'/'silence' takes precedence
    #                 elif current_part_label in ["hum", "silence"]:
    #                     winning_label = current_part_label # New 'hum'/'silence' takes precedence

    #                 new_timeline_entries.append((overlap_start, overlap_end, winning_label))

    #             # 3. Part after existing timeline entry (no overlap)
    #             if current_part_end > tl_end:
    #                 next_current_segment_parts.append((tl_end, current_part_end, current_part_label))

    #         current_segment_parts = next_current_segment_parts # Update parts for next iteration

    #     # Add any remaining parts of the current segment (those that didn't overlap or were fully outside)
    #     new_timeline_entries.extend(current_segment_parts)

    #     # Update and sort the main timeline. Merge adjacent/overlapping entries with same label.
    #     timeline.extend(new_timeline_entries)
    #     timeline.sort() # Sort by start time for merging

    #     merged_timeline = []
    #     if timeline:
    #         current_merged = list(timeline[0]) # Copy the first element
    #         for i in range(1, len(timeline)):
    #             tl_start, tl_end, tl_label = timeline[i]
    #             # If current segment is adjacent or overlaps and has the same label
    #             if tl_label == current_merged[2] and tl_start <= current_merged[1]:
    #                 current_merged[1] = max(current_merged[1], tl_end) # Extend the end
    #             else:
    #                 merged_timeline.append(tuple(current_merged))
    #                 current_merged = [tl_start, tl_end, tl_label]
    #         merged_timeline.append(tuple(current_merged))
    #     timeline = merged_timeline


    # # 3. Filter out 'hum' and 'silence' segments from the consolidated timeline
    # final_keep_intervals = []
    # for start_ms, end_ms, label in timeline:
    #     if label not in ["hum", "silence"]:
    #         final_keep_intervals.append((start_ms, end_ms))
    #         # print(f"KEEP: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s ({label})") # Debugging
    #     # else:
    #         # print(f"SKIP: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s ({label})") # Debugging

    # # 4. Concatenate the final "keep" intervals to form the processed audio
    # processed_audio = AudioSegment.empty()
    # for start_ms, end_ms in final_keep_intervals:
    #     if start_ms < end_ms:
    #         segment_audio = original_audio[start_ms:end_ms]
    #         processed_audio += segment_audio
    #         # print(f"Appended audio: {start_ms/1000:.2f}s to {end_ms/1000:.2f}s") # Debugging
    #     # else:
    #         # print(f"Skipped zero-length or invalid interval: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s") # Debugging


    # # 5. Export the processed audio
    # if processed_audio.duration_seconds > 0:
    #     try:
    #         processed_audio.export(output_path, format="wav")
    #         print(f"\nProcessed audio saved to {output_path}")
    #         print(f"Original audio duration: {original_audio_duration_ms/1000:.2f} seconds")
    #         print(f"Processed audio duration: {processed_audio.duration_seconds:.2f} seconds")
    #     except Exception as e:
    #         print(f"Error exporting processed audio to {output_path}: {e}")
    # else:
    #     print("\nNo valid segments found to include in the output audio.")