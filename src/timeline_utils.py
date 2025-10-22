


def merge_time_segments(segments: list[tuple[int, int, str]], include_classes=None) -> list[tuple[int, int, str]]:
    """
    Merge overlapping time segments while preserving labels.
    Identical labels are kept singular, different labels are comma-separated.
    
    Args:
        segments: List of tuples (start_ms, end_ms, label)
                 Must be sorted by start_ms
        
    Returns:
        List of merged tuples (start_ms, end_ms, label)
    """
    if not segments:
        return []
    
    # Convert include_classes to set if provided
    include_classes = set(include_classes) if include_classes else None
    
    merged = []
    
    for segment in segments:
        current_start, current_end, current_label = segment
        
        # Skip if class filtering enabled and label not included
        if include_classes and current_label not in include_classes:
            continue
            
        # Find overlapping segment in merged list
        overlap_found = False
        
        for i, merged_seg in enumerate(merged):
            last_start, last_end, _ = merged_seg
            
            if current_start <= last_end:
                # Update end time if current segment extends further
                merged[i][1] = max(last_end, current_end)
                
                # Handle labels
                if current_label == merged_seg[2]:
                    # Same label, keep singular
                    merged[i][2] = current_label
                else:
                    # Different labels, create comma-separated string
                    merged_labels = [label.strip() for label in merged_seg[2].split(',')]
                    merged_labels.append(current_label)
                    # Remove duplicates while maintaining order
                    seen = {}
                    final_labels = [seen.setdefault(label, label) 
                                 for label in merged_labels 
                                 if label not in seen]
                    merged[i][2] = ','.join(final_labels)
                
                overlap_found = True
                break
        
        if not overlap_found:
            # No overlap found, add new segment
            merged.append(list(segment))
    
    xd = [tuple(segment) for segment in merged]
    return xd


def fill_timeline_gaps(segments, gap_label='gap'):
    """
    Fill gaps in a timeline with a specified label.
    
    Args:
        segments: List of tuples (start_ms, end_ms, label)
                 Must be sorted by start_ms
        gap_label: Label to use for filling gaps
        
    Returns:
        List of tuples (start_ms, end_ms, label) representing filled timeline
    """
    if not segments:
        return []
    
    # Sort segments by start time if not already sorted
    segments = sorted(segments, key=lambda x: x[0])
    
    filled_timeline = []
    current_end = segments[0][0]
    
    # Add gap segment if there's a gap before first segment
    if current_end > 0:
        filled_timeline.append((0, current_end, gap_label))
    
    for i, segment in enumerate(segments):
        start_ms, end_ms, label = segment
        
        # Check for gap between current and previous segment
        if start_ms > current_end:
            # Add gap segment
            filled_timeline.append((current_end, start_ms, gap_label))
        
        # Add main segment
        filled_timeline.append(segment)
        current_end = end_ms
    
    return filled_timeline


def _insert_and_merge_segment(
    current_timeline: list[tuple[int, int, str]],
    all_raw_segments: list[list[dict]],
    total_audio_duration_ms: int,
    new_segment: tuple[int, int, str],
    priority_labels_for_removal: list[str]
) -> list[tuple[int, int, str]]:
    """
    Inserts a new segment into an existing sorted, non-overlapping timeline,
    resolving overlaps based on priority_labels_for_removal, and then re-merges
    adjacent segments with the same label.
    """
    new_timeline_raw = [] # Temporarily hold all pieces
    inserted_new_segment = False

    ns_start, ns_end, ns_label = new_segment

    for i, (ts_start, ts_end, ts_label) in enumerate(current_timeline):
        if not inserted_new_segment and ns_end <= ts_start:
            # New segment entirely precedes this timeline segment
            new_timeline_raw.append(new_segment)
            inserted_new_segment = True

        # No overlap with current timeline segment
        if ns_end <= ts_start or ns_start >= ts_end:
            new_timeline_raw.append((ts_start, ts_end, ts_label))
            continue

        # Overlap exists
        # 1. Part of timeline segment before new segment
        if ts_start < ns_start:
            new_timeline_raw.append((ts_start, ns_start, ts_label))

        # 2. Overlapping part
        overlap_start = max(ts_start, ns_start)
        overlap_end = min(ts_end, ns_end)

        if overlap_start < overlap_end:
            winning_label = ns_label # New segment's label wins by default
            if ts_label in priority_labels_for_removal and ns_label not in priority_labels_for_removal:
                winning_label = ts_label # Existing removal label wins
            elif ns_label in priority_labels_for_removal:
                winning_label = ns_label # New removal label wins

            new_timeline_raw.append((overlap_start, overlap_end, winning_label))
            # Mark the new segment as "processed" for this overlapping interval
            # This is complex when handling `ns_start` and `ns_end` directly.
            # A simpler way is to just let the new_segment itself be processed.

            # We need to ensure that the *original* new_segment's non-overlapping parts are also added
            # This logic is better handled by a 'sweep line' or by reconstructing.
            # Let's simplify and just add all segments and then merge.
            # This current `_insert_and_merge_segment` is attempting to do
            # too much for a single insert, it's better for a full rebuild.
            # Reverting to the simpler "append and re-merge all" logic for clarity and correctness.

    # --- Re-thinking timeline generation for robustness ---
    # The `_insert_and_merge_segment` logic is inherently complex for maintaining a *single* sorted,
    # non-overlapping timeline with priorities on the fly.
    # The most reliable way for arbitrary overlaps and priorities is to build it from event points.

    # Let's use the event-point-based approach, which is more robust for these scenarios.
    # This involves building a list of all start/end points, and for each interval
    # between these points, determining the active label.

    # Collect all unique event points (start/end of all segments)
    all_points = set()
    for segment in all_raw_segments:
        # for seg in segments_list:
            start_ms = int(segment.get("start", 0) * 1000)
            end_ms = int(segment.get("end", 0) * 1000)
            start_ms = max(0, start_ms)
            end_ms = min(total_audio_duration_ms, end_ms)
            if start_ms < end_ms:
                all_points.add(start_ms)
                all_points.add(end_ms)

    event_points = sorted(list(all_points))

    final_timeline = []

    # Iterate through each sub-interval defined by the event points
    for i in range(len(event_points) - 1):
        interval_start_ms = event_points[i]
        interval_end_ms = event_points[i+1]

        if interval_start_ms >= interval_end_ms:
            continue

        # Determine the label for this interval based on all segments
        active_label = "unknown" # Default to 'unknown' or 'speech'

        # Find all segments that cover this interval
        covering_segments = []
        for segment in all_raw_segments:
            # for seg in segments_list:
                seg_start_ms = int(segment.get("start", 0) * 1000)
                seg_end_ms = int(segment.get("end", 0) * 1000)
                seg_label = segment.get("label", "unknown")

                if seg_start_ms <= interval_start_ms and seg_end_ms >= interval_end_ms:
                    covering_segments.append((seg_label, segment.get("score", 0.0))) # Store label and score

        # Determine the winning label for this interval based on priority
        # Priority: (highest priority label in `priority_labels_for_removal`) > other `priority_labels_for_removal` > "other" labels
        current_winning_label = None
        has_removal_label = False

        # First, check if any removal label covers this interval
        for label, _ in covering_segments:
            if label in priority_labels_for_removal:
                current_winning_label = label
                has_removal_label = True
                break # A removal label exists, it wins

        if not has_removal_label:
            # If no removal label covers, then it's a "keep" interval
            # We can arbitrarily pick the first 'speech' or 'unknown' label
            # or just default to 'speech' if multiple are present.
            # For this context, if it's not removed, we can just call it 'speech'.
            # A more sophisticated system might aggregate scores or have more complex rules.
            active_label = "speech"
            for label, _ in covering_segments:
                 if label not in priority_labels_for_removal:
                     active_label = label # Take the first non-removal label found

        else:
            active_label = current_winning_label # The winning removal label

        if active_label:
            # Merge with previous if label is the same and intervals are adjacent
            if final_timeline and final_timeline[-1][2] == active_label and \
               final_timeline[-1][1] == interval_start_ms:
                # Extend the last timeline entry
                final_timeline[-1] = (final_timeline[-1][0], interval_end_ms, active_label)
            else:
                final_timeline.append((interval_start_ms, interval_end_ms, active_label))

    # Add a final "unknown" or "gap" segment if the audio duration is not fully covered
    # (e.g., if there were no segments for the very end)
    if total_audio_duration_ms > 0 and (not final_timeline or final_timeline[-1][1] < total_audio_duration_ms):
        last_end_ms = final_timeline[-1][1] if final_timeline else 0
        if last_end_ms < total_audio_duration_ms:
            # If there are no segments, or segments don't cover the full length
            # The remaining part can be considered "unknown" or "gap"
            final_timeline.append((last_end_ms, total_audio_duration_ms, "unknown_gap"))

    # Filter out empty intervals that might have resulted from clipping
    final_timeline = [(s, e, l) for s, e, l in final_timeline if s < e]

    return final_timeline