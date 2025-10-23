


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


def fill_timeline_gaps(segments, gap_label='gap') -> list[tuple[int, int, str]]:
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

