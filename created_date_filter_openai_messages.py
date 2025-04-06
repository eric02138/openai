#!/usr/bin/env python3

# Usage:
# Using ISO 8601 timestamps
# python created_date_filter_openai_messages.py --start "2023-04-01T00:00:00Z" --end "2023-04-30T23:59:59Z"
# Using Unix timestamps
# python created_date_filter_openai_messages.py --start 1680307200 --end 1682899199
# Reading from a local JSON file
# python created_date_filter_openai_messages.py --start "2023-04-01T00:00:00Z" --end "2023-04-30T23:59:59Z" --input messages.json
# Saving output to a file
# python created_date_filter_openai_messages.py --start "2023-04-01T00:00:00Z" --end "2023-04-30T23:59:59Z" --output filtered_messages.json

import argparse
import json
import sys
from datetime import datetime
import os
import requests
from typing import List, Dict, Any, Optional

def parse_timestamp(timestamp_str: str) -> int:
    """
    Parse a timestamp string into a Unix timestamp (seconds since epoch).
    Accepts ISO 8601 format (e.g., '2023-04-01T12:00:00Z') or Unix timestamp as string.
    """
    try:
        # Try parsing as Unix timestamp (integer)
        return int(timestamp_str)
    except ValueError:
        # Try parsing as ISO 8601 date string
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp())

def filter_messages(
    messages: List[Dict[str, Any]], 
    start_timestamp: int, 
    end_timestamp: int
) -> List[Dict[str, Any]]:
    """
    Filter messages based on 'created_at' timestamp.
    Returns messages where start_timestamp <= created_at <= end_timestamp.
    """
    filtered = []
    
    for message in messages:
        created_at = message.get("created_at")
        if created_at is None:
            continue
            
        if start_timestamp <= created_at <= end_timestamp:
            filtered.append(message)
            
    return filtered

def get_messages_from_api(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch messages from OpenAI API.
    Returns a list of message objects.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Note: This URL should be updated based on the actual OpenAI messages endpoint
    response = requests.get("https://api.openai.com/v1/messages", headers=headers)
    response.raise_for_status()
    
    return response.json().get("data", [])

def main():
    parser = argparse.ArgumentParser(description="Filter OpenAI messages by timestamp range")
    parser.add_argument("--start", required=True, help="Start timestamp (ISO format or Unix timestamp)")
    parser.add_argument("--end", required=True, help="End timestamp (ISO format or Unix timestamp)")
    parser.add_argument("--input", help="Input JSON file containing messages (optional)")
    parser.add_argument("--output", help="Output JSON file (optional, default: stdout)")
    parser.add_argument("--api-key", help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Parse timestamps
    start_timestamp = parse_timestamp(args.start)
    end_timestamp = parse_timestamp(args.end)
    
    # Get messages from file or API
    if args.input:
        with open(args.input, 'r') as f:
            messages = json.load(f)
            # Handle both direct list and {"data": [...]} format
            if isinstance(messages, dict) and "data" in messages:
                messages = messages["data"]
    else:
        # Fetch from API
        try:
            messages = get_messages_from_api(args.api_key)
        except Exception as e:
            print(f"Error fetching messages from API: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Filter messages
    filtered_messages = filter_messages(messages, start_timestamp, end_timestamp)
    
    # Output results
    output_data = {"data": filtered_messages, "count": len(filtered_messages)}
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Filtered {len(filtered_messages)} messages written to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()