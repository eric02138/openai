"""
Usage:
python metadata_filtering_openai_messages.py thread_123456 --api-key sk-yourapikeyhere
python metadata_filtering_openai_messages.py thread_123456 --metadata "user_email=emattison@something.com"
python metadata_filtering_openai_messages.py thread_123456 --limit 50 --output messages_filtered.json
"""

import requests
import json
import os
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Any

class OpenAIMessageClient:
    """Client for retrieving OpenAI messages based on metadata filters."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI Message API client.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_messages(self, 
                    thread_id: str,
                    metadata_filters: Optional[Dict[str, Any]] = None,
                    limit: int = 100,
                    order: str = "desc") -> List[Dict]:
        """
        Retrieve messages from a thread with optional metadata filtering.
        
        Args:
            thread_id: The ID of the thread to retrieve messages from
            metadata_filters: Dict of metadata key-value pairs to filter by
            limit: Maximum number of messages to return (1-100)
            order: Sort order - "asc" or "desc" (default) by creation time
            
        Returns:
            List of message objects matching the criteria
        """
        url = f"{self.base_url}/threads/{thread_id}/messages"
        
        params = {
            "limit": min(100, max(1, limit)),  # Ensure limit is between 1-100
            "order": order
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            messages = data.get("data", [])
            
            # Filter by metadata if specified
            if metadata_filters:
                filtered_messages = []
                for message in messages:
                    message_metadata = message.get("metadata", {})
                    if self._matches_metadata_filters(message_metadata, metadata_filters):
                        filtered_messages.append(message)
                return filtered_messages
            
            return messages
            
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving messages: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return []
    
    def _matches_metadata_filters(self, message_metadata: Dict, filters: Dict) -> bool:
        """
        Check if message metadata matches all specified filters.
        
        Args:
            message_metadata: Metadata dict from the message
            filters: Dict of metadata key-value pairs to match
            
        Returns:
            True if all filters match, False otherwise
        """
        for key, value in filters.items():
            if key not in message_metadata or message_metadata[key] != value:
                return False
        return True

    def export_messages_to_json(self, 
                               messages: List[Dict], 
                               output_file: str = None) -> str:
        """
        Export retrieved messages to a JSON file.
        
        Args:
            messages: List of message objects to export
            output_file: File path to save to. If None, generates a timestamped filename.
            
        Returns:
            Path to the saved file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"openai_messages_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(messages, f, indent=2)
        
        print(f"Exported {len(messages)} messages to {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Retrieve OpenAI messages based on metadata filters")
    parser.add_argument("thread_id", help="OpenAI thread ID to retrieve messages from")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--metadata", nargs="+", help="Metadata filters in key=value format")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of messages to retrieve")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--order", choices=["asc", "desc"], default="desc", help="Sort order")
    
    args = parser.parse_args()
    
    # Parse metadata filters
    metadata_filters = {}
    if args.metadata:
        for filter_str in args.metadata:
            if "=" in filter_str:
                key, value = filter_str.split("=", 1)
                metadata_filters[key.strip()] = value.strip()
    
    # Initialize client and get messages
    try:
        client = OpenAIMessageClient(args.api_key)
        messages = client.get_messages(
            thread_id=args.thread_id,
            metadata_filters=metadata_filters,
            limit=args.limit,
            order=args.order
        )
        
        if messages:
            client.export_messages_to_json(messages, args.output)
            print(f"Successfully retrieved {len(messages)} messages")
        else:
            print("No messages found matching the criteria")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
