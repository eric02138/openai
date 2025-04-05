import os
import json
import argparse
import base64
from openai import OpenAI

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, api_key=None):
    """
    Extract text from an image using OpenAI's Vision capabilities.
    Then process and structure it into key-value pairs as JSON.
    """
    # Set up the OpenAI client
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Encode the image
    base64_image = encode_image_to_base64(image_path)

    #print(base64_image)
    
    # First: Extract text from the image
    text_extraction_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    extracted_text = text_extraction_response.choices[0].message.content
    result = extracted_text
    # Second: Process the extracted text into key-value pairs
    # kv_response = client.chat.completions.create(
    #     model="gpt-4-turbo",
    #     messages=[
    #         {
    #             "role": "system", 
    #             "content": "You are a data extraction assistant. Convert the provided text into key-value pairs. "
    #                        "Identify labels, headers, or field names as keys, and their corresponding values. "
    #                        "Return only a JSON object with these key-value pairs. "
    #                        "If there is data that is not identifiable as key-value pairs, add this data to the JSON object as a list of 'additional_text' items. "
    #         },
    #         {
    #             "role": "user",
    #             "content": f"Convert this extracted text into key-value pairs and return as JSON:\n\n{extracted_text}"
    #         }
    #     ],
    #     response_format={"type": "json_object"}
    # )
    
    # Parse and return the JSON response
    # result = json.loads(kv_response.choices[0].message.content)
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract text from an image and convert to JSON')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--output', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    try:
        result = extract_text_from_image(args.image, args.api_key)
        
        # Print the result to console
        print(json.dumps(result, indent=2))
        
        # Save to file if output path is provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()