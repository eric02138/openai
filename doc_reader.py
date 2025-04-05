import os
from openai import OpenAI, APIError, RateLimitError
import base64
from pathlib import Path

def extract_id_text(client, image_path):
    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Extract text with GPT-4 Turbo
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract ALL text visible on this ID document. 
                        
Provide ONLY the exact text as it appears, preserving:
- Field labels and values (e.g., "Name: John Smith")
- Document numbers, dates, codes
- ALL text visible on the ID, front and back if shown
- Exact spacing and formatting where possible

Format as plain text with no commentary or analysis. Organize by sections if multiple sections exist."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # High detail for text extraction
                        }
                    }
                ]
            }]
        )
        return response.choices[0].message.content
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}. Please wait before retrying.")
        return None
    except APIError as e:
        print(f"API error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    # Init OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
        
    client = OpenAI(api_key=api_key)
    
    # Check if test_images directory exists
    image_dir = Path('test_images')
    if not image_dir.exists():
        print(f"Creating directory: {image_dir}")
        image_dir.mkdir()
        print(f"Please place ID images in the {image_dir} directory and run again")
        return
        
    # Process all images in directory
    image_files = list(image_dir.glob('**/*.jpg')) + list(image_dir.glob('**/*.png'))
    if not image_files:
        print(f"No images found in {image_dir} directory. Please add .jpg or .png files.")
        return
        
    # Create output directory for text files
    output_dir = Path('extracted_text')
    if not output_dir.exists():
        output_dir.mkdir()
        
    for image_path in image_files:
        print(f"\nProcessing {image_path}:")
        result = extract_id_text(client, image_path)
        if result:
            # Write to file
            output_file = output_dir / f"{image_path.stem}_text.txt"
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"Text extracted and saved to {output_file}")
            print("\nExtracted text:")
            print("-" * 50)
            print(result)
            print("-" * 50)

if __name__ == "__main__":
    main()