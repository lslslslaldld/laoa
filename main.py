import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables from the .env file
load_dotenv()

def main():
    # 2. Set up the -m command-line flag
    parser = argparse.ArgumentParser(description="Send a prompt to NVIDIA Nemotron Super.")
    parser.add_argument("-m", "--message", required=True, help="The prompt to send")
    args = parser.parse_args()

    # 3. Initialize the client pointing to NVIDIA's NIM endpoint
    api_key = "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"
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found. Make sure it is set in your .env file.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    # 4. Generate and print the response
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl-120b-a12b",
        messages=[{"role": "user", "content": args.message}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
    )
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    main()
