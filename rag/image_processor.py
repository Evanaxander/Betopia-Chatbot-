import base64
from openai import OpenAI

def describe_image(image_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail. If it is a chart, explain the data. If it is a diagram, explain the steps."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ],
            }
        ],
    )
    return response.choices[0].message.content