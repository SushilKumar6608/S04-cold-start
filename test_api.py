from dotenv import load_dotenv
import os, anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
msg = client.messages.create(
    model='claude-sonnet-4-5',
    max_tokens=50,
    messages=[{'role': 'user', 'content': 'say hello'}]
)
print(msg.content[0].text)