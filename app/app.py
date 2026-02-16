from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import logging



load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


# Paths relative to this script so the app works when run from any directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Jayakanthan Durairaj"
        logging.getLogger("pypdf").setLevel(logging.ERROR)  # suppress "wrong pointing object" warnings from malformed PDFs
        pdf_path = os.path.join(_BASE_DIR, "me", "LinkedIn.pdf")
        reader = PdfReader(pdf_path)
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        summary_path = os.path.join(_BASE_DIR, "me", "summary.txt")
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name} and introduce yourself as {self.name} to the user before starting the conversation. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
dont forget to ask their name and email before start the conversation. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
if __name__ == "__main__":
    me = Me()
    avatar_image_path = os.path.join(_BASE_DIR, "me", "avatar.png")
    avatar_path = avatar_image_path if os.path.exists(avatar_image_path) else None

    examples = [
        "Can you describe Jay’s experience scaling engineering teams?",
        "What is his experience with AI/LLM-based systems?",
        "What are his top 3 technical strengths?",
        "How can I schedule an interview?"
    ]

    # Note: Removed theme and css from here
    with gr.Blocks(title="Meet Jay") as app_chat:
        gr.Markdown("# 🚀 Jay's Professional Liaison")
        
        # Removed type="messages" (it's now the default)
        my_chatbot = gr.Chatbot(
            label="Jay's Career Agent",
            avatar_images=(None, avatar_path),
            placeholder="<strong>Hello! I am Jay’s Professional Liaison</strong><br>I'm an AI agent trained on Jay's career history. Ask me about his experience, skills, or how to get in touch or choose a topic below."
        )

        gr.ChatInterface(
            fn=me.chat,
            chatbot=my_chatbot,
            examples=examples
        )

    # Moved theme and css to launch()
    app_chat.launch(
        share=True,
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}" # Example CSS to hide Gradio footer
    )