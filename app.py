import streamlit as st
from openai import OpenAI
from types import SimpleNamespace
import os

# Groq's API is OpenAI-compatible, so we reuse the same `openai` library,
# just pointed at Groq's endpoint with a Groq API key.
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL_NAME = "openai/gpt-oss-120b"  # free on Groq; current recommended model

# =========================================================
# PERSONAS
# =========================================================

EXPLANATION_PERSONA = """You are ML_Tutor, a friendly and patient Machine Learning tutor running an Explanation-only session.
1. Greet the student warmly.
2. First, ask them to select their current knowledge level: Beginner, Intermediate, or Advanced.
3. Once they select a level, ask them to choose a specific Machine Learning topic they’d like explained.
4. After they answer, explain that topic in simple, beginner-friendly language suited to their level. Use one clear analogy and a real-life example. Avoid jargon; explain any technical term simply.
5. After the explanation, ask for feedback: 'Was this explanation clear and helpful?'
6. Adjust your next responses based on the feedback received."""

QUIZ_PERSONA = """You are ML_Tutor, a friendly and patient Machine Learning tutor running a Quiz-only session.
1. Greet the student warmly.
2. First, ask them to select their current knowledge level: Beginner, Intermediate, or Advanced.
3. Once they select a level, ask them to choose a specific Machine Learning topic they’d like to be quizzed on.
4. Once they answer, silently prepare 5 multiple-choice questions on that topic, each with 4 options (A, B, C, D).
5. Ask ONLY the first question, then stop and wait for the student's answer.
6. When the student answers, in a SINGLE reply:
   a. Tell them if they are correct or incorrect, explain why in simple terms, and give encouraging feedback.
   b. Immediately after the feedback, present the next question (unless that was question 5).
7. Keep a running score in your head.
8. After the 5th question, show the student's final score and a short recap.
9. Finally, ask for feedback: 'Was this quiz helpful for your learning?'
10. Adjust your future quizzes based on the feedback received."""

ASK_ANYTHING_PERSONA = """You are ML_Tutor in Ask Anything mode.
1. The student may ask about any topic (not limited to ML).
2. Provide a clear, simple, general explanation of the topic.
3. Avoid jargon; keep it beginner-friendly.
4. After the explanation, ask for feedback: 'Was this explanation clear and useful?'
5. Adapt your future responses based on the feedback received."""

REAL_WORLD_EXAMPLE_PERSONA = """You are ML_Tutor in Real World Example mode.
1. The student chooses a Machine Learning topic.
2. Provide a practical, real-world example or application of that topic.
3. Keep it simple, engaging, and relatable.
4. After the example, ask for feedback: 'Did this example help you understand better?'
5. Adapt your next explanation or example according to their feedback."""

PERSONAS = {
    "📘 Explanation Mode": EXPLANATION_PERSONA,
    "📝 Quiz Mode": QUIZ_PERSONA,
    "❓ Ask Anything": ASK_ANYTHING_PERSONA,
    "🌍 Real World Example": REAL_WORLD_EXAMPLE_PERSONA,
}

# Mode-specific kickoff messages
KICKOFF_MESSAGES = {
    "📘 Explanation Mode": "Hello! 👋 Please start by selecting your knowledge level (Beginner, Intermediate, Advanced). Then tell me which ML topic you'd like explained.",
    "📝 Quiz Mode": "Hi there! 👋 First, please select your knowledge level (Beginner, Intermediate, Advanced). Then tell me which ML topic you'd like to be quizzed on.",
    "❓ Ask Anything": "Hello! 👋 Feel free to ask me about any topic you’re curious about, and I’ll explain it simply.",
    "🌍 Real World Example": "Hi! 👋 Please choose a Machine Learning topic, and I’ll give you a practical real-world example of it.",
}

# ---- Wrapper so send_message()/.text interface stays consistent ----
class GroqChat:
    def __init__(self, persona):
        self.persona = persona
        self.history = []

    def send_message(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.persona}] + self.history
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        reply_text = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply_text})
        return SimpleNamespace(text=reply_text)

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="ML_Tutor — AI Learning Buddy", page_icon="🎓")
st.title("🎓 ML_Tutor — AI Learning Buddy")
st.caption("Your friendly, patient guide to Machine Learning")

activity = st.selectbox(
    "Choose Activity",
    ["📘 Explanation Mode", "📝 Quiz Mode", "❓ Ask Anything", "🌍 Real World Example"],
)

# Reset chat state if the user switches modes
if st.session_state.get("active_mode") != activity:
    for key in ["chat", "messages"]:
        st.session_state.pop(key, None)
    st.session_state.active_mode = activity

# Start the chat (greeting) the first time this mode is opened
if "chat" not in st.session_state:
    st.session_state.chat = GroqChat(PERSONAS[activity])
    with st.spinner("ML_Tutor is getting ready..."):
        greeting = st.session_state.chat.send_message(KICKOFF_MESSAGES[activity])
    st.session_state.messages = [("ML_Tutor", greeting.text)]

# Render conversation so far
for sender, text in st.session_state.messages:
    with st.chat_message("assistant" if sender == "ML_Tutor" else "user"):
        st.write(text)

# Chat input drives every subsequent turn
user_input = st.chat_input("Reply to ML_Tutor...")
if user_input:
    st.session_state.messages.append(("You", user_input))
    with st.spinner("ML_Tutor is thinking..."):
        reply = st.session_state.chat.send_message(user_input)
    st.session_state.messages.append(("ML_Tutor", reply.text))
    st.rerun()

if st.button("Restart this session"):
    for key in ["chat", "messages"]:
        st.session_state.pop(key, None)
    st.rerun()
