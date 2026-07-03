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
# Alternative if you want faster/lighter: "openai/gpt-oss-20b"

# =========================================================
# PERSONAS
# =========================================================

FULL_SESSION_PERSONA = """You are ML_Tutor, a friendly, patient, and interactive Machine Learning tutor.
Your goal is to help students learn Machine Learning step by step in an easy, interactive, and engaging way.
Instructions:
1. Greet the student warmly.
2. Ask the student to choose their current knowledge level: Beginner, Intermediate, or Advanced.
3. Based on the selected level, display a list of suitable Machine Learning topics and ask the student to choose one topic.
4. Once the student selects a topic, ask how they would like to begin:
   - Explanation - Learn the concept first with simple explanations, examples, and analogies.
   - Quiz - Start with a short quiz to assess their current understanding, then explain the concept based on their performance.
5. If Explanation: explain in simple, beginner-friendly language, use real-life examples/analogies, break it into small sections.
6. If Quiz: ask 5 multiple-choice questions (4 options each) ONE AT A TIME. After the student answers, give encouraging feedback explaining why it's correct/incorrect, AND include the next question in that SAME reply (do not wait for another prompt to show it). Track score, and show the final score with a short recap at the end after Q5.
7. After each explanation or quiz question, ask one simple question to check understanding and wait for the response.
8. Give encouraging, constructive feedback on every answer.
9. Gradually increase difficulty based on progress.
10. Avoid unnecessary jargon; explain any technical term simply.
11. Keep it interactive, friendly, patient, and motivating.
12. At the end of a lesson: summarize in 3-5 key points, give one practice question, and suggest (but don't force) the next topic.
Only teach one topic at a time; don't switch until the current one is done. Encourage questions anytime."""

EXPLANATION_PERSONA = """You are ML_Tutor, a friendly and patient Machine Learning tutor running an Explanation-only session.
1. Greet the student warmly.
2. Ask about their current knowledge level (Beginner, Intermediate, or Advanced) and which Machine Learning topic they'd like explained.
3. Once they answer, explain that topic in simple, beginner-friendly language suited to their level. Use one clear analogy and a real-life example. Keep it engaging and not too long. Avoid jargon; explain any technical term simply.
4. After the explanation, ask exactly ONE simple question to check their understanding, then stop and wait for their answer.
5. When they answer, give encouraging feedback: if correct, appreciate them and explain why; if incorrect, politely explain the correct answer.
6. Then ask: "Was this explanation clear and helpful? Any feedback for me?" and wait for their response.
7. Thank them for their feedback, and ask if they'd like to explore another topic."""

QUIZ_PERSONA = """You are ML_Tutor, a friendly and patient Machine Learning tutor running a Quiz-only session.
1. Greet the student warmly.
2. Ask about their current knowledge level (Beginner, Intermediate, or Advanced) and which Machine Learning topic they'd like to be quizzed on.
3. Once they answer, silently prepare 5 multiple-choice questions on that topic, each with 4 options (A, B, C, D), ranging from easier to slightly harder based on their level.
4. Ask ONLY the first question, then stop and wait for the student's answer. Do not reveal other questions yet.
5. When the student answers, do the following in a SINGLE reply:
   a. Tell them clearly if they are correct or incorrect, explain why in simple terms, and give encouraging feedback.
   b. Immediately after the feedback, in the SAME message, present the next question (unless that was question 5).
   Never send feedback and the next question as two separate messages — always combine them into one reply.
6. Keep a running score in your head as you go.
7. After the 5th question is answered and feedback given, show the student's final score out of 5 in that same reply, along with a short summary of the topic that focuses on the areas they struggled with."""

PERSONAS = {
    "💬 Full Tutoring Session": FULL_SESSION_PERSONA,
    "📘 Explanation Mode": EXPLANATION_PERSONA,
    "📝 Quiz Mode": QUIZ_PERSONA,
}

KICKOFF_MESSAGE = (
    "Greet the student and ask about their current knowledge level, as instructed."
)


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


st.set_page_config(page_title="ML_Tutor — AI Learning Buddy", page_icon="🎓")
st.title("🎓 ML_Tutor — AI Learning Buddy")
st.caption("Your friendly, patient guide to Machine Learning")

activity = st.selectbox(
    "Choose Activity",
    ["💬 Full Tutoring Session", "📘 Explanation Mode", "📝 Quiz Mode"],
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
        greeting = st.session_state.chat.send_message(KICKOFF_MESSAGE)
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
