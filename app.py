import os
import json
import uuid
import re
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Video Interviewer")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

sessions = {}


# ============================================================
# DATA MODELS
# ============================================================

class StartInterviewRequest(BaseModel):
    name: str = "Candidate"
    role: str = "Software Engineer"
    experience: str = "Fresher"
    difficulty: str = "Medium"


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class NextQuestionRequest(BaseModel):
    session_id: str
    answer: str


# ============================================================
# OPENAI HELPER
# ============================================================

async def ask_openai(system_prompt: str, user_prompt: str):
    """
    Calls OpenAI directly through HTTP.
    If no API key is configured, returns None.
    """

    if not OPENAI_API_KEY:
        return None

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 700,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("OpenAI error:", e)
        return None


# ============================================================
# FALLBACK QUESTIONS
# ============================================================

QUESTION_BANK = {
    "Software Engineer": [
        "Tell me about yourself and your technical background.",
        "Explain one project you are particularly proud of.",
        "What is the difference between a process and a thread?",
        "How would you design a scalable web application?",
        "What happens when you enter a URL into a browser?",
        "Explain REST APIs and how you have used them.",
        "How do you debug a difficult software problem?",
        "Tell me about a time you worked with a team to solve a technical problem.",
    ],
    "Data Scientist": [
        "Tell me about yourself and your data science background.",
        "Explain one machine learning project you have worked on.",
        "What is the difference between supervised and unsupervised learning?",
        "How do you handle missing data?",
        "Explain overfitting and how you would prevent it.",
        "What is cross-validation?",
        "How would you evaluate a classification model?",
        "Tell me about a difficult data problem you solved.",
    ],
    "Machine Learning Engineer": [
        "Tell me about your machine learning experience.",
        "Describe an ML project you have built.",
        "What is the bias-variance tradeoff?",
        "How does gradient descent work?",
        "How would you deploy a machine learning model?",
        "What is feature engineering?",
        "How would you handle an imbalanced dataset?",
        "How would you monitor an ML model in production?",
    ],
    "HR": [
        "Tell me about yourself.",
        "Why are you interested in this position?",
        "What are your greatest strengths?",
        "What is one weakness you are currently working on?",
        "Tell me about a challenging situation you handled.",
        "Describe a time you worked in a team.",
        "Where do you see yourself in five years?",
        "Why should we hire you?",
    ],
}


def get_fallback_question(role, index):
    questions = QUESTION_BANK.get(
        role,
        QUESTION_BANK["Software Engineer"],
    )

    return questions[index % len(questions)]


# ============================================================
# INTERVIEWER PROMPT
# ============================================================

def build_question_prompt(session):
    previous_answers = session["answers"][-5:]

    answers_text = "\n".join(
        [
            f"Question: {item['question']}\nAnswer: {item['answer']}"
            for item in previous_answers
        ]
    )

    return f"""
You are an experienced professional AI interviewer.

Candidate:
Name: {session['name']}
Role: {session['role']}
Experience: {session['experience']}
Difficulty: {session['difficulty']}

The candidate has already answered some questions:

{answers_text if answers_text else "No previous answers."}

Generate ONE interview question.

Rules:
- Ask only one question.
- Make it relevant to the candidate's role.
- Adapt the difficulty to the candidate's experience.
- If the previous answer was weak, ask a useful follow-up.
- If the previous answer was strong, gradually increase difficulty.
- Do not repeat previous questions.
- Do not provide the answer.
- Do not add explanations.
- Return only the question.
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return HTMLResponse(
        content=HTML_PAGE
    )


# ============================================================
# START INTERVIEW
# ============================================================

@app.post("/api/start")
async def start_interview(data: StartInterviewRequest):

    session_id = str(uuid.uuid4())

    session = {
        "id": session_id,
        "name": data.name,
        "role": data.role,
        "experience": data.experience,
        "difficulty": data.difficulty,
        "started_at": datetime.now().isoformat(),
        "question_number": 0,
        "questions": [],
        "answers": [],
    }

    sessions[session_id] = session

    question = await ask_openai(
        """
You are an expert technical and behavioral interviewer.
Generate concise interview questions.
Return only the question.
""",
        build_question_prompt(session),
    )

    if not question:
        question = get_fallback_question(
            data.role,
            0,
        )

    question = clean_question(question)

    session["questions"].append(question)
    session["question_number"] = 1

    return {
        "success": True,
        "session_id": session_id,
        "question": question,
        "question_number": 1,
    }


# ============================================================
# NEXT QUESTION
# ============================================================

@app.post("/api/next")
async def next_question(data: NextQuestionRequest):

    session = sessions.get(data.session_id)

    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Interview session not found."},
        )

    if data.answer.strip():
        previous_question = session["questions"][-1]

        session["answers"].append(
            {
                "question": previous_question,
                "answer": data.answer.strip(),
            }
        )

    session["question_number"] += 1

    # Finish after 8 questions
    if session["question_number"] > 8:
        return {
            "finished": True,
        }

    question = await ask_openai(
        """
You are an expert AI interviewer.
Ask one relevant interview question.
Return only the question.
""",
        build_question_prompt(session),
    )

    if not question:
        question = get_fallback_question(
            session["role"],
            session["question_number"] - 1,
        )

    question = clean_question(question)

    session["questions"].append(question)

    return {
        "finished": False,
        "question": question,
        "question_number": session["question_number"],
    }


# ============================================================
# FINAL EVALUATION
# ============================================================

@app.post("/api/evaluate")
async def evaluate(data: AnswerRequest):

    session = sessions.get(data.session_id)

    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Interview session not found."},
        )

    if data.answer.strip():
        session["answers"].append(
            {
                "question": session["questions"][-1],
                "answer": data.answer.strip(),
            }
        )

    conversation = "\n\n".join(
        [
            f"""
QUESTION:
{item['question']}

CANDIDATE ANSWER:
{item['answer']}
"""
            for item in session["answers"]
        ]
    )

    evaluation_prompt = f"""
Evaluate this candidate's interview.

Candidate:
Name: {session['name']}
Role: {session['role']}
Experience: {session['experience']}

Interview:
{conversation}

Return ONLY valid JSON in exactly this format:

{{
  "overall_score": 0,
  "technical_score": 0,
  "communication_score": 0,
  "confidence_score": 0,
  "relevance_score": 0,
  "strengths": [
    "strength 1",
    "strength 2",
    "strength 3"
  ],
  "improvements": [
    "improvement 1",
    "improvement 2",
    "improvement 3"
  ],
  "summary": "short professional summary",
  "recommendation": "Strong Candidate"
}}

Scores must be between 0 and 100.
"""

    result = await ask_openai(
        "You are an expert interview evaluator. Return valid JSON only.",
        evaluation_prompt,
    )

    if result:

        try:
            result = clean_json(result)
            evaluation = json.loads(result)

        except Exception:
            evaluation = fallback_evaluation(session)

    else:
        evaluation = fallback_evaluation(session)

    session["evaluation"] = evaluation

    return {
        "success": True,
        "evaluation": evaluation,
    }


# ============================================================
# UTILITIES
# ============================================================

def clean_question(question):

    question = question.strip()

    question = re.sub(
        r"^(question|interviewer)\s*:\s*",
        "",
        question,
        flags=re.IGNORECASE,
    )

    question = question.replace('"', "")

    return question


def clean_json(text):

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"```json",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end >= 0:
        return text[start:end + 1]

    return text


def fallback_evaluation(session):

    answers = session.get("answers", [])

    total_length = sum(
        len(a["answer"].split())
        for a in answers
    )

    if not answers:
        communication = 0
    else:
        communication = min(
            95,
            55 + total_length // 4,
        )

    technical = min(
        90,
        50 + len(answers) * 5,
    )

    confidence = min(
        90,
        55 + len(answers) * 4,
    )

    relevance = min(
        90,
        55 + len(answers) * 5,
    )

    overall = int(
        (
            technical
            + communication
            + confidence
            + relevance
        ) / 4
    )

    return {
        "overall_score": overall,
        "technical_score": technical,
        "communication_score": communication,
        "confidence_score": confidence,
        "relevance_score": relevance,
        "strengths": [
            "Completed the interview.",
            "Provided responses to the interview questions.",
            "Demonstrated willingness to communicate.",
        ],
        "improvements": [
            "Provide more specific examples.",
            "Structure answers more clearly.",
            "Include measurable results when discussing projects.",
        ],
        "summary": (
            "The candidate completed the interview. "
            "Further evaluation with an AI model is recommended "
            "for more detailed feedback."
        ),
        "recommendation": (
            "Strong Candidate"
            if overall >= 75
            else "Consider Further Review"
        ),
    }


# ============================================================
# FRONTEND
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>AI Video Interviewer</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #243b70,
            #0a0d16 45%,
            #05060a
        );

    color: white;
    min-height: 100vh;
}

.container {
    width: 94%;
    max-width: 1400px;
    margin: auto;
    padding: 25px 0 50px;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 15px 20px;
    margin-bottom: 20px;

    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;

    backdrop-filter: blur(20px);
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;

    font-size: 21px;
    font-weight: 800;
}

.logo-icon {
    width: 42px;
    height: 42px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #7c5cff,
            #19c8ff
        );

    font-size: 22px;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;

    color: #b8c1d9;
    font-size: 13px;
}

.status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #45e69b;
}

.setup {
    max-width: 700px;
    margin: 70px auto;

    padding: 40px;

    background:
        rgba(255,255,255,.07);

    border:
        1px solid rgba(255,255,255,.12);

    border-radius: 28px;

    backdrop-filter: blur(20px);

    box-shadow:
        0 30px 100px rgba(0,0,0,.4);
}

.setup h1 {
    margin-top: 0;
    font-size: 38px;
}

.setup p {
    color: #aab3cb;
    line-height: 1.6;
}

.form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-top: 25px;
}

.field {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.field.full {
    grid-column: 1 / -1;
}

label {
    color: #aeb7d0;
    font-size: 13px;
}

input, select {
    width: 100%;
    padding: 14px 15px;

    color: white;
    background: rgba(0,0,0,.3);

    border:
        1px solid rgba(255,255,255,.13);

    border-radius: 12px;

    outline: none;
}

button {
    border: none;
    cursor: pointer;

    color: white;
    font-weight: 700;

    padding: 14px 20px;

    border-radius: 12px;

    transition: .2s;
}

button:hover {
    transform: translateY(-2px);
}

.primary {
    width: 100%;
    margin-top: 25px;

    background:
        linear-gradient(
            135deg,
            #7256ff,
            #159fe8
        );
}

.interview {
    display: none;
}

.main-grid {
    display: grid;
    grid-template-columns:
        minmax(0, 1.45fr)
        minmax(320px, .8fr);

    gap: 20px;
}

.video-card,
.question-card,
.panel {
    background:
        rgba(255,255,255,.06);

    border:
        1px solid rgba(255,255,255,.1);

    border-radius: 22px;

    backdrop-filter: blur(20px);
}

.video-card {
    padding: 15px;
}

.video-wrapper {
    position: relative;

    overflow: hidden;

    aspect-ratio: 16 / 9;

    background: #080a10;

    border-radius: 16px;
}

video {
    width: 100%;
    height: 100%;

    object-fit: cover;

    transform: scaleX(-1);
}

.interviewer {
    position: absolute;

    left: 20px;
    bottom: 20px;

    width: 150px;
    height: 185px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            #5c4bb2,
            #172445
        );

    border:
        2px solid rgba(255,255,255,.3);

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    box-shadow:
        0 15px 40px rgba(0,0,0,.5);
}

.avatar-head {
    width: 65px;
    height: 65px;

    border-radius: 50%;

    background:
        linear-gradient(
            135deg,
            #f1c4a5,
            #bd806b
        );

    position: relative;
}

.avatar-head::before {
    content: "";

    position: absolute;

    width: 75px;
    height: 35px;

    left: -5px;
    top: -7px;

    border-radius: 50px 50px 10px 10px;

    background: #271c2e;
}

.avatar-body {
    width: 95px;
    height: 70px;

    margin-top: 10px;

    border-radius: 50px 50px 10px 10px;

    background:
        linear-gradient(
            135deg,
            #7759dd,
            #29204e
        );
}

.avatar-name {
    margin-top: 8px;
    font-size: 12px;
    color: #e7eaff;
}

.question-card {
    margin-top: 15px;
    padding: 25px;
}

.question-label {
    color: #8e9bff;
    font-size: 12px;
    font-weight: 800;

    text-transform: uppercase;
    letter-spacing: 1px;
}

.question {
    font-size: 23px;
    line-height: 1.45;

    margin: 12px 0 0;
}

.controls {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.controls button {
    flex: 1;
}

.mic {
    background: #1c263d;
}

.mic.active {
    background: #db3e68;
}

.next {
    background:
        linear-gradient(
            135deg,
            #7256ff,
            #159fe8
        );
}

.panel {
    padding: 25px;
}

.panel-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 18px;
}

.timer {
    font-size: 30px;
    font-weight: 800;

    margin-bottom: 20px;
}

.progress {
    height: 7px;

    background: rgba(255,255,255,.08);

    border-radius: 10px;

    overflow: hidden;

    margin-bottom: 25px;
}

.progress-bar {
    width: 0%;
    height: 100%;

    background:
        linear-gradient(
            90deg,
            #7256ff,
            #19c8ff
        );

    transition: .3s;
}

.transcript {
    min-height: 180px;

    padding: 15px;

    background:
        rgba(0,0,0,.2);

    border-radius: 14px;

    color: #bfc7dc;

    line-height: 1.6;

    margin-bottom: 20px;
}

.live {
    display: flex;
    align-items: center;
    gap: 8px;

    font-size: 12px;
    color: #93a0ba;
}

.live-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #ff526e;
}

.results {
    display: none;

    max-width: 950px;
    margin: 50px auto;
}

.score {
    text-align: center;

    padding: 40px;

    background:
        rgba(255,255,255,.07);

    border:
        1px solid rgba(255,255,255,.1);

    border-radius: 25px;
}

.score-number {
    font-size: 90px;
    font-weight: 900;

    background:
        linear-gradient(
            135deg,
            #8b76ff,
            #19c8ff
        );

    -webkit-background-clip: text;
    color: transparent;
}

.metrics {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 15px;

    margin-top: 20px;
}

.metric {
    padding: 20px;

    background:
        rgba(255,255,255,.06);

    border-radius: 16px;

    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    color: #98a4bd;
    font-size: 12px;
    margin-top: 5px;
}

.result-columns {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 20px;

    margin-top: 20px;
}

.result-box {
    padding: 25px;

    background:
        rgba(255,255,255,.06);

    border-radius: 18px;
}

.result-box h3 {
    margin-top: 0;
}

.result-box li {
    color: #b9c1d5;
    margin: 10px 0;
}

.summary {
    margin-top: 20px;

    padding: 25px;

    background:
        rgba(255,255,255,.06);

    border-radius: 18px;

    line-height: 1.7;

    color: #c2c9da;
}

@media(max-width: 900px) {

    .main-grid {
        grid-template-columns: 1fr;
    }

    .metrics {
        grid-template-columns: 1fr 1fr;
    }

}

@media(max-width: 600px) {

    .form-grid {
        grid-template-columns: 1fr;
    }

    .field.full {
        grid-column: auto;
    }

    .setup {
        padding: 25px;
        margin: 30px auto;
    }

    .setup h1 {
        font-size: 30px;
    }

    .result-columns {
        grid-template-columns: 1fr;
    }

}

</style>

</head>

<body>

<div class="container">

<header>

<div class="logo">

<div class="logo-icon">
🎙️
</div>

AI Interviewer

</div>

<div class="status">

<div class="status-dot"></div>

AI System Ready

</div>

</header>


<!-- ======================================================
     SETUP
======================================================= -->

<section id="setup" class="setup">

<h1>AI Video Interview</h1>

<p>
Practice a realistic interview with an AI interviewer.
Your webcam and microphone can be used during the session.
The interviewer will ask adaptive questions and generate
a performance report at the end.
</p>

<div class="form-grid">

<div class="field">

<label>Your Name</label>

<input
id="name"
placeholder="Enter your name"
value="Candidate"
>

</div>


<div class="field">

<label>Experience</label>

<select id="experience">

<option>Fresher</option>
<option>0-2 Years</option>
<option>2-5 Years</option>
<option>5+ Years</option>

</select>

</div>


<div class="field full">

<label>Interview Role</label>

<select id="role">

<option>Software Engineer</option>
<option>Machine Learning Engineer</option>
<option>Data Scientist</option>
<option>HR</option>

</select>

</div>


<div class="field full">

<label>Difficulty</label>

<select id="difficulty">

<option>Easy</option>
<option selected>Medium</option>
<option>Hard</option>

</select>

</div>

</div>

<button
class="primary"
onclick="startInterview()"
>

Start AI Interview →

</button>

</section>


<!-- ======================================================
     INTERVIEW
======================================================= -->

<section id="interview" class="interview">

<div class="main-grid">

<div>

<div class="video-card">

<div class="video-wrapper">

<video
id="camera"
autoplay
muted
playsinline
></video>


<div class="interviewer">

<div class="avatar-head"></div>

<div class="avatar-body"></div>

<div class="avatar-name">
AI Interviewer
</div>

</div>

</div>

</div>


<div class="question-card">

<div class="question-label">
Question <span id="questionNumber">1</span> / 8
</div>

<div
id="question"
class="question"
>
Preparing your interview...
</div>

<div class="controls">

<button
id="micButton"
class="mic"
onclick="toggleRecording()"
>

🎤 Start Answer

</button>

<button
class="next"
onclick="nextQuestion()"
>

Next →

</button>

</div>

</div>

</div>


<div class="panel">

<div class="panel-title">
Interview Progress
</div>

<div
id="timer"
class="timer"
>
00:00
</div>

<div class="progress">

<div
id="progressBar"
class="progress-bar"
></div>

</div>

<div class="live">

<div class="live-dot"></div>

Live interview

</div>

<br>

<div class="panel-title">
Your Answer
</div>

<div
id="transcript"
class="transcript"
>
Your spoken answer will appear here...
</div>

<div class="panel-title">
Interview Tips
</div>

<ul style="color:#aeb7cc;line-height:1.7">

<li>Speak clearly.</li>

<li>Give specific examples.</li>

<li>Structure your answer.</li>

<li>Maintain eye contact.</li>

<li>Keep answers concise.</li>

</ul>

</div>

</div>

</section>


<!-- ======================================================
     RESULTS
======================================================= -->

<section id="results" class="results">

<div class="score">

<div
style="color:#9ba6c0"
>
Overall Interview Score
</div>

<div
id="overallScore"
class="score-number"
>
0
</div>

<div
id="recommendation"
style="font-size:20px;font-weight:800"
>
-
</div>

</div>


<div class="metrics">

<div class="metric">

<div
id="technical"
class="metric-value"
>
0
</div>

<div class="metric-label">
Technical
</div>

</div>


<div class="metric">

<div
id="communication"
class="metric-value"
>
0
</div>

<div class="metric-label">
Communication
</div>

</div>


<div class="metric">

<div
id="confidence"
class="metric-value"
>
0
</div>

<div class="metric-label">
Confidence
</div>

</div>


<div class="metric">

<div
id="relevance"
class="metric-value"
>
0
</div>

<div class="metric-label">
Relevance
</div>

</div>

</div>


<div class="result-columns">

<div class="result-box">

<h3>💪 Strengths</h3>

<ul id="strengths"></ul>

</div>


<div class="result-box">

<h3>🎯 Improvements</h3>

<ul id="improvements"></ul>

</div>

</div>


<div class="summary">

<h3>AI Interview Summary</h3>

<p id="summary"></p>

</div>

<button
class="primary"
onclick="location.reload()"
>

Start New Interview

</button>

</section>

</div>


<script>

let sessionId = null;

let recognition = null;

let recording = false;

let currentAnswer = "";

let questionNumber = 1;

let startTime = null;

let timerInterval = null;

let cameraStream = null;


async function startCamera() {

    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });

        document
            .getElementById("camera")
            .srcObject = cameraStream;

    } catch(error) {

        console.error(error);

        alert(
            "Camera/microphone permission was not granted. " +
            "You can still continue the interview."
        );

    }

}


function setupSpeechRecognition() {

    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    if (!SpeechRecognition) {

        console.log(
            "Speech recognition is not supported."
        );

        return;

    }

    recognition =
        new SpeechRecognition();

    recognition.continuous = true;

    recognition.interimResults = true;

    recognition.lang = "en-US";


    recognition.onresult = function(event) {

        let finalText = "";

        for (
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ) {

            finalText +=
                event.results[i][0].transcript;

        }

        currentAnswer = finalText.trim();

        document
            .getElementById("transcript")
            .textContent =
                currentAnswer ||
                "Listening...";

    };


    recognition.onerror = function(event) {

        console.log(
            "Speech recognition error:",
            event.error
        );

    };


    recognition.onend = function() {

        if (recording) {

            try {
                recognition.start();
            } catch(e) {}

        }

    };

}


function speak(text) {

    if (!window.speechSynthesis) {
        return;
    }

    window.speechSynthesis.cancel();

    const utterance =
        new SpeechSynthesisUtterance(text);

    utterance.rate = 0.95;

    utterance.pitch = 1.05;

    utterance.volume = 1;

    const voices =
        window.speechSynthesis.getVoices();

    const femaleVoice =
        voices.find(
            voice =>
                /female|samantha|victoria|karen|zira/i
                .test(voice.name)
        );

    if (femaleVoice) {

        utterance.voice =
            femaleVoice;

    }

    window.speechSynthesis.speak(
        utterance
    );

}


async function startInterview() {

    const name =
        document
            .getElementById("name")
            .value
            .trim() ||
        "Candidate";

    const role =
        document
            .getElementById("role")
            .value;

    const experience =
        document
            .getElementById("experience")
            .value;

    const difficulty =
        document
            .getElementById("difficulty")
            .value;


    const response =
        await fetch("/api/start", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                name,
                role,
                experience,
                difficulty
            })

        });


    const data =
        await response.json();


    if (!data.success) {

        alert(
            data.error ||
            "Could not start interview."
        );

        return;

    }


    sessionId =
        data.session_id;

    questionNumber =
        data.question_number;


    document
        .getElementById("setup")
        .style.display = "none";


    document
        .getElementById("interview")
        .style.display = "block";


    startTime =
        Date.now();


    timerInterval =
        setInterval(updateTimer, 1000);


    setupSpeechRecognition();

    await startCamera();


    showQuestion(
        data.question,
        questionNumber
    );

}


function showQuestion(
    question,
    number
) {

    document
        .getElementById("question")
        .textContent =
            question;

    document
        .getElementById("questionNumber")
        .textContent =
            number;

    const progress =
        ((number - 1) / 8) * 100;

    document
        .getElementById("progressBar")
        .style.width =
            progress + "%";


    currentAnswer = "";

    document
        .getElementById("transcript")
        .textContent =
            "Your spoken answer will appear here...";


    speak(question);

}


function toggleRecording() {

    if (!recognition) {

        alert(
            "Speech recognition is not available in this browser."
        );

        return;

    }


    const button =
        document
            .getElementById("micButton");


    if (!recording) {

        recording = true;

        currentAnswer = "";

        button.textContent =
            "⏹ Stop Answer";

        button.classList.add("active");


        try {
            recognition.start();
        } catch(e) {}

    }

    else {

        recording = false;

        button.textContent =
            "🎤 Start Answer";

        button.classList.remove("active");


        try {
            recognition.stop();
        } catch(e) {}

    }

}


async function nextQuestion() {

    if (!sessionId) {
        return;
    }


    if (recording) {

        recording = false;

        try {
            recognition.stop();
        } catch(e) {}

        const button =
            document
                .getElementById("micButton");

        button.textContent =
            "🎤 Start Answer";

        button.classList.remove("active");

    }


    const response =
        await fetch("/api/next", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({

                session_id:
                    sessionId,

                answer:
                    currentAnswer

            })

        });


    const data =
        await response.json();


    if (data.finished) {

        await finishInterview();

        return;

    }


    questionNumber =
        data.question_number;


    showQuestion(
        data.question,
        questionNumber
    );

}


async function finishInterview() {

    clearInterval(timerInterval);


    const response =
        await fetch("/api/evaluate", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({

                session_id:
                    sessionId,

                answer:
                    currentAnswer

            })

        });


    const data =
        await response.json();


    if (!data.success) {

        alert(
            "Could not generate interview report."
        );

        return;

    }


    displayResults(
        data.evaluation
    );


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

    }

}


function displayResults(evaluation) {

    document
        .getElementById("interview")
        .style.display = "none";


    document
        .getElementById("results")
        .style.display = "block";


    document
        .getElementById("overallScore")
        .textContent =
            evaluation.overall_score;


    document
        .getElementById("technical")
        .textContent =
            evaluation.technical_score;


    document
        .getElementById("communication")
        .textContent =
            evaluation.communication_score;


    document
        .getElementById("confidence")
        .textContent =
            evaluation.confidence_score;


    document
        .getElementById("relevance")
        .textContent =
            evaluation.relevance_score;


    document
        .getElementById("recommendation")
        .textContent =
            evaluation.recommendation;


    const strengths =
        document
            .getElementById("strengths");

    strengths.innerHTML = "";

    evaluation.strengths
        .forEach(item => {

            const li =
                document.createElement("li");

            li.textContent = item;

            strengths.appendChild(li);

        });


    const improvements =
        document
            .getElementById("improvements");

    improvements.innerHTML = "";

    evaluation.improvements
        .forEach(item => {

            const li =
                document.createElement("li");

            li.textContent = item;

            improvements.appendChild(li);

        });


    document
        .getElementById("summary")
        .textContent =
            evaluation.summary;

}


function updateTimer() {

    if (!startTime) {
        return;
    }

    const seconds =
        Math.floor(
            (Date.now() - startTime)
            / 1000
        );

    const minutes =
        Math.floor(seconds / 60);

    const remaining =
        seconds % 60;


    document
        .getElementById("timer")
        .textContent =
            String(minutes)
                .padStart(2, "0")
            + ":" +
            String(remaining)
                .padStart(2, "0");

}


</script>

</body>

</html>
"""


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
    )