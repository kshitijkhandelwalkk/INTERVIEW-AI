# AI Video Interviewer — Interactive Avatar Edition

This version upgrades the original project with a named interviewer system.

## Interviewers

- **Sarah** — HR Manager — warm and conversational
- **Michael** — Technical Lead — direct and technical
- **Emily** — Product Manager — analytical and curious
- **David** — Engineering Manager — challenging and leadership-focused

The selected name is sent to the AI prompt, so the question style changes with the interviewer.

## Realistic avatars

The project supports **LiveAvatar**. When `LIVEAVATAR_API_KEY` is configured, each interviewer card is automatically mapped to a different public LiveAvatar and the interview room can show the realistic talking video avatar.

Without a LiveAvatar key, the app still works using the built-in animated fallback avatar and browser text-to-speech.

## Run

From the project folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Add your keys to `.env`.

Then:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

Chrome is recommended because browser speech recognition works best there.

## Camera / microphone

When Chrome asks, click **Allow** for Camera and Microphone.

If permissions were blocked:
Chrome → Settings → Privacy and security → Site settings → Camera / Microphone → allow `localhost`.

## LiveAvatar

Set:

```text
LIVEAVATAR_API_KEY=your_key_here
LIVEAVATAR_SANDBOX=true
```

Restart the server after changing `.env`.

The setup screen will automatically show `REALISTIC VIDEO` on interviewer cards when LiveAvatar avatars are available.

## Important

Never commit `.env` or API keys to GitHub.
