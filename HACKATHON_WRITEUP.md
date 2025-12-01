# Hackathon Write-up — EduInvest

## Problem statement

Many students and beginner investors struggle to connect concepts taught in the classroom to real market behavior. There is a gap between academic indicators and intuitive understanding of short-term market movements.

EduInvest asks: how can we make learning about markets interactive, approachable, and fun for students?

---

## Target audience

- Students learning personal finance, economics, or data science
- Educators who want a hands-on demonstration to complement lectures
- Hackathon judges and mentors looking for a brief, demoable project

---

## Why this project matters

- It lowers the barrier to experimenting with market data and AI predictions.
- Gamification helps motivate repeated practice and discovery.
- Demonstrates a full-stack mini-project: GUI, data ingestion, model inference, and visualization.

---

## What's innovative

- Combines a teaching-driven UX (game, score, streaks) with modular AI components.
- Designed to plug-in complex multimodal models while keeping the UI lightweight.
- Emphasizes explainability and extensibility: educators can replace or adapt models quickly.

---

## Challenges faced

1. Packaging ML dependencies:
   - Solution: Provide a simple install script and recommend remote inference or conda for heavy dependencies.
2. Keeping the UI responsive while fetching data and running models:
   - Solution: For the hackathon prototype we ran things synchronously, but the architecture supports moving work to threads.
3. Balancing realism vs demo speed:
   - Solution: Use a placeholder model for instant results while leaving hooks for improved models.

---

## Demo script (2-3 minutes)

1. Open the app and show the trending list and chart.
2. Make a guess on a suggested stock and show the result dialog.
3. Explain where the model would be swapped in and how the chart helps with intuition.
4. Mention future ideas: remote inference, lesson-mode, and a leaderboard.

---

## Team credits & acknowledgements

- Team: [Your name(s) here]
- Tools: Python, PyQt6, yfinance, matplotlib, PyTorch (optional)

---

## Contact

For follow-up, include an email or GitHub handle.
