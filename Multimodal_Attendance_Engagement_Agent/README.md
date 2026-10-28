# 🎓 Multimodal Attendance & Engagement Agent

Skill Wallet Capstone Project 5: a privacy-preserving, fully offline classroom intelligence system
that combines **computer vision**, **audio signal processing** and **agent-based reasoning**.

## Features
| Epic | Feature |
|------|---------|
| 2 | Face detection + centroid tracking, **sustained-presence attendance** (default >= 70% of session), CSV export |
| 3 | Audio windows classified as Silence / Instructor Lecture / Student Discussion / High Ambient Distraction (FFT, speech-band ratio, spectral flatness) |
| 4 | Multimodal fusion into a normalised **Engagement Index (0-100)** with timeline plot |
| 5 | Teaching Suggestion Engine (attention drops, interaction peaks, pacing, participation) |
| 6 | Streamlit dashboard + downloadable TXT/JSON session report |

## Privacy
All processing happens locally. No video/audio is sent to any cloud API, nothing is written to disk
permanently, and the feedback is collective (pedagogical), not punitive.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Click **Load full demo session** in the sidebar to see every tab working without any upload.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io -> **New app** -> pick repo, branch `main`, file `app.py` -> **Deploy**.

## Tech stack
Python 3.9+, Streamlit, OpenCV (Haar cascade face detection), NumPy/SciPy (spectral analysis), Pandas, Matplotlib.

## Known limitations
- Haar cascades work best on frontal, well-lit faces; heavy occlusion lowers accuracy.
- Audio classes are heuristic (no trained model), tuned for typical classroom recordings.
- Head pose is approximated by frontal vs profile detection.
