"""
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5

Privacy-first, 100% offline classroom analytics:
  Tab 1  Presence-based attendance (OpenCV face detection + centroid tracking)
  Tab 2  Acoustic interaction analysis (FFT-based voice activity + noise dynamics)
  Tab 3  Multimodal fusion -> Engagement Index (0-100)
  Tab 4  Teaching suggestion engine (rule-based agent)
  Tab 5  Session analytics + downloadable educator report

No video/audio is ever sent to any cloud service. Nothing is stored on disk
(uploads are processed in memory / temp files that are deleted immediately).
"""

import io
import json
import os
import tempfile
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LABELS = ["Silence", "Instructor Lecture", "Student Discussion", "High Ambient Distraction"]
AUDIO_SCORE = {
    "Silence": 40,
    "Instructor Lecture": 70,
    "Student Discussion": 100,
    "High Ambient Distraction": 10,
}
LABEL_COLORS = {
    "Silence": "#9aa5b1",
    "Instructor Lecture": "#2563eb",
    "Student Discussion": "#16a34a",
    "High Ambient Distraction": "#dc2626",
}

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .hero {background: linear-gradient(90deg,#1d4ed8,#2563eb 60%,#38bdf8);
           padding: 1.1rem 1.4rem; border-radius: 12px; color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 1.7rem;}
    .hero p {margin: .25rem 0 0 0; opacity: .92;}
    .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem;
           font-weight:600; color:white;}
    .sev-High {background:#dc2626;} .sev-Medium {background:#f59e0b;}
    .sev-Low {background:#16a34a;} .sev-Info {background:#2563eb;}
    </style>
    <div class="hero">
      <h1>🎓 Multimodal Attendance &amp; Engagement Agent</h1>
      <p>Privacy-preserving, offline classroom intelligence: vision + audio + agent-based reasoning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_t(sec):
    sec = int(max(0, sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def find_runs(mask):
    """Return list of (start, end_exclusive) for consecutive True values."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@st.cache_resource
def load_cascades():
    base = cv2.data.haarcascades
    front = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
    prof = cv2.CascadeClassifier(os.path.join(base, "haarcascade_profileface.xml"))
    return front, prof


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect_faces(frame_bgr):
    """Return list of (x, y, w, h, pose) where pose is 'frontal' or 'profile'."""
    front, prof = load_cascades()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 800.0 / w if w > 800 else 1.0
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    gray = cv2.equalizeHist(gray)
    f_boxes = front.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
    p_boxes = prof.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
    out = []
    for (x, y, bw, bh) in f_boxes:
        out.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale), "frontal"))
    for (x, y, bw, bh) in p_boxes:
        box = (x, y, bw, bh)
        if all(_iou(box, tuple(fb)) < 0.25 for fb in f_boxes):
            out.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale), "profile"))
    return out


class CentroidTracker:
    """Greedy nearest-centroid tracker that assigns stable participant IDs."""

    def __init__(self, max_missed=4):
        self.next_id = 1
        self.tracks = {}
        self.max_missed = max_missed

    def update(self, dets, t):
        ids = [None] * len(dets)
        cents = [(d[0] + d[2] / 2, d[1] + d[3] / 2, d[2]) for d in dets]
        pairs = []
        for tid, tr in self.tracks.items():
            if tr["missed"] > self.max_missed:
                continue
            for j, c in enumerate(cents):
                dist = float(np.hypot(tr["c"][0] - c[0], tr["c"][1] - c[1]))
                if dist < max(40.0, 1.5 * c[2]):
                    pairs.append((dist, tid, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            ids[j] = tid
        for j, c in enumerate(cents):
            if ids[j] is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"first": t, "last": t, "frames": 0, "frontal": 0,
                                    "missed": 0, "c": (c[0], c[1])}
                ids[j] = tid
            tr = self.tracks[ids[j]]
            tr["last"] = t
            tr["missed"] = 0
            tr["c"] = (c[0], c[1])
            tr["frames"] += 1
            if dets[j][4] == "frontal":
                tr["frontal"] += 1
        active = set(ids)
        for tid, tr in self.tracks.items():
            if tid not in active:
                tr["missed"] += 1
        return ids


def draw_boxes(frame, dets, ids):
    img = frame.copy()
    for (x, y, w, h, pose), tid in zip(dets, ids):
        color = (22, 163, 74) if pose == "frontal" else (245, 158, 11)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"ID{tid}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_attendance(tracker, total_samples, threshold, sample_s):
    rows = []
    valid = [(tid, tr) for tid, tr in tracker.tracks.items()
             if tr["frames"] >= 2 or total_samples < 3]
    valid.sort(key=lambda kv: kv[1]["first"])
    for n, (_, tr) in enumerate(valid, start=1):
        pct = tr["frames"] / max(total_samples, 1)
        rows.append({
            "Student ID": f"S{n:02d}",
            "First Seen": fmt_t(tr["first"]),
            "Last Seen": fmt_t(tr["last"]),
            "Total Duration (s)": int(round(tr["frames"] * sample_s)),
            "Presence %": round(100 * pct, 1),
            "Facing Forward %": round(100 * tr["frontal"] / max(tr["frames"], 1), 1),
            "Status": "Present" if pct >= threshold else "Absent",
        })
    return pd.DataFrame(rows)


def analyse_video(path, sample_s, threshold, progress=None, max_samples=600):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps * sample_s)))
    if total_frames and total_frames / step > max_samples:
        step = int(np.ceil(total_frames / max_samples))
    eff_sample_s = step / fps
    tracker = CentroidTracker()
    rows, best, best_n = [], None, -1
    idx, samples = 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            dets = detect_faces(frame)
            ids = tracker.update(dets, t)
            rows.append({"t": t, "faces": len(dets),
                         "frontal": sum(1 for d in dets if d[4] == "frontal")})
            if len(dets) > best_n:
                best_n, best = len(dets), draw_boxes(frame, dets, ids)
            samples += 1
            if progress is not None and total_frames:
                progress.progress(min(1.0, idx / total_frames))
        idx += 1
    cap.release()
    if samples == 0:
        raise ValueError("No frames could be read from the video.")
    att = build_attendance(tracker, samples, threshold, eff_sample_s)
    return att, pd.DataFrame(rows), best, rows[-1]["t"] + eff_sample_s


# ----------------------------------------------------------------------------
# Audio analysis
# ----------------------------------------------------------------------------
def load_audio(raw, name):
    try:
        import soundfile as sf
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        try:
            import librosa
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                os.remove(tmp_path)
        except Exception as exc:
            raise ValueError("Unsupported audio file. Please upload WAV / FLAC / OGG.") from exc
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype("float32"), int(sr)


def analyse_audio(y, sr, win_s=1.0):
    from scipy.signal import resample_poly

    factor = max(1, sr // 16000)
    if factor > 1:
        y = resample_poly(y, 1, factor).astype("float32")
        sr = sr // factor
    n = int(sr * win_s)
    nwin = len(y) // n
    if nwin < 2:
        raise ValueError("Audio is too short (need at least 2 seconds).")
    hann = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 300) & (freqs <= 3400)
    rows = []
    for i in range(nwin):
        seg = y[i * n:(i + 1) * n]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        spec = np.abs(np.fft.rfft(seg * hann)) ** 2 + 1e-12
        ratio = float(spec[band].sum() / spec.sum())
        flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
        rows.append((i * win_s, rms, ratio, flat, zcr))
    df = pd.DataFrame(rows, columns=["t", "rms", "speech_band_ratio", "flatness", "zcr"])

    ref = np.percentile(df["rms"], 90) + 1e-9
    silent = (df["rms"] < max(1e-4, 0.12 * ref)).values
    speech = (~silent) & (df["speech_band_ratio"].values > 0.3) & (df["flatness"].values < 0.2)
    noisy = (~silent) & (~speech)

    # bridge 1-window pauses inside speech
    sp = speech.copy()
    for i in range(1, len(sp) - 1):
        if not sp[i] and silent[i] and sp[i - 1] and sp[i + 1]:
            sp[i] = True

    labels = np.array(["Silence"] * len(df), dtype=object)
    labels[noisy] = "High Ambient Distraction"
    for s, e in find_runs(sp):
        lab = "Instructor Lecture" if (e - s) >= 6 else "Student Discussion"
        labels[s:e] = lab
    df["label"] = labels
    return df


# ----------------------------------------------------------------------------
# Fusion + advisory engine
# ----------------------------------------------------------------------------
def fuse(vis, aud, bin_s, w_v, w_a):
    vb = ab = None
    if vis is not None and len(vis):
        v = vis.copy()
        v["bin"] = (v["t"] // bin_s).astype(int)
        v["att"] = np.where(v["faces"] > 0, v["frontal"] / v["faces"].clip(lower=1), 0.0)
        peak = max(float(v["faces"].max()), 1.0)
        v["visual"] = 100 * (0.7 * v["att"] + 0.3 * v["faces"] / peak)
        vb = v.groupby("bin").agg(visual=("visual", "mean"), faces=("faces", "mean"),
                                  attention=("att", "mean"))
    if aud is not None and len(aud):
        a = aud.copy()
        a["bin"] = (a["t"] // bin_s).astype(int)
        a["audio"] = a["label"].map(AUDIO_SCORE)
        ab = a.groupby("bin").agg(audio=("audio", "mean"),
                                  label=("label", lambda s: s.mode().iat[0]))
    parts = [p for p in (vb, ab) if p is not None]
    if not parts:
        return None
    df = pd.concat(parts, axis=1).sort_index()
    for col in ("visual", "audio"):
        if col not in df:
            df[col] = np.nan
    wv = np.where(df["visual"].notna(), w_v, 0.0)
    wa = np.where(df["audio"].notna(), w_a, 0.0)
    denom = np.where((wv + wa) == 0, np.nan, wv + wa)
    df["engagement"] = (df["visual"].fillna(0) * wv + df["audio"].fillna(0) * wa) / denom
    df = df.dropna(subset=["engagement"]).copy()
    df["time_s"] = df.index * bin_s
    df["engagement_smooth"] = df["engagement"].rolling(3, min_periods=1, center=True).mean()
    return df.reset_index(drop=True)


def advise(eng, aud, att, bin_s, low=45, dip_s=120, high=75):
    adv = []
    if eng is None or eng.empty:
        return pd.DataFrame(columns=["Time Range", "Type", "Severity", "Suggestion"])
    need = max(1, int(np.ceil(dip_s / bin_s)))
    low_mask = (eng["engagement_smooth"] < low).values
    for s, e in find_runs(low_mask):
        if e - s < need:
            continue
        seg = eng.iloc[s:e]
        t0, t1 = seg["time_s"].iloc[0], seg["time_s"].iloc[-1] + bin_s
        labs = seg["label"].dropna()
        dom = labs.mode().iat[0] if len(labs) else "Unknown"
        mins = (t1 - t0) / 60
        base = f"Collective engagement stayed below {low}% for {mins:.1f} min. "
        if dom == "Instructor Lecture":
            msg = base + "Long one-way lecture detected: launch a quick poll, think-pair-share or a worked example."
        elif dom == "Silence":
            msg = base + "Prolonged silence: check understanding with a cold-call question or a 2-minute exercise."
        elif dom == "High Ambient Distraction":
            msg = base + "High ambient noise: pause, reset expectations, then switch to a structured group task."
        else:
            msg = base + "Consider a 2-minute stretch break or an interactive discussion prompt."
        adv.append({"Time Range": f"{fmt_t(t0)} - {fmt_t(t1)}", "Type": "Attention Drop",
                    "Severity": "High" if mins >= 8 else "Medium", "Suggestion": msg})

    for s, e in find_runs((eng["engagement_smooth"] >= high).values):
        if e - s >= need:
            seg = eng.iloc[s:e]
            t0, t1 = seg["time_s"].iloc[0], seg["time_s"].iloc[-1] + bin_s
            adv.append({"Time Range": f"{fmt_t(t0)} - {fmt_t(t1)}", "Type": "Interaction Peak",
                        "Severity": "Info",
                        "Suggestion": "Engagement peak. Good window to introduce the hardest concept or a key takeaway."})

    if aud is not None and len(aud):
        share = aud["label"].value_counts(normalize=True)
        lec, dis = share.get("Instructor Lecture", 0), share.get("Student Discussion", 0)
        noise = share.get("High Ambient Distraction", 0)
        if lec > 0.7:
            adv.append({"Time Range": "Whole session", "Type": "Pacing", "Severity": "Medium",
                        "Suggestion": f"Lecture occupied {lec:.0%} of the session. Add interaction checkpoints every 10-15 minutes."})
        if dis < 0.05:
            adv.append({"Time Range": "Whole session", "Type": "Participation", "Severity": "Medium",
                        "Suggestion": "Very little student discussion. Try questions, polls or peer activities."})
        if noise > 0.2:
            adv.append({"Time Range": "Whole session", "Type": "Classroom Management", "Severity": "Medium",
                        "Suggestion": f"Ambient distraction in {noise:.0%} of the session. Review seating, noise sources or activity transitions."})
    if att is not None and len(att):
        rate = (att["Status"] == "Present").mean()
        if rate < 0.75:
            adv.append({"Time Range": "Whole session", "Type": "Attendance", "Severity": "Medium",
                        "Suggestion": f"Sustained-presence rate is {rate:.0%}. Check whether late arrivals or early exits are affecting delivery."})
    if not adv:
        adv.append({"Time Range": "Whole session", "Type": "Summary", "Severity": "Low",
                    "Suggestion": "Engagement was healthy throughout. No intervention needed. Keep the current pacing."})
    return pd.DataFrame(adv)


# ----------------------------------------------------------------------------
# Demo data (works with no uploads - great for the live demo link)
# ----------------------------------------------------------------------------
def make_demo(seed=7):
    rng = np.random.default_rng(seed)
    dur = 30 * 60
    t = np.arange(0, dur, 5.0)
    n_students = 12
    faces = np.clip(np.round(rng.normal(n_students - 1, 0.7, len(t))), 7, n_students)
    base = np.where((t > 600) & (t < 1000), 0.2, 0.86)
    frontal = np.round(faces * np.clip(base + rng.normal(0, 0.06, len(t)), 0.1, 1.0))
    vis = pd.DataFrame({"t": t, "faces": faces, "frontal": frontal})

    labels = np.array(["Instructor Lecture"] * dur, dtype=object)
    labels[0:60] = "Silence"
    labels[400:470] = "Student Discussion"
    labels[600:1000] = "Instructor Lecture"
    for s in rng.choice(np.arange(600, 1000), 150, replace=False):
        labels[s] = "High Ambient Distraction"
    rest = np.setdiff1d(np.arange(600, 1000), np.where(labels == "High Ambient Distraction")[0])
    for s in rng.choice(rest, 120, replace=False):
        labels[s] = "Silence"
    labels[1000:1150] = "Student Discussion"
    labels[1500:1530] = "Silence"
    aud = pd.DataFrame({"t": np.arange(dur, dtype=float), "label": labels})

    rows = []
    for i in range(n_students):
        pct = float(np.clip(rng.normal(0.85, 0.18), 0.15, 1.0))
        if i in (3, 9):
            pct = float(rng.uniform(0.2, 0.55))
        first = 0 if pct > 0.8 else int(rng.integers(30, 600))
        rows.append({
            "Student ID": f"S{i + 1:02d}", "First Seen": fmt_t(first),
            "Last Seen": fmt_t(min(dur, first + pct * dur)),
            "Total Duration (s)": int(pct * dur), "Presence %": round(100 * pct, 1),
            "Facing Forward %": round(float(np.clip(rng.normal(78, 10), 40, 99)), 1),
            "Status": "Present" if pct >= 0.7 else "Absent",
        })
    return pd.DataFrame(rows), vis, aud, dur


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
for key, default in {"att": None, "vis": None, "aud": None, "eng": None, "adv": None,
                     "duration": 0, "source": "none"}.items():
    st.session_state.setdefault(key, default)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.success("🔒 100% offline processing. No video/audio leaves this machine.")
    st.caption("Ethical design: collective pedagogical feedback, not punitive surveillance.")
    st.divider()
    threshold = st.slider("Sustained presence threshold", 0.3, 1.0, 0.7, 0.05,
                          help="Participant must be detected in at least this share of the session.")
    sample_s = st.slider("Video sampling interval (s)", 1, 10, 2)
    bin_s = st.slider("Fusion time window (s)", 5, 60, 10, 5)
    w_v = st.slider("Weight: visual attentiveness", 0.0, 1.0, 0.6, 0.05)
    w_a = round(1.0 - w_v, 2)
    st.caption(f"Weight: audio interaction = {w_a}")
    low_thr = st.slider("Low-engagement threshold (%)", 20, 70, 45)
    dip_s = st.slider("Sustained dip duration (s)", 30, 600, 120, 30)
    st.divider()
    if st.button("🎬 Load full demo session", use_container_width=True):
        att, vis, aud, dur = make_demo()
        st.session_state.update(att=att, vis=vis, aud=aud, duration=dur,
                                eng=None, adv=None, source="demo")
        st.rerun()
    if st.button("🗑️ Reset session", use_container_width=True):
        for k in ("att", "vis", "aud", "eng", "adv"):
            st.session_state[k] = None
        st.session_state.update(duration=0, source="none")
        st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 Attendance Tracking", "🎙️ Acoustic Analysis", "📈 Engagement Fusion",
    "💡 Teaching Suggestions", "📊 Session Analytics",
])

# ----------------------------------------------------------------------------
# Tab 1 - Attendance
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Automated presence-based attendance")
    st.caption("Faces are detected and tracked across sampled frames. Attendance is based on "
               "sustained presence, not a single frame, so hallway pass-throughs are ignored.")
    mode = st.radio("Input", ["Classroom video", "Classroom photo", "Webcam snapshot"], horizontal=True)

    if mode == "Classroom video":
        up = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up and st.button("▶️ Run attendance detection", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
                tmp.write(up.read())
                path = tmp.name
            try:
                bar = st.progress(0.0, text="Analysing video...")
                att, vis, preview, dur = analyse_video(path, sample_s, threshold, bar)
                bar.empty()
                st.session_state.update(att=att, vis=vis, duration=dur, eng=None, adv=None,
                                        source="upload")
                st.session_state["preview"] = preview
                st.success(f"Processed {len(vis)} sampled frames ({fmt_t(dur)} of video).")
            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")
            finally:
                if os.path.exists(path):
                    os.remove(path)
    else:
        img_file = (st.file_uploader("Upload classroom photo", type=["jpg", "jpeg", "png"])
                    if mode == "Classroom photo" else st.camera_input("Take a snapshot"))
        if img_file is not None and st.button("▶️ Run attendance detection", type="primary"):
            try:
                arr = np.frombuffer(img_file.getvalue(), np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Could not decode the image.")
                dets = detect_faces(frame)
                tr = CentroidTracker()
                ids = tr.update(dets, 0.0)
                att = build_attendance(tr, 1, 0.0, 1.0)
                if not att.empty:
                    att["Total Duration (s)"] = "single frame"
                    att["Last Seen"] = "single frame"
                vis = pd.DataFrame([{"t": 0.0, "faces": len(dets),
                                     "frontal": sum(1 for d in dets if d[4] == "frontal")}])
                st.session_state.update(att=att, vis=vis, duration=0, eng=None, adv=None,
                                        source="upload")
                st.session_state["preview"] = draw_boxes(frame, dets, ids)
                st.info("Single image: every detected face is marked Present. "
                        "Use a video for true sustained-presence logic.")
            except Exception as exc:
                st.error(f"Image analysis failed: {exc}")

    att = st.session_state["att"]
    if att is not None and len(att):
        c1, c2, c3 = st.columns(3)
        n_present = int((att["Status"] == "Present").sum())
        c1.metric("Participants tracked", len(att))
        c2.metric("Present (sustained)", n_present)
        c3.metric("Attendance rate", f"{n_present / len(att):.0%}")
        left, right = st.columns([1, 1])
        with left:
            if st.session_state.get("preview") is not None and st.session_state["source"] == "upload":
                st.image(st.session_state["preview"], caption="Detected participants (green = facing forward)",
                         use_container_width=True)
            else:
                st.caption("Load a video/photo to see detection preview.")
        with right:
            if pd.api.types.is_numeric_dtype(att["Presence %"]) and st.session_state["duration"] > 0:
                fig, ax = plt.subplots(figsize=(5, 3.2))
                colors = ["#16a34a" if s == "Present" else "#dc2626" for s in att["Status"]]
                ax.barh(att["Student ID"], att["Presence %"], color=colors)
                ax.axvline(threshold * 100, color="#111", ls="--", lw=1)
                ax.set_xlabel("Presence % of session")
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close(fig)
        st.dataframe(att, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download attendance register (CSV)", att.to_csv(index=False),
                           "attendance_register.csv", "text/csv")
    else:
        st.info("No attendance data yet. Upload media or click **Load full demo session** in the sidebar.")

# ----------------------------------------------------------------------------
# Tab 2 - Audio
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Acoustic interaction & noise dynamics")
    st.caption("1-second windows are classified using spectral energy, speech-band ratio and spectral "
               "flatness (voice-activity detection). Run lengths separate lectures from discussion bursts.")
    up_a = st.file_uploader("Upload classroom audio (WAV / FLAC / OGG)", type=["wav", "flac", "ogg", "mp3"])
    if up_a and st.button("▶️ Analyse audio", type="primary"):
        try:
            y, sr = load_audio(up_a.read(), up_a.name)
            with st.spinner("Processing audio windows..."):
                aud = analyse_audio(y, sr)
            st.session_state.update(aud=aud, eng=None, adv=None, source="upload")
            if st.session_state["duration"] == 0:
                st.session_state["duration"] = float(aud["t"].max() + 1)
        except Exception as exc:
            st.error(f"Audio analysis failed: {exc}")

    aud = st.session_state["aud"]
    if aud is not None and len(aud):
        share = aud["label"].value_counts(normalize=True)
        cols = st.columns(4)
        for col, lab in zip(cols, LABELS):
            col.metric(lab, f"{share.get(lab, 0):.0%}")
        fig, ax = plt.subplots(figsize=(10, 2.4))
        for lab in LABELS:
            sub = aud[aud["label"] == lab]
            ax.scatter(sub["t"] / 60, [lab] * len(sub), s=8, c=LABEL_COLORS[lab], marker="|")
        ax.set_xlabel("Minutes")
        ax.set_title("Audio activity timeline")
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Raw window-level table"):
            st.dataframe(aud, use_container_width=True, hide_index=True)
    else:
        st.info("No audio data yet. Upload a file or load the demo session from the sidebar.")

# ----------------------------------------------------------------------------
# Tab 3 - Fusion
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Multimodal fusion & engagement scoring")
    st.caption("Visual attentiveness (forward-facing share + presence) and audio interaction energy are "
               "merged into one normalised Engagement Index (0-100).")
    if st.button("⚡ Compute engagement timeline", type="primary"):
        eng = fuse(st.session_state["vis"], st.session_state["aud"], bin_s, w_v, w_a)
        if eng is None or eng.empty:
            st.warning("Provide video/photo and/or audio data first.")
        else:
            st.session_state["eng"] = eng
            st.session_state["adv"] = None

    eng = st.session_state["eng"]
    if eng is not None and len(eng):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average engagement", f"{eng['engagement'].mean():.1f}")
        c2.metric("Peak", f"{eng['engagement_smooth'].max():.1f}")
        c3.metric("Lowest", f"{eng['engagement_smooth'].min():.1f}")
        c4.metric("Windows", len(eng))
        fig, ax = plt.subplots(figsize=(10, 3.6))
        x = eng["time_s"] / 60
        ax.plot(x, eng["engagement_smooth"], color="#2563eb", lw=2, label="Engagement Index")
        if eng["visual"].notna().any():
            ax.plot(x, eng["visual"], color="#16a34a", lw=1, alpha=.6, label="Visual")
        if eng["audio"].notna().any():
            ax.plot(x, eng["audio"], color="#f59e0b", lw=1, alpha=.6, label="Audio")
        ax.axhline(low_thr, color="#dc2626", ls="--", lw=1, label=f"Low threshold ({low_thr})")
        ax.fill_between(x, 0, 100, where=eng["engagement_smooth"] < low_thr, color="#dc2626", alpha=.1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Score")
        ax.legend(loc="lower right", ncol=4, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        show = eng[["time_s", "visual", "audio", "label", "engagement"]].copy()
        show["time_s"] = show["time_s"].map(fmt_t)
        st.dataframe(show.round(1), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download engagement timeline (CSV)", eng.to_csv(index=False),
                           "engagement_timeline.csv", "text/csv")
    else:
        st.info("Click **Compute engagement timeline** once data is available.")

# ----------------------------------------------------------------------------
# Tab 4 - Suggestions
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("AI teaching suggestion engine")
    st.caption("A heuristic advisory agent. It never enforces decisions; it only supports the educator.")
    if st.button("🧠 Generate teaching advisories", type="primary"):
        if st.session_state["eng"] is None:
            eng = fuse(st.session_state["vis"], st.session_state["aud"], bin_s, w_v, w_a)
            st.session_state["eng"] = eng
        st.session_state["adv"] = advise(st.session_state["eng"], st.session_state["aud"],
                                         st.session_state["att"], bin_s, low_thr, dip_s)
    adv = st.session_state["adv"]
    if adv is not None and len(adv):
        for _, r in adv.iterrows():
            st.markdown(
                f"<span class='pill sev-{r['Severity']}'>{r['Severity']}</span> "
                f"**{r['Type']}** &nbsp;·&nbsp; `{r['Time Range']}`",
                unsafe_allow_html=True,
            )
            st.write(r["Suggestion"])
            st.divider()
    else:
        st.info("No advisories yet. Load data, then click **Generate teaching advisories**.")

# ----------------------------------------------------------------------------
# Tab 5 - Analytics
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Session analytics & educator report")
    if st.button("📝 Generate session report", type="primary"):
        if st.session_state["eng"] is None:
            st.session_state["eng"] = fuse(st.session_state["vis"], st.session_state["aud"],
                                           bin_s, w_v, w_a)
        if st.session_state["adv"] is None:
            st.session_state["adv"] = advise(st.session_state["eng"], st.session_state["aud"],
                                             st.session_state["att"], bin_s, low_thr, dip_s)
        st.session_state["report_ready"] = True

    if st.session_state.get("report_ready"):
        att, eng = st.session_state["att"], st.session_state["eng"]
        aud, adv = st.session_state["aud"], st.session_state["adv"]
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "session_duration": fmt_t(st.session_state["duration"]),
            "privacy": "Offline processing; no raw media stored or transmitted.",
        }
        if att is not None and len(att):
            report["attendance"] = {
                "participants": int(len(att)),
                "present": int((att["Status"] == "Present").sum()),
                "attendance_rate_pct": round(100 * float((att["Status"] == "Present").mean()), 1),
            }
        if eng is not None and len(eng):
            report["engagement"] = {
                "average": round(float(eng["engagement"].mean()), 1),
                "peak": round(float(eng["engagement_smooth"].max()), 1),
                "lowest": round(float(eng["engagement_smooth"].min()), 1),
                "peak_time": fmt_t(eng.loc[eng["engagement_smooth"].idxmax(), "time_s"]),
                "lowest_time": fmt_t(eng.loc[eng["engagement_smooth"].idxmin(), "time_s"]),
            }
        if aud is not None and len(aud):
            report["audio_distribution_pct"] = {
                k: round(100 * float(v), 1) for k, v in aud["label"].value_counts(normalize=True).items()}
            report["participation_rate_pct"] = round(
                100 * float((aud["label"] == "Student Discussion").mean()), 1)
        if adv is not None:
            report["advisories"] = adv.to_dict(orient="records")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Attendance rate", f"{report.get('attendance', {}).get('attendance_rate_pct', '-')}%")
        k2.metric("Avg engagement", report.get("engagement", {}).get("average", "-"))
        k3.metric("Participation", f"{report.get('participation_rate_pct', '-')}%")
        k4.metric("Advisories", len(report.get("advisories", [])))

        if aud is not None and len(aud):
            fig, ax = plt.subplots(figsize=(4.5, 3))
            share = aud["label"].value_counts()
            ax.pie(share.values, labels=share.index, autopct="%1.0f%%", startangle=90,
                   colors=[LABEL_COLORS[l] for l in share.index], textprops={"fontsize": 8})
            ax.set_title("Session audio composition", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

        lines = ["MULTIMODAL ATTENDANCE & ENGAGEMENT REPORT",
                 f"Generated: {report['generated_at']}",
                 f"Session duration: {report['session_duration']}", ""]
        for sec in ("attendance", "engagement", "audio_distribution_pct"):
            if sec in report:
                lines.append(sec.replace("_", " ").upper())
                lines += [f"  - {k}: {v}" for k, v in report[sec].items()]
                lines.append("")
        lines.append("TEACHING ADVISORIES")
        for a in report.get("advisories", []):
            lines.append(f"  [{a['Severity']}] {a['Time Range']} - {a['Type']}: {a['Suggestion']}")
        text = "\n".join(lines)
        st.text_area("Report preview", text, height=280)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download report (TXT)", text, "session_report.txt", "text/plain")
        d2.download_button("⬇️ Download report (JSON)", json.dumps(report, indent=2),
                           "session_report.json", "application/json")
    else:
        st.info("Click **Generate session report** to build the educator summary.")

st.caption("Skill Wallet Capstone Project 5 · Multimodal Attendance & Engagement Agent · Offline & privacy-first")
