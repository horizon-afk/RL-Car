import streamlit as st
import numpy as np
import torch
import gymnasium as gym
import os
import time
from collections import deque

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarRacing Agent",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
}

/* Dark racing aesthetic */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.mono {
    font-family: 'Share Tech Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10101a;
    border-right: 1px solid #2a2a3a;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, #12121e 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-left: 3px solid #e63946;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #aab;
}

.score-card .label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #556;
    margin-bottom: 2px;
}

.score-card .value {
    font-size: 1.6rem;
    color: #e63946;
    font-weight: bold;
}

/* Buttons */
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: #e63946;
    color: #fff;
    border: none;
    border-radius: 2px;
    padding: 0.5rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #c1121f;
}

/* Selectbox */
.stSelectbox label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #556 !important;
}

/* Slider */
.stSlider label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #556 !important;
}

/* Episode log */
.episode-log {
    background: #0d0d18;
    border: 1px solid #1e1e30;
    padding: 0.8rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #667;
    max-height: 160px;
    overflow-y: auto;
}

.episode-log .ep-line {
    color: #8899aa;
    border-bottom: 1px solid #1a1a28;
    padding: 3px 0;
}

.episode-log .ep-line span {
    color: #e63946;
}

/* Title stripe */
.title-stripe {
    background: linear-gradient(90deg, #e63946 0%, #c1121f 60%, #0a0a0f 100%);
    height: 3px;
    margin-bottom: 1.5rem;
}

/* Status pill */
.status-pill {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.status-running { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
.status-idle    { background: #1a1a1a; color: #556;    border: 1px solid #2a2a3a; }
.status-done    { background: #1a2a3a; color: #4a9eff; border: 1px solid #4a9eff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def rgb2gray(rgb, norm=True):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        gray = gray / 128. - 1.
    return gray


IMG_STACK = 4
ACTION_REPEAT = 10


class Wrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb, _ = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * IMG_STACK
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(ACTION_REPEAT):
            img_rgb, reward, die, _, _ = self.env.step(action)
            if die:
                reward += 100
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            done = self.av_r(reward) <= -0.1
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        return np.array(self.stack), total_reward, done, die

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        return memory


@st.cache_resource
def load_env():
    return gym.make('CarRacing-v3', render_mode='rgb_array')


def load_agent(weight_path: str):
    """Import Agent lazily so missing file gives a clean error."""
    from agent import Agent
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    agent = Agent(device)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    agent.net.load_state_dict(state_dict)
    agent.net.float().to(device)
    return agent, device


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ CarRacing\nPPO Agent")
    st.markdown('<div class="title-stripe"></div>', unsafe_allow_html=True)

    WEIGHT_DIR = "dir_chk"
    weight_options = {
        "Score 350–550  (early)":   "model_weights_350-550.pth",
        "Score 480–660  (mid)":     "model_weights_480-660.pth",
        "Score 820–980  (expert)":  "model_weights_820-980.pth",
    }
    selected_label = st.selectbox("Model checkpoint", list(weight_options.keys()))
    weight_file = weight_options[selected_label]
    weight_path = os.path.join(WEIGHT_DIR, weight_file)

    n_episodes = st.slider("Episodes to run", 1, 10, 3)
    fps = st.slider("Playback speed (fps)", 5, 60, 30)

    st.markdown("---")
    run = st.button("▶  Run Agent")
    stop = st.button("⏹  Stop")

    st.markdown("---")
    st.markdown("""
<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#445;line-height:1.8'>
PPO · Beta policy<br>
CarRacing-v3<br>
CPU inference<br>
frame skip × 10
</div>
""", unsafe_allow_html=True)


# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown("# Watch the Agent Drive")
st.markdown('<div class="title-stripe"></div>', unsafe_allow_html=True)

status_ph  = st.empty()
frame_ph   = st.empty()
metrics_ph = st.empty()
log_ph     = st.empty()

if "episode_log" not in st.session_state:
    st.session_state.episode_log = []
if "running" not in st.session_state:
    st.session_state.running = False

if stop:
    st.session_state.running = False

if run:
    if not os.path.exists(weight_path):
        st.error(f"Weight file not found: `{weight_path}`\n\nMake sure `{WEIGHT_DIR}/` is next to `app.py`.")
        st.stop()

    st.session_state.running = True
    st.session_state.episode_log = []

    try:
        agent, device = load_agent(weight_path)
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        st.stop()

    env  = load_env()
    wrap = Wrapper(env)

    scores_deque = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        if not st.session_state.running:
            break

        state     = wrap.reset()
        score     = 0.0
        t_start   = time.time()
        step_num  = 0

        status_ph.markdown(
            f'<div class="status-pill status-running">● EPISODE {ep} / {n_episodes} &nbsp;·&nbsp; RUNNING</div>',
            unsafe_allow_html=True,
        )

        while st.session_state.running:
            action, _ = agent.select_action(state)
            frame = env.render()

            # Display frame
            frame_ph.image(frame, use_container_width=True)
            time.sleep(1 / fps)

            # Step
            scaled = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            state, reward, done, die = wrap.step(scaled)
            score    += reward
            step_num += 1

            # Live metrics row
            elapsed = int(time.time() - t_start)
            metrics_ph.markdown(f"""
<div style='display:flex;gap:1rem;margin-bottom:1rem'>
  <div class="score-card" style='flex:1'>
    <div class="label">Episode Score</div>
    <div class="value">{score:+.1f}</div>
  </div>
  <div class="score-card" style='flex:1'>
    <div class="label">Steps</div>
    <div class="value">{step_num}</div>
  </div>
  <div class="score-card" style='flex:1'>
    <div class="label">Elapsed</div>
    <div class="value">{elapsed:02d}s</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if done or die:
                break

        # Episode done
        scores_deque.append(score)
        elapsed = int(time.time() - t_start)
        log_entry = (
            f"EP {ep:02d} &nbsp;·&nbsp; "
            f"score <span>{score:+.1f}</span> &nbsp;·&nbsp; "
            f"avg <span>{np.mean(scores_deque):.1f}</span> &nbsp;·&nbsp; "
            f"{elapsed}s"
        )
        st.session_state.episode_log.insert(0, log_entry)

        lines = "".join(
            f'<div class="ep-line">{l}</div>'
            for l in st.session_state.episode_log
        )
        log_ph.markdown(
            f'<div class="episode-log">{lines}</div>',
            unsafe_allow_html=True,
        )

    st.session_state.running = False
    status_ph.markdown(
        '<div class="status-pill status-done">✓ DONE</div>',
        unsafe_allow_html=True,
    )

else:
    status_ph.markdown(
        '<div class="status-pill status-idle">○ IDLE — select a checkpoint and press Run</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.episode_log:
        lines = "".join(
            f'<div class="ep-line">{l}</div>'
            for l in st.session_state.episode_log
        )
        log_ph.markdown(
            f'<div class="episode-log">{lines}</div>',
            unsafe_allow_html=True,
        )
