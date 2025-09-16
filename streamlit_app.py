import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

# Load engineering guidelines (fallback defaults)
@st.cache_data
def load_guidelines():
    default = {
        "dynamic_eq": {"fabfilter_f6": {"attack_ms": [1, 10], "release_ms": [50, 250]}},
        "compressor": {"neve_33609": {"attack_ms": [10, 30], "release_ms": [100, 300]}},
        "deesser": {"default": {"threshold_db": [-20, -10], "ratio": [2, 6]}}
    }
    try:
        with open('engineer_guidelines.json') as f:
            data = json.load(f)
            for k, v in default.items():
                data.setdefault(k, v)
            return data
    except FileNotFoundError:
        return default


guidelines = load_guidelines()


def ms_to_subdivision(ms, bpm):
    beat_ms = 60000 / bpm
    subs = {'1/4': beat_ms, '1/8': beat_ms / 2, '1/16': beat_ms / 4, '1/32': beat_ms / 8, '1/64': beat_ms / 16}
    name, val = min(subs.items(), key=lambda x: abs(x[1] - ms))
    return name, int(val)


st.title("🎤 Reference-Matched Engineer Vocal Chain Analyzer")
st.write("Upload dry vocal + reference WAV; get detailed, BPM-synced vocal mix chain with all plugin parameters")

# Upload inputs
dry_file = st.file_uploader("Dry Vocal (WAV)", type='wav')
ref_file = st.file_uploader("Reference Track (WAV)", type='wav')

# Session defaults
if 'chain_ready' not in st.session_state:
    st.session_state.chain_ready = False
if 'chain_df' not in st.session_state:
    st.session_state.chain_df = pd.DataFrame()
if 'ref_bpm' not in st.session_state:
    st.session_state.ref_bpm = 120
if 'ref_key' not in st.session_state:
    st.session_state.ref_key = "Unknown"

# Key/BPM detection + manual override UI
st.subheader("🔑 Key & 🎵 Tempo Detection (Auto + Manual)")


def detect_key_bpm(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True, duration=30)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][chroma.mean(axis=1).argmax()]
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return key, int(bpm)
    except:
        return "Unknown", 120


if dry_file and 'auto_dry' not in st.session_state:
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(dry_file.read())
    tmp.close()
    st.session_state.auto_dry = detect_key_bpm(tmp.name)
    os.unlink(tmp.name)

if ref_file and 'auto_ref' not in st.session_state:
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(ref_file.read())
    tmp.close()
    st.session_state.auto_ref = detect_key_bpm(tmp.name)
    os.unlink(tmp.name)

auto_dry_key, auto_dry_bpm = st.session_state.auto_dry if 'auto_dry' in st.session_state else ("Unknown", 120)
auto_ref_key, auto_ref_bpm = st.session_state.auto_ref if 'auto_ref' in st.session_state else ("Unknown", 120)

st.session_state.edit_dry = st.checkbox("Edit Dry Key/BPM?", key="chk_dry")
if st.session_state.edit_dry:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.manual_dry_key = st.text_input("Dry Key", value=st.session_state.get('manual_dry_key', auto_dry_key))
    with col2:
        st.session_state.manual_dry_bpm = st.number_input("Dry BPM", min_value=0, max_value=300,
                                                        value=st.session_state.get('manual_dry_bpm', auto_dry_bpm))
    dry_key, dry_bpm = st.session_state.manual_dry_key, st.session_state.manual_dry_bpm
else:
    dry_key, dry_bpm = auto_dry_key, auto_dry_bpm

st.session_state.edit_ref = st.checkbox("Edit Ref Key/BPM?", key="chk_ref")
if st.session_state.edit_ref:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.manual_ref_key = st.text_input("Ref Key", value=st.session_state.get('manual_ref_key', auto_ref_key))
    with col2:
        st.session_state.manual_ref_bpm = st.number_input("Ref BPM", min_value=0, max_value=300,
                                                         value=st.session_state.get('manual_ref_bpm', auto_ref_bpm))
    ref_key, ref_bpm = st.session_state.manual_ref_key, st.session_state.manual_ref_bpm
else:
    ref_key, ref_bpm = auto_ref_key, auto_ref_bpm

st.session_state.ref_key = ref_key
st.session_state.ref_bpm = ref_bpm

st.write(f"• Dry → Key: **{dry_key}**, BPM: **{dry_bpm}**")
st.write(f"• Ref →
