import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

# Load engineering guidelines (with fallback defaults)
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
            for section in default:
                if section not in data:
                    data[section] = default[section]
            return data
    except FileNotFoundError:
        return default

guidelines = load_guidelines()

def ms_to_subdivision(ms, bpm):
    beat_ms = 60000 / bpm
    subdivisions = {
        '1/4': beat_ms,
        '1/8': beat_ms/2,
        '1/16': beat_ms/4,
        '1/32': beat_ms/8,
        '1/64': beat_ms/16
    }
    name, val = min(subdivisions.items(), key=lambda x: abs(x[1] - ms))
    return name, int(val)

st.title("🎤 Advanced Vocal EQ Analyzer")
st.write("Iterative subtractive-additive workflow with musically-aware, engineer-grade recommendations")

# 1) Upload files
dry_file = st.file_uploader("Upload Dry Vocal (WAV)", type='wav')
ref_file = st.file_uploader("Upload Reference Track (WAV)", type='wav')

# Prepare session state for chain builder
if 'chain_ready' not in st.session_state:
    st.session_state.chain_ready = False
if 'chain_df' not in st.session_state:
    st.session_state.chain_df = None

# 2) Pre-Analysis Key/BPM UI (always rendered) — unchanged from before
# … (same auto/manual key bpm code) …

# 3) Analyze when button clicked
if dry_file and ref_file and st.button("Analyze Vocals"):
    # Save temp files, do spectrum and surgical df_ed — as before …
    # After df_ed created:
    st.session_state.chain_df = df_ed.copy()
    st.session_state.chain_ready = True

# 4) Chain builder outside analysis block
if st.session_state.chain_ready:
    st.subheader("🔧 Build Iterative Mix Chain")
    if st.button("Generate Mix Chain Instructions"):
        s = st.session_state.chain_df.copy()
        s["P1"] = (s["Cut"] * 0.10).round(2)
        s["P2"] = (s["Cut"] * 0.20).round(2)
        s["P3"] = (s["Cut"] * 0.10).round(2)
        s["P4"] = (s["Cut"] * 0.30).round(2)
        s["P5"] = (s["Cut"] * 0.20).round(2)
        s["P6"] = (s["Cut"] * 0.10).round(2)

        st.markdown("**Phase 2: Subtractive Passes**")
        st.write("**Pass A (5–10% broad cuts)**")
        for _, r in s.iterrows():
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q 1.5, Cut {r['P1']} dB")

        st.write("**Pass B (10–20% narrow notches)**")
        for _, r in s.iterrows():
            qv = 3.0 if r['Band']=="Low" else 4.0
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q {qv}, Cut {r['P2']} dB")

        st.write("**Pass C (10–20% dynamic EQ)**")
        for _, r in s.iterrows():
            atk_sub, atk_ms = ms_to_subdivision(5, ref_bpm)
            rel_sub, rel_ms = ms_to_subdivision(120, ref_bpm)
            st.write(f"• {r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB reduction, Ratio 3:1")
            st.write(f"  Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms)")

        st.markdown("**Pass D (Serial Compression)**")
        atk_sub, atk_ms = ms_to_subdivision(15, ref_bpm)
        rel_sub, rel_ms = ms_to_subdivision(250, ref_bpm)
        st.write(f"• Comp A 4:1 — Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms), GR ~2 dB")
        st.write(f"• Comp B 2.5:1 — Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms), GR ~2 dB")

        st.markdown("**Phase 3: Additive Passes**")
        st.write("**Pass E (Shelving & Parallel Sat 5–10%)**")
        for _, r in s.iterrows():
            st.write(f"• Parallel Notch @ {r['Center']:.1f} Hz — Q 0.7, Cut {r['P5']} dB")
        st.write("• High-Shelf @ 10 kHz — +1.5 dB, Q 0.7")
        st.write("• Parallel Saturation Bus (HP 800 Hz, LP 12 kHz) at 10% blend")

        st.write("**Pass F (Final Touches)**")
        th_min, th_max = guidelines['deesser']['default']['threshold_db']
        r_min, r_max = guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser 5–8 kHz — Threshold {th_min} to {th_max} dB, Ratio {r_min}:{r_max}")
        st.write("• Manual clip-gain rides")
        st.write(f"• A/B vs reference (Key: {ref_key}, BPM: {ref_bpm})")
