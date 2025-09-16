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

# Upload files
dry_file = st.file_uploader("Upload Dry Vocal (WAV)", type='wav')
ref_file = st.file_uploader("Upload Reference Track (WAV)", type='wav')

# Prepare session state for manual overrides
if 'edit_dry' not in st.session_state:
    st.session_state.edit_dry = False
if 'edit_ref' not in st.session_state:
    st.session_state.edit_ref = False

if dry_file and ref_file and st.button("Analyze Vocals"):
    # Save temp files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t1:
        t1.write(dry_file.read()); dry_path = t1.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t2:
        t2.write(ref_file.read()); ref_path = t2.name

    # Phase 1: Analysis & Diagnosis
    st.subheader("🔑 Key & 🎵 Tempo Detection")

    def detect(path):
        try:
            y, sr = librosa.load(path, sr=None, mono=True, duration=30)
            chroma = librosa.feature.chroma_cens(y=y, sr=sr)
            key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][chroma.mean(axis=1).argmax()]
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            return key, int(bpm)
        except:
            return "Unknown", 120

    auto_dry_key, auto_dry_bpm = detect(dry_path)
    auto_ref_key, auto_ref_bpm = detect(ref_path)

    # Dry vocal detection & override
    st.markdown(f"**Dry Vocal** → Key: `{auto_dry_key}` | BPM: `{auto_dry_bpm}`")
    st.session_state.edit_dry = st.checkbox("Edit Dry Key/BPM?", value=st.session_state.edit_dry, key="chk_dry")
    if st.session_state.edit_dry:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.manual_dry_key = st.text_input(
                "Dry Key", value=st.session_state.get('manual_dry_key', auto_dry_key), key="ti_dry_key"
            )
        with col2:
            st.session_state.manual_dry_bpm = st.number_input(
                "Dry BPM", min_value=0, max_value=300,
                value=st.session_state.get('manual_dry_bpm', auto_dry_bpm),
                key="ni_dry_bpm"
            )
        dry_key = st.session_state.manual_dry_key
        dry_bpm = st.session_state.manual_dry_bpm
    else:
        dry_key, dry_bpm = auto_dry_key, auto_dry_bpm

    # Reference vocal detection & override
    st.markdown(f"**Reference** → Key: `{auto_ref_key}` | BPM: `{auto_ref_bpm}`")
    st.session_state.edit_ref = st.checkbox("Edit Ref Key/BPM?", value=st.session_state.edit_ref, key="chk_ref")
    if st.session_state.edit_ref:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.manual_ref_key = st.text_input(
                "Ref Key", value=st.session_state.get('manual_ref_key', auto_ref_key), key="ti_ref_key"
            )
        with col2:
            st.session_state.manual_ref_bpm = st.number_input(
                "Ref BPM", min_value=0, max_value=300,
                value=st.session_state.get('manual_ref_bpm', auto_ref_bpm),
                key="ni_ref_bpm"
            )
        ref_key = st.session_state.manual_ref_key
        ref_bpm = st.session_state.manual_ref_bpm
    else:
        ref_key, ref_bpm = auto_ref_key, auto_ref_bpm

    st.write(f"• Dry → Key: **{dry_key}**, BPM: **{dry_bpm}**")
    st.write(f"• Ref → Key: **{ref_key}**, BPM: **{ref_bpm}**")

    # Spectral analysis
    seg_start, seg_end = 0.33, 0.66
    def spectrum(path):
        sr, data = wavfile.read(path)
        if data.ndim > 1: data = data.mean(1)
        seg = data[int(len(data)*seg_start):int(len(data)*seg_end)]
        w = seg * windows.hann(len(seg))
        fft = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(seg), 1/sr)
        return freqs, np.abs(fft)

    freqs, mag_dry = spectrum(dry_path)
    _, mag_ref = spectrum(ref_path)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mag_dry, name="Dry", line_color='royalblue'))
    fig.add_trace(go.Scatter(x=freqs, y=mag_ref, name="Ref", line_color='orange'))
    fig.update_layout(title="Spectrum (Dry vs Ref)", xaxis_type='log', yaxis_title='Magnitude')
    st.plotly_chart(fig, use_container_width=True)

    # Identify surgical peaks
    bands = {"Low": (20,600), "Mid": (600,1200), "High": (1200,12000)}
    peaks = []
    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        bm, bf = mag_dry[idx], freqs[idx]
        p, props = find_peaks(bm, height=bm.max()*0.05)
        if not p.size: continue
        top = p[np.argsort(props["peak_heights"])[-3:]]
        for i in top:
            cf = bf[i]
            half = bm[i] / np.sqrt(2)
            l = i
            while l > 0 and bm[l] > half: l -= 1
            r = i
            while r < len(bm)-1 and bm[r] > half: r += 1
            bw = bf[r] - bf[l] if r < len(bf) and l >= 0 else cf/10
            Q = cf / bw if bw > 0 else 10
            cut = np.clip(10 * np.log10(bm[i] / np.median(bm)), 1, 8)
            peaks.append({"Band": band, "Center": cf, "Q": Q, "Cut": cut})

    df = pd.DataFrame(peaks)
    st.subheader("🔧 Surgical Notches")
    edits = []
    for i, row in df.iterrows():
        st.markdown(f"**{row['Band']} @ {row['Center']:.1f} Hz**")
        c = st.slider("Cut dB", 1.0, 8.0, value=float(row["Cut"]), step=0.1, key=f"c{i}")
        q = st.slider("Q", 1.0, 10.0, value=float(row["Q"]), step=0.1, key=f"q{i}")
        edits.append({"Band": row["Band"], "Center": row["Center"], "Q": q, "Cut": c})
    df_ed = pd.DataFrame(edits)
    st.dataframe(df_ed)

    # Phase 2 & 3: Iterative Mix Chain
    if st.button("🔧 Build Iterative Mix Chain"):
        s = df_ed.copy()
        s["P1"] = (s["Cut"] * 0.10).round(2)
        s["P2"] = (s["Cut"] * 0.20).round(2)
        s["P3"] = (s["Cut"] * 0.10).round(2)
        s["P4"] = (s["Cut"] * 0.30).round(2)
        s["P5"] = (s["Cut"] * 0.20).round(2)
        s["P6"] = (s["Cut"] * 0.10).round(2)

        st.subheader("Phase 2: Subtractive Passes")
        st.write("**Pass A (5–10% broad cuts)**: Broad-Q shelf/notch")
        for _, r in s.iterrows():
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q 1.5, Cut {r['P1']} dB")

        st.write("**Pass B (10–20% narrow notches)**")
        for _, r in s.iterrows():
            q_val = 3.0 if r['Band'] == "Low" else 4.0
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q {q_val}, Cut {r['P2']} dB")

        st.write("**Pass C (10–20% dynamic EQ)**")
        for _, r in s.iterrows():
            atk_sub, atk_ms = ms_to_subdivision(5, ref_bpm)
            rel_sub, rel_ms = ms_to_subdivision(120, ref_bpm)
            st.write(f"• {r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB reduction, Ratio 3:1")
            st.write(f"  Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms)")

        st.subheader("Pass D (Serial Compression)")
        atk_sub, atk_ms = ms_to_subdivision(15, ref_bpm)
        rel_sub, rel_ms = ms_to_subdivision(250, ref_bpm)
        st.write(f"• Comp A 4:1 — Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms), GR ~2 dB")
        st.write(f"• Comp B 2.5:1 — Attack ~{atk_sub} ({atk_ms} ms), Release ~{rel_sub} ({rel_ms} ms), GR ~2 dB")

        st.subheader("Phase 3: Additive Passes")
        st.write("**Pass E (Shelving & Parallel Sat 5–10%)**")
        for _, r in s.iterrows():
            st.write(f"• Parallel Notch @ {r['Center']:.1f} Hz — Q 0.7, Cut {r['P5']} dB")
        st.write("• High-Shelf @ 10 kHz — +1.5 dB, Q 0.7")
        st.write("• Parallel Saturation Bus (HP 800 Hz, LP 12 kHz) at 10% blend")

        st.write("**Pass F (Final Touches)**")
        th_min, th_max = guidelines['deesser']['default']['threshold_db']
        ratio_min, ratio_max = guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser 5–8 kHz — Threshold {th_min} to {th_max} dB, Ratio {ratio_min}:{ratio_max}")
        st.write("• Manual clip-gain rides")
        st.write(f"• A/B vs reference (Key: {ref_key}, BPM: {ref_bpm})")

    # Cleanup
    os.unlink(dry_path)
    os.unlink(ref_path)
