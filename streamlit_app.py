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
        "dynamic_eq": {"fabfilter_f6": {"attack_ms":[1,10],"release_ms":[50,250]}},
        "compressor": {"neve_33609": {"attack_ms":[10,30],"release_ms":[100,300]}},
        "deesser": {"default": {"threshold_db":[-20,-10],"ratio":[2,6]}}
    }
    try:
        with open('engineer_guidelines.json') as f:
            data = json.load(f)
            for k,v in default.items():
                data.setdefault(k, v)
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
st.write("Automated engineer-ready mix chain formatting")

# Upload audio
dry_file = st.file_uploader("Dry Vocal (WAV)", type='wav')
ref_file = st.file_uploader("Reference Track (WAV)", type='wav')

# Init session state defaults
if 'chain_ready' not in st.session_state: st.session_state.chain_ready = False
if 'chain_df' not in st.session_state: st.session_state.chain_df = pd.DataFrame()
if 'ref_bpm' not in st.session_state: st.session_state.ref_bpm = 120
if 'ref_key' not in st.session_state: st.session_state.ref_key = "Unknown"

# Key and BPM detection & manual override UI
st.subheader("🔑 Key & 🎵 Tempo Detection (Auto + Manual)")

def detect(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True, duration=30)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][chroma.mean(axis=1).argmax()]
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return key, int(bpm)
    except:
        return "Unknown", 120

# Auto-detect on upload
if dry_file and 'auto_dry' not in st.session_state:
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(dry_file.read())
    tmp.close()
    st.session_state.auto_dry = detect(tmp.name)
    os.unlink(tmp.name)
if ref_file and 'auto_ref' not in st.session_state:
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(ref_file.read())
    tmp.close()
    st.session_state.auto_ref = detect(tmp.name)
    os.unlink(tmp.name)

auto_dry_key, auto_dry_bpm = st.session_state.auto_dry if 'auto_dry' in st.session_state else ("Unknown", 120)
auto_ref_key, auto_ref_bpm = st.session_state.auto_ref if 'auto_ref' in st.session_state else ("Unknown", 120)

# Manual override checkboxes
st.session_state.edit_dry = st.checkbox("Edit Dry Key/BPM?", key="chk_dry")
if st.session_state.edit_dry:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.manual_dry_key = st.text_input("Dry Key", value=st.session_state.get('manual_dry_key', auto_dry_key))
    with col2:
        st.session_state.manual_dry_bpm = st.number_input("Dry BPM", min_value=0, max_value=300, value=st.session_state.get('manual_dry_bpm', auto_dry_bpm))
    dry_key, dry_bpm = st.session_state.manual_dry_key, st.session_state.manual_dry_bpm
else:
    dry_key, dry_bpm = auto_dry_key, auto_dry_bpm

st.session_state.edit_ref = st.checkbox("Edit Ref Key/BPM?", key="chk_ref")
if st.session_state.edit_ref:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.manual_ref_key = st.text_input("Ref Key", value=st.session_state.get('manual_ref_key', auto_ref_key))
    with col2:
        st.session_state.manual_ref_bpm = st.number_input("Ref BPM", min_value=0, max_value=300, value=st.session_state.get('manual_ref_bpm', auto_ref_bpm))
    ref_key, ref_bpm = st.session_state.manual_ref_key, st.session_state.manual_ref_bpm
else:
    ref_key, ref_bpm = auto_ref_key, auto_ref_bpm

st.session_state.ref_key = ref_key
st.session_state.ref_bpm = ref_bpm

st.write(f"• Dry → Key: **{dry_key}**, BPM: **{dry_bpm}**")
st.write(f"• Ref → Key: **{ref_key}**, BPM: **{ref_bpm}**")

# Analyze on button press
if dry_file and ref_file and st.button("Analyze Vocals"):
    t1 = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    t1.write(dry_file.read())
    t1.close()
    t2 = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    t2.write(ref_file.read())
    t2.close()
    dry_path, ref_path = t1.name, t2.name

    seg_start, seg_end = 0.33, 0.66
    def spectrum(path):
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(1)
        seg = data[int(len(data)*seg_start):int(len(data)*seg_end)]
        w = seg * windows.hann(len(seg))
        fft = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(seg), 1/sr)
        return freqs, np.abs(fft)

    freqs, mag_dry = spectrum(dry_path)
    _, mag_ref = spectrum(ref_path)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mag_dry, name="Dry"))
    fig.add_trace(go.Scatter(x=freqs, y=mag_ref, name="Ref"))
    fig.update_layout(title="Spectrum", xaxis_type='log', yaxis_title='Magnitude')
    st.plotly_chart(fig, use_container_width=True)

    bands = {"Low": (20, 600), "Mid": (600, 1200), "High": (1200, 12000)}
    peaks = []
    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        bm, bf = mag_dry[idx], freqs[idx]
        p, props = find_peaks(bm, height=bm.max() * 0.05)
        if not p.size:
            continue
        top = p[np.argsort(props["peak_heights"])[-3:]]
        for i in top:
            cf = bf[i]
            half = bm[i] / np.sqrt(2)
            l, r = i, i
            while l > 0 and bm[l] > half:
                l -= 1
            while r < len(bm) - 1 and bm[r] > half:
                r += 1
            bw = bf[r] - bf[l] if r < len(bf) and l >= 0 else cf / 10
            Q = cf / bw if bw > 0 else 10
            cut = np.clip(10 * np.log10(bm[i] / np.median(bm)), 1, 8)
            peaks.append({"Band": band, "Center": cf, "Q": Q, "Cut": cut})

    df_ed = pd.DataFrame(peaks)
    st.subheader("🔧 Surgical Notches")
    edits = []
    for i, row in df_ed.iterrows():
        st.markdown(f"**{row['Band']} @ {row['Center']:.1f} Hz**")
        c = st.slider("Cut dB", 1.0, 8.0, value=float(row["Cut"]), step=0.1, key=f"c{i}")
        q = st.slider("Q", 1.0, 10.0, value=float(row["Q"]), step=0.1, key=f"q{i}")
        edits.append({"Band": row["Band"], "Center": row["Center"], "Q": q, "Cut": c})
    df_ed = pd.DataFrame(edits)

    st.session_state.chain_df = df_ed.copy()
    st.session_state.chain_ready = True
    st.success("Analysis complete. Ready to generate mix chain.")

    os.unlink(dry_path)
    os.unlink(ref_path)

# Mix chain builder and output
if st.session_state.chain_ready:
    st.subheader("🔧 Iterative Mix Chain Builder")
    if st.button("Generate Engineer-Ready Chain"):
        s = st.session_state.chain_df.copy()
        for idx, pct in enumerate([0.10, 0.20, 0.10, 0.30, 0.20, 0.10], start=1):
            s[f"P{idx}"] = (s["Cut"] * pct).round(2)

        raw = {f"P{idx}": [] for idx in range(1, 7)}
        for _, r in s.iterrows():
            raw["P1"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q 1.5, Cut {r['P1']} dB")
            raw["P2"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q {3 if r['Band'] == 'Low' else 4}, Cut {r['P2']} dB")
        # Pass C: single shelf for all
        mid_shelf_avg = s['P3'].mean()
        raw["P3"].append(f"Mid-Shelf @ 1.2 kHz — Cut {mid_shelf_avg:.2f} dB")
        raw["P3"].append(f"High-Shelf @ 12 kHz — Cut {mid_shelf_avg:.2f} dB")
        for _, r in s.iterrows():
            raw["P4"].append(f"{r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB, Dynamic EQ")
            raw["P5"].append(f"{r['Band']} @ {r['Center']:.1f} Hz — Q 0.7, Cut {r['P5']} dB")

        templates = {
            "P1": ("Pass A: Broad Subtractive Cuts (10%)", "FabFilter Pro-Q3 High-Pass & broad notch"),
            "P2": ("Pass B: Narrow Notches (20%)", "Waves Q10 / FabFilter Pro-Q3"),
            "P3": ("Pass C: Dynamic Shelves (10%)", "FabFilter Pro-Q3 Dynamic Shelves"),
            "P4": ("Pass D: Dynamic Notches (30%)", "FabFilter Pro-Q3 Dynamic / Waves F6"),
            "P5": ("Pass E: Parallel Notches & Air (20%)", "Maag EQ4 Air + Tape Saturation"),
            "P6": ("Pass F: Compression & De-essing (10%)", "UAD 1176→SSL G-Bus→FabFilter Pro-DS")
        }

        for key in ["P1", "P2", "P3", "P4", "P5"]:
            header, plugin = templates[key]
            st.markdown(f"### {header}\n*Plugin:* {plugin}")
            for item in raw[key]:
                st.write(f"• {item}")

        # Pass F details
        st.markdown(f"### {templates['P6'][0]}\n*Plugins:* {templates['P6'][1]}")
        atk_sub, atk_ms = ms_to_subdivision(15, st.session_state.ref_bpm)
        rel_sub, rel_ms = ms_to_subdivision(250, st.session_state.ref_bpm)
        st.write(f"• Comp A 4:1 — Attack {atk_sub} ({atk_ms} ms), Release {rel_ms} ms")
        st.write(f"• Comp B 2.5:1 — Attack {atk_sub} ({atk_ms} ms), Release {rel_ms} ms")
        th_min, th_max = guidelines['deesser']['default']['threshold_db']
        r_min, r_max = guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser 5–8 kHz — Threshold {th_min} to {th_max} dB, Ratio {r_min}:{r_max}")
        st.write(f"• Final A/B check vs reference (Key: {st.session_state.ref_key}, BPM: {st.session_state.ref_bpm})")
