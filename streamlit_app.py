import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import windows, find_peaks, butter, sosfilt
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

@st.cache_data
def load_guidelines():
    return {
        'broad_difference_threshold_db': 0.5,
        'resonance_threshold_db': 3.0,
        'sibilance_threshold_rms': 0.02,
        'highpass_cutoff': 100,
        'highpass_slope': 12,
        'broad_eq_q': 1.5,
        'surgical_eq_q_range': [5, 15],
        'dynamic_eq_attack_ms': 15,
        'dynamic_eq_release_ms': 120,
        'compression_ratios': [(4, 'UAD 1176'), (2.5, 'SSL G-Bus')],
        'deesser_freq_range': (5000, 8000),
        'deesser_threshold_range': [-20, -10],
        'deesser_ratio_range': [2, 6],
        'saturation_blend': 0.1,
        'freq_bands': {
            'Low': (20, 600),
            'Low-Mid': (600, 1200),
            'Mid': (1200, 3000),
            'High-Mid': (3000, 8000),
            'High': (8000, 16000)
        }
    }

guidelines = load_guidelines()

def ms_to_subdivision(ms, bpm):
    beat_ms = 60000 / bpm
    subs = {'1/4': beat_ms, '1/8': beat_ms/2, '1/16': beat_ms/4, '1/32': beat_ms/8, '1/64': beat_ms/16}
    name, val = min(subs.items(), key=lambda x: abs(x[1] - ms))
    return name, int(val)

def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfilt(sos, data)

def compute_spectrum_db(y, sr, seg_start=0.33, seg_end=0.66):
    a = int(len(y) * seg_start)
    b = int(len(y) * seg_end)
    seg = y[a:b] * windows.hann(b - a)
    fft = np.fft.rfft(seg)
    mag = np.abs(fft)
    mag_db = 20 * np.log10(mag + 1e-10)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    return freqs, mag_db

def find_broad_differences(freqs, dry_db, ref_db):
    diff = dry_db - ref_db
    corrections = []
    for band, (fmin, fmax) in guidelines['freq_bands'].items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            continue
        avg = np.mean(diff[mask])
        if abs(avg) >= guidelines['broad_difference_threshold_db']:
            cf = np.sqrt(fmin * fmax)
            corrections.append({'band': band, 'center': cf, 'gain': -avg, 'q': guidelines['broad_eq_q']})
    return corrections

def estimate_bandwidth(freqs, mag, idx):
    hp = mag[idx] - 3
    l, r = idx, idx
    while l > 0 and mag[l] > hp:
        l -= 1
    while r < len(mag) - 1 and mag[r] > hp:
        r += 1
    return max(freqs[r] - freqs[l], freqs[idx] / 20)

def detect_surgical_res(freqs, dry_db, ref_db):
    diff = dry_db - ref_db
    idxs, _ = find_peaks(dry_db, height=np.max(dry_db) * 0.05)
    res = []
    for i in idxs:
        if diff[i] > guidelines['resonance_threshold_db']:
            bw = estimate_bandwidth(freqs, dry_db, i)
            q = np.clip(freqs[i] / bw, *guidelines['surgical_eq_q_range'])
            res.append({'freq': freqs[i], 'q': q, 'cut': diff[i]})
    return res

def measure_sibilance(y, sr):
    bp = bandpass_filter(y, *guidelines['deesser_freq_range'], sr)
    return np.sqrt(np.mean(bp ** 2))

def detect_key_bpm_from_bytes(wav_bytes):
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(wav_bytes)
    tmp.close()
    try:
        y, sr = librosa.load(tmp.name, sr=None, duration=30)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key = keys[chroma.mean(axis=1).argmax()]
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
    finally:
        os.unlink(tmp.name)
    return key, bpm

def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        y, sr = librosa.load(path, sr=None)
    else:
        y, sr = sf.read(path)
    return y, sr

def generate_chain(broad, res, s_dry, s_ref, bpm):
    chain = []
    chain.append({
        'step': 'Cleaning',
        'plugin': 'Pro-Q3',
        'settings': [f"HPF {guidelines['highpass_cutoff']}Hz {guidelines['highpass_slope']}dB/oct"]
    })
    if broad:
        s = [f"{c['band']} EQ @ {c['center']:.0f}Hz Q{c['q']} Gain{c['gain']:+.1f}dB" for c in broad]
        chain.append({'step': 'Broad EQ', 'plugin': 'Pro-Q3', 'settings': s})
    if res:
        s_notch = [f"Notch @ {r['freq']:.0f}Hz Q{r['q']:.1f} Cut-{r['cut']:.1f}dB" for r in res]
        chain.append({'step': 'Surgical Notches', 'plugin': 'Pro-Q3', 'settings': s_notch})
        atk_nm, atk_ms = ms_to_subdivision(guidelines['dynamic_eq_attack_ms'], bpm)
        rel_nm, rel_ms = ms_to_subdivision(guidelines['dynamic_eq_release_ms'], bpm)
        s_dyn = [
            f"Dynamic @ {r['freq']:.0f}Hz Q{r['q']:.1f} Th-{r['cut']*0.75:.1f}dB "
            f"Atk{atk_nm}({atk_ms}ms) Rel{rel_nm}({rel_ms}ms)"
            for r in res
        ]
        chain.append({'step': 'Dynamic EQ', 'plugin': 'Pro-Q3 Dyn', 'settings': s_dyn})
    if s_dry > s_ref + guidelines['sibilance_threshold_rms']:
        excess = ((s_dry - s_ref) / s_ref) * 100
        th = guidelines['deesser_threshold_range'][0] + excess * 0.1
        ra = guidelines['deesser_ratio_range'][0] + excess * 0.02
        chain.append({
            'step': 'De-essing',
            'plugin': 'Pro-DS',
            'settings': [
                f"{guidelines['deesser_freq_range'][0]}–{guidelines['deesser_freq_range'][1]}Hz",
                f"Thr{th:.1f}dB Ratio{ra:.1f}:1"
            ]
        })
    atk_nm, atk_ms = ms_to_subdivision(guidelines['dynamic_eq_attack_ms'], bpm)
    rel_nm, rel_ms = ms_to_subdivision(guidelines['dynamic_eq_release_ms'] * 2, bpm)
    s_comp = [f"{p}:R{r}:1 Atk{atk_nm} Rel{rel_nm}"
              for r, p in guidelines['compression_ratios']]
    chain.append({'step': 'Compression', 'plugin': 'Serial', 'settings': s_comp})
    chain.append({
        'step': 'Saturation',
        'plugin': 'Parallel',
        'settings': [f"SatBlend{int(guidelines['saturation_blend']*100)}%", "+0.5dB HS@10kHz"]
    })
    return chain

st.title("Professional Vocal Reference Mix Assistant")

col1, col2 = st.columns(2)
with col1:
    dry = st.file_uploader("Dry WAV", type='wav')
with col2:
    ref = st.file_uploader("Ref WAV", type='wav')

if dry and ref:
    # Read uploads once into memory
    dry_bytes = dry.read()
    ref_bytes = ref.read()

    # Detect key/BPM on reference bytes
    key, bpm = detect_key_bpm_from_bytes(ref_bytes)
    st.write("Key:", key, "BPM:", bpm)

    if st.button("Analyze"):
        # Write both temp files from memory bytes
        dt = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dt.write(dry_bytes)
        dt.close()

        rt = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        rt.write(ref_bytes)
        rt.close()

        # Load, analyze, and build chain
        y_d, sr_d = load_audio(dt.name)
        y_r, sr_r = load_audio(rt.name)
        f_d, m_d = compute_spectrum_db(y_d, sr_d)
        f_r, m_r = compute_spectrum_db(y_r, sr_r)
        m_ri = np.interp(f_d, f_r, m_r)
        broad = find_broad_differences(f_d, m_d, m_ri)
        res = detect_surgical_res(f_d, m_d, m_ri)
        s_d = measure_sibilance(y_d, sr_d)
        s_r = measure_sibilance(y_r, sr_r)
        chain = generate_chain(broad, res, s_d, s_r, bpm)

        # Display results
        st.subheader("Spectral Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f_d, y=m_d, name="Dry"))
        fig.add_trace(go.Scatter(x=f_d, y=m_ri, name="Ref"))
        fig.update_layout(xaxis_type='log', yaxis_title='dB', height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mix Chain")
        for step in chain:
            st.markdown(f"**{step['step']}** — {step['plugin']}")
            for s in step['settings']:
                st.write("•", s)
            st.write("---")

        # Cleanup
        os.unlink(dt.name)
        os.unlink(rt.name)
