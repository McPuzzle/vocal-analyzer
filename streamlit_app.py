import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import librosa
import tempfile, os
import json

@st.cache_data
def load_guidelines():
    return {
        'deesser': {'threshold_db': [-20, -10], 'ratio': [2, 6]},
        'saturation_blend': 0.1,
        'highpass_cutoff': 100,
        'highpass_slope': 12,
        'broad_eq_q': 1.5,
        'narrow_eq_q_low': 3,
        'narrow_eq_q_mid_high': 4,
        'max_q_dynamic_eq': 15,
        'min_q_dynamic_eq': 5,
        'dynamic_eq_attack_ms': 15,
        'dynamic_eq_release_ms': 120,
        'compression_gr_db': 2,
        'compression_ratios': [(4, 'Comp A'), (2.5, 'Comp B')],
        'deesser_freq_range': (5000, 8000),
        'sibilance_threshold_rms': 0.02  # Adjust empirically
    }

def ms_to_subdivision(ms, bpm):
    beat_ms = 60000 / bpm
    subs = {'1/4': beat_ms, '1/8': beat_ms/2, '1/16': beat_ms/4, '1/32': beat_ms/8, '1/64': beat_ms/16}
    name, val = min(subs.items(), key=lambda x: abs(x[1] - ms))
    return name, int(val)

def clamp_q(q, min_q=5, max_q=15):
    return max(min_q, min(q, max_q))

def detect_resonances(freqs, mag_dry, threshold_db=-15, max_peaks=9):
    peak_indices, properties = find_peaks(mag_dry, height=np.max(mag_dry)*0.05)
    if len(peak_indices) == 0:
        return []
    # Sort by peak height descending
    sorted_indices = peak_indices[np.argsort(properties['peak_heights'])[::-1]]
    resonances = []
    for idx in sorted_indices[:max_peaks]:
        cf = freqs[idx]
        mag = mag_dry[idx]
        bw = estimate_bandwidth(freqs, mag_dry, idx)
        q_val = clamp_q(cf / bw if bw > 0 else 10)
        cut_db = 10 * np.log10(mag / np.median(mag_dry))
        if cut_db < -threshold_db:
            resonances.append({'freq': cf, 'q': q_val, 'gain': abs(cut_db)})
    return resonances

def estimate_bandwidth(freqs, mag, peak_idx):
    half_power = mag[peak_idx] / np.sqrt(2)
    left = peak_idx
    right = peak_idx
    while left > 0 and mag[left] > half_power:
        left -= 1
    while right < len(mag) - 1 and mag[right] > half_power:
        right += 1
    bw = freqs[right] - freqs[left] if right > left else freqs[peak_idx] / 10
    return bw

def measure_sibilance(y, sr):
    # Rough RMS in 5–8 kHz range as sibilance proxy
    s = librosa.effects.harmonic(y)
    s_band = librosa.effects.bandpass(s, 5000, 8000, sr=sr)
    rms = np.sqrt(np.mean(s_band**2))
    return rms

# Streamlit app logic continues here...

st.title("🎤 Musical Vocal Mix Chain Assistant")
dry_file = st.file_uploader("Upload Dry Vocal (WAV)", type="wav")
ref_file = st.file_uploader("Upload Reference WAV", type="wav")

if dry_file and ref_file and st.button("Analyze Mix Chain"):
    # Save files temporarly
    t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    t1.write(dry_file.read())
    t1.close()
    t2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    t2.write(ref_file.read())
    t2.close()
    dry_path, ref_path = t1.name, t2.name

    # Load audio for sibilance
    y_dry, sr_dry = librosa.load(dry_path, sr=None)
    sibilance_rms = measure_sibilance(y_dry, sr_dry)

    # Spectrum calculation
    def spectrum(path):
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        seg = data[int(len(data)*0.33):int(len(data)*0.66)]
        w = seg * windows.hann(len(seg))
        fft = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(seg), 1/sr)
        mags = np.abs(fft)
        return freqs, mags

    freqs, mag_dry = spectrum(dry_path)

    # Detect resonances
    resonances = detect_resonances(freqs, mag_dry)

    # Show mix chain suggestions
    st.markdown("## Suggested Mix Chain")

    st.markdown("### 1. Cleaning")
    st.write(f"• High-pass filter at 100 Hz, 12 dB/oct")

    if len(resonances) > 0:
        st.markdown("• Broad subtractive EQ cuts:")
        for r in resonances:
            st.write(f" – Center {r['freq']:.1f} Hz | Q {r['q']:.2f} | Cut –{r['gain']:.2f} dB")

    st.markdown("### 2. Tone shaping")
    st.write("• Gentle shelving EQ to shape mid and high frequencies dynamically.")

    if sibilance_rms > 0.02:  # threshold value to tune as needed
        st.markdown("### 3. De-Essing")
        st.write(f"• Apply De-Esser around 5–8 kHz with threshold -20 to -10 dB, ratio 2:1 to 6:1.")

    st.markdown("### 4. Compression & Glue")
    st.write(f"• Use two-stage compression (e.g. 1176 at 4:1 and SSL G-bus at 2.5:1) with BPM synced attack/release.")
    st.write(f"• Add ~10% parallel tape/tube saturation for warmth.")

    st.markdown("### 5. Final touches")
    st.write(f"• Manual gain riding and A/B with reference track.")

    os.unlink(dry_path)
    os.unlink(ref_path)
