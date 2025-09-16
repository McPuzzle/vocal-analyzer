import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

# Load engineering guidelines
@st.cache_data
def load_guidelines():
    try:
        with open('engineer_guidelines.json') as f:
            return json.load(f)
    except:
        return {"dynamic_eq": {"default": {"attack_ms": [1, 10], "release_ms": [50, 250]}}, 
                "compressor": {"default": {"attack_ms": [10, 30], "release_ms": [100, 300]}}}

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
    name, val = min(subdivisions.items(), key=lambda x: abs(x[1]-ms))
    return name, int(val)

st.title("🎤 Advanced Vocal EQ Analyzer")
st.write("Upload your dry vocal and reference track, then get musically-aware EQ recommendations")

# Upload
dry_file = st.file_uploader("Dry Vocal (WAV)", type='wav')
ref_file = st.file_uploader("Reference Vocal (WAV)", type='wav')

if dry_file and ref_file and st.button("Start Analysis"):
    # Save temp files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp1:
        tmp1.write(dry_file.read())
        dry_path = tmp1.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp2:
        tmp2.write(ref_file.read())
        ref_path = tmp2.name

    # Load audio
    sr_dry, data = wavfile.read(dry_path)
    sr_ref, data_ref = wavfile.read(ref_path)
    if data.ndim > 1: data = data.mean(1)
    if data_ref.ndim > 1: data_ref = data_ref.mean(1)
    
    # Key & BPM Detection
    try:
        y_ref, sr = librosa.load(ref_path, sr=None, mono=True, duration=30)
        chromagram = librosa.feature.chroma_cens(y=y_ref, sr=sr)
        key_index = chromagram.mean(axis=1).argmax()
        key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        detected_key = key_names[key_index]
        tempo, _ = librosa.beat.beat_track(y=y_ref, sr=sr)
        st.info(f"🔑 Detected Key: {detected_key} | 🎵 Tempo: {int(tempo)} BPM")
    except:
        detected_key, tempo = "Unknown", 120
        st.warning("Could not detect key/tempo, using defaults")
    
    # Settings
    seg_start, seg_end = 0.33, 0.66
    bands = {"Low":(20,600), "Mid":(600,1200), "High":(1200,12000)}
    
    def get_spec(wav, sr):
        seg = wav[int(len(wav)*seg_start):int(len(wav)*seg_end)]
        w = seg * windows.hann(len(seg))
        fft = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(seg), 1/sr)
        return freqs, np.abs(fft)
    
    freqs, mag_dry = get_spec(data, sr_dry)
    _, mag_ref = get_spec(data_ref, sr_ref)

    # Plot spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mag_dry, name="Dry", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=freqs, y=mag_ref, name="Ref", line=dict(color='orange')))
    fig.update_layout(title="Frequency Spectrum", xaxis_type='log', yaxis_title='Magnitude')
    st.plotly_chart(fig, use_container_width=True)

    # Surgical peaks detection
    peaks_list = []
    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx):
            continue
        band_mag = mag_dry[idx]
        band_freqs = freqs[idx]
        peaks, props = find_peaks(band_mag, height=band_mag.max()*0.05)
        if len(peaks) == 0:
            continue
        heights = props['peak_heights']
        top = peaks[np.argsort(heights)[-3:]]
        
        for p in top:
            cf = band_freqs[p]
            half = band_mag[p] / np.sqrt(2)
            left = p
            while left > 0 and band_mag[left] > half:
                left -= 1
            right = p
            while right < len(band_mag)-1 and band_mag[right] > half:
                right += 1
            bw = band_freqs[right] - band_freqs[left] if right < len(band_freqs) and left >= 0 else cf/10
            Q = cf/bw if bw > 0 else 10
            peak_db = min(max(10*np.log10(band_mag[p]/np.median(band_mag)), 1), 8)
            peaks_list.append({"Band":band, "Center":cf, "Q":Q, "Cut":peak_db})

    df_peaks = pd.DataFrame(peaks_list)
    st.subheader("🔧 Surgical Notch Settings")
    
    edited = []
    for i, row in df_peaks.iterrows():
        st.markdown(f"**{row['Band']} Peak @ {row['Center']:.1f} Hz**")
        c = st.slider(f"Cut dB ({row['Center']:.1f}Hz)", 1.0, 8.0, value=float(row['Cut']), step=0.1, key=f"cut{i}")
        q = st.slider(f"Q ({row['Center']:.1f}Hz)", 1.0, 10.0, value=float(row['Q']), step=0.1, key=f"q{i}")
        edited.append({"Band":row['Band'], "Center":row['Center'], "Q":q, "Cut":c})
    
    df_edited = pd.DataFrame(edited)
    st.dataframe(df_edited)

    # Ultra-Granular Mix Chain Builder
    if st.button("🔧 Build Ultra-Granular Mix Chain"):
        surgical = pd.DataFrame(edited)

        # Calculate splits: 10%, 20%, 30%, 20%, 10%, 10% of Cut
        surgical['Stage1_Cut'] = (surgical['Cut'] * 0.10).round(2)
        surgical['Stage2_Cut'] = (surgical['Cut'] * 0.20).round(2)
        surgical['Stage3_Cut'] = (surgical['Cut'] * 0.10).round(2)
        surgical['Stage4_Cut'] = (surgical['Cut'] * 0.30).round(2)
        surgical['Stage5_Cut'] = (surgical['Cut'] * 0.20).round(2)
        surgical['Stage6_Cut'] = (surgical['Cut'] * 0.10).round(2)

        # 1. Static Broad-Q & Low-Cut (10%)
        st.subheader("Stage 1: Static Broad-Q & Low-Cut (10%)")
        st.write("High-Pass Filter at 100 Hz, 12 dB/oct")
        for _,r in surgical.iterrows():
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q 1.5, Cut {r['Stage1_Cut']} dB")

        # 2. Narrow Static Notches (20%)
        st.subheader("Stage 2: Narrow Static Notches (20%)")
        for _,r in surgical.iterrows():
            q = 3.0 if r['Band']=='Low' else 4.0 if r['Band']=='Mid' else 4.0
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q {q}, Cut {r['Stage2_Cut']} dB")

        # 3. Shelf EQ (10%)
        st.subheader("Stage 3: Shelf EQ (10%)")
        st.write(f"• Mid-Shelf @ 1.2 kHz, Q 1.0, Cut {surgical['Stage3_Cut'].mean():.2f} dB")
        st.write(f"• High-Pass Shelf @ 12 kHz, Q 0.7, Cut {surgical['Stage3_Cut'].mean():.2f} dB")

        # 4. Dynamic Notches (30%) - with musical timing
        st.subheader("Stage 4: Dynamic Notches (30%) - Musically Timed")
        for _,r in surgical.iterrows():
            attack_subdiv, attack_ms = ms_to_subdivision(5, tempo)
            release_subdiv, release_ms = ms_to_subdivision(120, tempo)
            min_rel, max_rel = guidelines['dynamic_eq']['fabfilter_f6']['release_ms']
            st.write(f"• {r['Band']} ~{r['Center']:.1f} Hz — Max {r['Stage4_Cut']} dB reduction, Ratio 3:1")
            st.write(f"  Attack ~{attack_subdiv} ({attack_ms} ms), Release ~{release_subdiv} ({release_ms} ms)")
            st.write(f"  *F6 Recommended: {min_rel}–{max_rel} ms release*")
        
        # 5. Parallel Notches & Air (20%)
        st.subheader("Stage 5: Parallel Notches & Air (20%)")
        st.write("Send 10% to parallel saturation bus (HPF 800 Hz, LPF 12 kHz)")
        for _,r in surgical.iterrows():
            st.write(f"• Parallel Notch @ {r['Center']:.1f} Hz, Q 0.7, Cut {r['Stage5_Cut']} dB")
        st.write("• High-Shelf @ 10 kHz, Q 0.7, Boost +1.5 dB")

        # 6. Limiting & Compression (10%) - with musical timing
        st.subheader("Stage 6: Limiting & Compression - Musically Timed")
        limit_attack_subdiv, limit_attack_ms = ms_to_subdivision(3, tempo)
        comp_attack_subdiv, comp_attack_ms = ms_to_subdivision(15, tempo)
        comp_release_subdiv, comp_release_ms = ms_to_subdivision(250, tempo)
        
        min_comp_atk, max_comp_atk = guidelines['compressor']['neve_33609']['attack_ms']
        min_comp_rel, max_comp_rel = guidelines['compressor']['neve_33609']['release_ms']
        
        st.write(f"• Limiter: Attack ~{limit_attack_subdiv} ({limit_attack_ms} ms), Release 50 ms, GR ~2 dB")
        st.write(f"• Comp A (Neve 33609 style): 4:1 ratio")
        st.write(f"  Attack ~{comp_attack_subdiv} ({comp_attack_ms} ms; recommended {min_comp_atk}–{max_comp_atk} ms)")
        st.write(f"  Release ~{comp_release_subdiv} ({comp_release_ms} ms; recommended {min_comp_rel}–{max_comp_rel} ms)")
        st.write("• Comp B (SSL Bus style): 2.5:1, Attack 15 ms, Release 250 ms, GR ~2 dB")

        # Final Touches
        st.subheader("Final Touches")
        min_de_thresh, max_de_thresh = guidelines['deesser']['default']['threshold_db']
        min_de_ratio, max_de_ratio = guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser 5–8 kHz, Threshold {min_de_thresh}–{max_de_thresh} dB, Ratio {min_de_ratio}–{max_de_ratio}:1")
        st.write("• Manual clip-gain rides as needed")
        st.write(f"• A/B Check against reference in {detected_key} at {int(tempo)} BPM")

    # Download CSV
    csv_data = df_edited.to_csv(index=False)
    st.download_button("📥 Download Surgical EQ Settings", csv_data, "surgical_eq_settings.csv")

    # Cleanup
    os.unlink(dry_path)
    os.unlink(ref_path)
