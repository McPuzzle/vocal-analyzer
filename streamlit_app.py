import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import plotly.graph_objects as go
import tempfile, os

st.title("🎤 Advanced Vocal EQ Analyzer")
st.write("Upload your dry vocal and reference track, then refine surgical EQ moves interactively.")

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
            # Find bandwidth
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

    # Download CSV
    csv_data = df_edited.to_csv(index=False)
    st.download_button("📥 Download Surgical EQ Settings", csv_data, "surgical_eq_settings.csv")

    # Cleanup
    os.unlink(dry_path)
    os.unlink(ref_path)
