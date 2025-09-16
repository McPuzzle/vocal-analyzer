import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks
import tempfile
import os

st.title("🎤 Professional Vocal EQ Analyzer")
st.write("Upload your dry vocal and reference track to get precise EQ recommendations")

# File uploaders
dry_vocal = st.file_uploader("Upload Dry Vocal (WAV)", type=['wav'])
reference = st.file_uploader("Upload Reference Vocal (WAV)", type=['wav'])

if dry_vocal and reference and st.button("Analyze Vocals"):
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp1:
        tmp1.write(dry_vocal.read())
        dry_path = tmp1.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp2:
        tmp2.write(reference.read())
        ref_path = tmp2.name

    # Configuration
    seg_start = 0.33
    seg_end = 0.66
    bands = {
        "Low":    (20,   600),
        "Mid":    (600,  1200),
        "High":   (1200, 12000),
    }

    def load_segment(path):
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        n = len(data)
        return sr, data[int(n*seg_start):int(n*seg_end)]

    def spectrum(segment, sr):
        w = segment * windows.hann(len(segment))
        fft_vals = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(segment), 1.0/sr)
        mag = np.abs(fft_vals)
        return freqs, mag

    # Load and analyze both vocals
    sr_dry, seg_dry = load_segment(dry_path)
    sr_ref, seg_ref = load_segment(ref_path)
    assert sr_dry == sr_ref, "Sample rates must match"

    freqs, mag_dry = spectrum(seg_dry, sr_dry)
    _,     mag_ref = spectrum(seg_ref, sr_dry)

    # Broad-band comparison
    broad_results = []
    for name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        avg_dry    = np.mean(mag_dry[idx])
        avg_ref    = np.mean(mag_ref[idx])
        diff_db    = 20 * np.log10((avg_ref+1e-6)/(avg_dry+1e-6))
        broad_results.append({
            "Band":        name,
            "Dry_Avg_Mag": round(avg_dry, 1),
            "Ref_Avg_Mag": round(avg_ref, 1),
            "Delta_dB":    round(diff_db, 1),
        })

    broad_df = pd.DataFrame(broad_results)
    st.subheader("🎯 Frequency Band Comparison")
    st.dataframe(broad_df)

    st.subheader("🎚 Suggested PAZ EQ Adjustments (Broad)")
    for r in broad_results:
        action = f"BOOST {abs(r['Delta_dB'])} dB" if r['Delta_dB'] > 0 else f"CUT {abs(r['Delta_dB'])} dB"
        st.write(f"{r['Band']}: {action} in {r['Band']} band")

    # Surgical peak detection
    peak_results = []
    threshold_fraction = 0.05

    for name, (fmin, fmax) in bands.items():
        idx_band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        # Find peaks in dry magnitude within band
        band_mag = mag_dry[idx_band]
        peaks, props = find_peaks(band_mag, height=np.max(band_mag)*threshold_fraction)
        peak_heights = props["peak_heights"]
        if len(peaks) == 0:
            continue
        # Sort by height descending, take top 3
        top_idxs = peaks[np.argsort(peak_heights)[-3:]]
        for p in top_idxs:
            bin_idx = idx_band[p]
            cf = freqs[bin_idx]
            half_power = mag_dry[bin_idx] / np.sqrt(2)
            # Find –3dB bandwidth edges
            left = bin_idx
            while left > 0 and mag_dry[left] > half_power:
                left -= 1
            right = bin_idx
            while right < len(mag_dry)-1 and mag_dry[right] > half_power:
                right += 1
            bw = freqs[right] - freqs[left]
            q = cf / bw if bw > 0 else np.nan
            # Suggest cut depth
            median_mag = np.median(band_mag)
            rel = mag_dry[bin_idx] / median_mag
            cut_db = np.clip(10 * np.log10(rel), 1, 8)
            peak_results.append({
                "Band":      name,
                "Center_Hz": round(cf, 1),
                "Q":         round(q, 2),
                "Cut_dB":    round(cut_db, 1),
            })

    if peak_results:
        peaks_df = pd.DataFrame(peak_results)
        st.subheader("🔍 Surgical EQ Suggestions")
        st.dataframe(peaks_df)
        csv2 = peaks_df.to_csv(index=False)
        st.download_button("📥 Download Surgical EQ List", csv2, "surgical_eq.csv")

    # Cleanup
    os.unlink(dry_path)
    os.unlink(ref_path)
