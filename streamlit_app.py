import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows
import tempfile
import os

st.title("🎤 Professional Vocal EQ Analyzer")
st.write("Upload your dry vocal and reference track to get precise EQ recommendations")

# File uploaders
dry_vocal = st.file_uploader("Upload Dry Vocal (WAV)", type=['wav'])
reference = st.file_uploader("Upload Reference Vocal (WAV)", type=['wav'])

# Run analysis when both files are provided
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
        "Low": (20, 600),
        "Mid": (600, 1200),
        "High": (1200, 12000),
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
    _, mag_ref = spectrum(seg_ref, sr_dry)

    # Compare bands and build results
    results = []
    for name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        avg_dry = np.mean(mag_dry[idx])
        avg_ref = np.mean(mag_ref[idx])
        diff_db = 20 * np.log10((avg_ref+1e-6)/(avg_dry+1e-6))
        results.append({
            "Band": name,
            "Dry_Avg_Mag": round(avg_dry, 1),
            "Ref_Avg_Mag": round(avg_ref, 1),
            "Delta_dB": round(diff_db, 1),
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    st.subheader("🎯 Frequency Band Comparison")
    st.dataframe(results_df)

    # Show suggested EQ moves
    st.subheader("🎚 Suggested PAZ EQ Adjustments")
    for r in results:
        action = f"BOOST {abs(r['Delta_dB'])} dB" if r['Delta_dB'] > 0 else f"CUT {abs(r['Delta_dB'])} dB"
        st.write(f"{r['Band']}: {action} in {r['Band']} band")

    # Download CSV
    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download EQ Settings", csv, "eq_recommendations.csv")

    # Cleanup temp files
    os.unlink(dry_path)
    os.unlink(ref_path)
