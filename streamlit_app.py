import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import windows, find_peaks, butter, sosfilt
import plotly.graph_objects as go
import tempfile, os
import json
import librosa

# Professional engineer guidelines with realistic parameters
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
    subdivisions = {
        '1/4': beat_ms,
        '1/8': beat_ms / 2,
        '1/16': beat_ms / 4,
        '1/32': beat_ms / 8,
        '1/64': beat_ms / 16
    }
    name, val = min(subdivisions.items(), key=lambda x: abs(x[1] - ms))
    return name, int(val)

def butter_bandpass(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def compute_spectrum_db(y, sr, seg_start=0.33, seg_end=0.66):
    """Compute frequency spectrum in dB for consistent segment"""
    start_idx = int(len(y) * seg_start)
    end_idx = int(len(y) * seg_end)
    segment = y[start_idx:end_idx]
    
    # Apply window and compute FFT
    windowed = segment * windows.hann(len(segment))
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)
    
    # Convert to dB with noise floor protection
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    freqs = np.fft.rfftfreq(len(segment), 1/sr)
    
    return freqs, magnitude_db

def find_broad_tonal_differences(freqs, mag_dry, mag_ref):
    """Find broad frequency bands requiring tonal correction"""
    differences = mag_dry - mag_ref
    corrections = []
    
    for band_name, (fmin, fmax) in guidelines['freq_bands'].items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(band_mask):
            continue
            
        band_diff = np.mean(differences[band_mask])
        
        if abs(band_diff) >= guidelines['broad_difference_threshold_db']:
            center_freq = np.sqrt(fmin * fmax)  # Geometric center
            corrections.append({
                'band': band_name,
                'center_freq': center_freq,
                'gain_adjustment': -band_diff,  # Negative to correct
                'q': guidelines['broad_eq_q'],
                'type': 'broad_correction'
            })
    
    return corrections

def detect_surgical_resonances(freqs, mag_dry, mag_ref):
    """Detect narrow resonances requiring surgical cuts"""
    difference = mag_dry - mag_ref
    
    # Find peaks in dry vocal that exceed reference
    peak_indices, properties = find_peaks(mag_dry, height=np.max(mag_dry) * 0.05)
    
    resonances = []
    for idx in peak_indices:
        # Check if this peak is significantly higher than reference
        if difference[idx] > guidelines['resonance_threshold_db']:
            freq = freqs[idx]
            bandwidth = estimate_bandwidth(freqs, mag_dry, idx)
            q_factor = np.clip(freq / bandwidth, 
                             guidelines['surgical_eq_q_range'][0], 
                             guidelines['surgical_eq_q_range'][1])
            
            resonances.append({
                'freq': freq,
                'q': q_factor,
                'gain_cut': difference[idx],
                'type': 'surgical_notch'
            })
    
    return resonances

def estimate_bandwidth(freqs, magnitude, peak_idx):
    """Estimate -3dB bandwidth around peak"""
    peak_mag = magnitude[peak_idx]
    half_power = peak_mag - 3  # -3dB point
    
    # Find left and right -3dB points
    left_idx = peak_idx
    while left_idx > 0 and magnitude[left_idx] > half_power:
        left_idx -= 1
        
    right_idx = peak_idx
    while right_idx < len(magnitude) - 1 and magnitude[right_idx] > half_power:
        right_idx += 1
    
    bandwidth = freqs[right_idx] - freqs[left_idx]
    return max(bandwidth, freqs[peak_idx] / 20)  # Minimum bandwidth

def measure_sibilance_level(y, sr):
    """Measure RMS energy in sibilant frequency range"""
    sibilant_band = bandpass_filter(y, 
                                   guidelines['deesser_freq_range'][0], 
                                   guidelines['deesser_freq_range'][1], 
                                   sr)
    return np.sqrt(np.mean(sibilant_band**2))

def detect_key_bpm(audio_path):
    """Auto-detect key and BPM from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=30)
        
        # Key detection using chroma
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_names[chroma.mean(axis=1).argmax()]
        
        # BPM detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return key, int(tempo)
    except:
        return "Unknown", 120

def generate_mix_chain(broad_corrections, resonances, sibilance_dry, sibilance_ref, ref_bpm):
    """Generate complete engineer-ready mix chain"""
    chain_steps = []
    
    # Step 1: Cleaning and High-Pass
    chain_steps.append({
        'step': 'Step 1: Cleaning',
        'plugin': 'FabFilter Pro-Q3',
        'settings': [
            f"High-Pass Filter: {guidelines['highpass_cutoff']} Hz, {guidelines['highpass_slope']} dB/octave"
        ]
    })
    
    # Step 2: Broad Tonal Corrections (if needed)
    if broad_corrections:
        settings = []
        for corr in broad_corrections:
            settings.append(f"Bell EQ: {corr['center_freq']:.0f} Hz | Q: {corr['q']:.1f} | Gain: {corr['gain_adjustment']:+.1f} dB")
        
        chain_steps.append({
            'step': 'Step 2: Broad Tonal Corrections',
            'plugin': 'FabFilter Pro-Q3 or Waves Renaissance EQ',
            'settings': settings
        })
    
    # Step 3: Surgical Resonance Control (if needed)
    if resonances:
        settings = []
        for res in resonances:
            settings.append(f"Surgical Notch: {res['freq']:.0f} Hz | Q: {res['q']:.1f} | Cut: -{res['gain_cut']:.1f} dB")
        
        chain_steps.append({
            'step': 'Step 3: Surgical Resonance Control',
            'plugin': 'FabFilter Pro-Q3 or Waves Q10',
            'settings': settings
        })
    
    # Step 4: Dynamic EQ for Variable Resonances
    if resonances:
        settings = []
        attack_sub, attack_ms = ms_to_subdivision(guidelines['dynamic_eq_attack_ms'], ref_bpm)
        release_sub, release_ms = ms_to_subdivision(guidelines['dynamic_eq_release_ms'], ref_bpm)
        
        for res in resonances:
            threshold = -res['gain_cut'] * 0.75  # Set threshold at 75% of peak excess
            settings.append(f"Dynamic Bell: {res['freq']:.0f} Hz | Q: {res['q']:.1f} | "
                          f"Threshold: {threshold:.1f} dB | Attack: {attack_sub} ({attack_ms}ms) | "
                          f"Release: {release_sub} ({release_ms}ms)")
        
        chain_steps.append({
            'step': 'Step 4: Dynamic Resonance Control',
            'plugin': 'FabFilter Pro-Q3 Dynamic or Waves F6',
            'settings': settings
        })
    
    # Step 5: De-essing (if needed)
    if sibilance_dry > sibilance_ref + guidelines['sibilance_threshold_rms']:
        sibilance_excess = (sibilance_dry - sibilance_ref) / sibilance_ref
        threshold = guidelines['deesser_threshold_range'][0] + sibilance_excess * 10
        ratio = guidelines['deesser_ratio_range'][0] + sibilance_excess * 2
        
        chain_steps.append({
            'step': 'Step 5: De-essing',
            'plugin': 'FabFilter Pro-DS or Waves De-Esser',
            'settings': [
                f"Frequency Range: {guidelines['deesser_freq_range'][0]}-{guidelines['deesser_freq_range'][1]} Hz",
                f"Threshold: {threshold:.1f} dB",
                f"Ratio: {ratio:.1f}:1"
            ]
        })
    
    # Step 6: Compression
    settings = []
    attack_sub, attack_ms = ms_to_subdivision(guidelines['dynamic_eq_attack_ms'], ref_bpm)
    release_sub, release_ms = ms_to_subdivision(guidelines['dynamic_eq_release_ms']*2, ref_bpm)
    
    for ratio, plugin in guidelines['compression_ratios']:
        settings.append(f"{plugin}: Ratio {ratio}:1 | Attack: {attack_sub} ({attack_ms}ms) | "
                       f"Release: {release_sub} ({release_ms}ms) | Target GR: ~2dB")
    
    chain_steps.append({
        'step': 'Step 6: Compression Chain',
        'plugin': 'Serial Compression',
        'settings': settings
    })
    
    # Step 7: Saturation and Glue
    chain_steps.append({
        'step': 'Step 7: Saturation and Glue',
        'plugin': 'Parallel Processing',
        'settings': [
            f"Tape/Tube Saturation: {int(guidelines['saturation_blend']*100)}% parallel blend",
            "Optional bus EQ: gentle high-shelf +0.5dB @ 10kHz"
        ]
    })
    
    return chain_steps

# Streamlit UI
st.title("🎤 Professional Vocal Reference Matching Assistant")
st.write("Upload your dry vocal and reference track for detailed, engineer-ready mix chain generation")

# File uploads
col1, col2 = st.columns(2)
with col1:
    dry_file = st.file_uploader("Dry Vocal (WAV)", type=['wav'])
with col2:
    ref_file = st.file_uploader("Reference Vocal (WAV)", type=['wav'])

# Key/BPM detection and manual override
if dry_file or ref_file:
    st.subheader("🔑 Key & BPM Detection")
    
    # Auto-detect on file upload
    detected_key, detected_bpm = "Unknown", 120
    if ref_file:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(ref_file.read())
            tmp.flush()
            detected_key, detected_bpm = detect_key_bpm(tmp.name)
            os.unlink(tmp.name)
    
    # Manual override options
    col1, col2 = st.columns(2)
    with col1:
        key = st.text_input("Reference Key", value=detected_key)
    with col2:
        bpm = st.number_input("Reference BPM", min_value=60, max_value=200, value=detected_bpm)

# Main analysis
if dry_file and ref_file and st.button("🔬 Analyze & Generate Mix Chain", type="primary"):
    with st.spinner("Analyzing vocals and generating professional mix chain..."):
        # Save temporary files
        dry_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        ref_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        dry_temp.write(dry_file.read())
        dry_temp.close()
        ref_temp.write(ref_file.read())
        ref_temp.close()
        
        try:
            # Load and analyze audio
            y_dry, sr_dry = librosa.load(dry_temp.name, sr=None)
            y_ref, sr_ref = librosa.load(ref_temp.name, sr=None)
            
            # Compute spectra
            freqs_dry, mag_dry = compute_spectrum_db(y_dry, sr_dry)
            freqs_ref, mag_ref = compute_spectrum_db(y_ref, sr_ref)
            
            # Interpolate reference to match dry vocal frequency resolution
            mag_ref_interp = np.interp(freqs_dry, freqs_ref, mag_ref)
            
            # Spectral analysis
            broad_corrections = find_broad_tonal_differences(freqs_dry, mag_dry, mag_ref_interp)
            resonances = detect_surgical_resonances(freqs_dry, mag_dry, mag_ref_interp)
            
            # Sibilance analysis
            sibilance_dry = measure_sibilance_level(y_dry, sr_dry)
            sibilance_ref = measure_sibilance_level(y_ref, sr_ref)
            
            # Generate mix chain
            mix_chain = generate_mix_chain(broad_corrections, resonances, 
                                         sibilance_dry, sibilance_ref, bpm)
            
            # Display spectral comparison
            st.subheader("📊 Spectral Analysis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs_dry, y=mag_dry, name="Dry Vocal", 
                                   line=dict(color='red', width=1)))
            fig.add_trace(go.Scatter(x=freqs_dry, y=mag_ref_interp, name="Reference", 
                                   line=dict(color='blue', width=1)))
            fig.update_layout(
                title="Spectral Comparison",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                xaxis_type="log",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display mix chain
            st.subheader("🎛️ Professional Mix Chain")
            st.info(f"**Reference Track Info:** Key: {key}, BPM: {bpm}")
            
            for step_info in mix_chain:
                st.markdown(f"### {step_info['step']}")
                st.markdown(f"*Plugin: {step_info['plugin']}*")
                for setting in step_info['settings']:
                    st.write(f"• {setting}")
                st.write("---")
            
            # Summary and recommendations
            st.subheader("📝 Engineer Notes")
            notes = []
            
            if broad_corrections:
                notes.append(f"**Tonal Balance:** {len(broad_corrections)} frequency bands need broad correction")
            
            if resonances:
                notes.append(f"**Resonances:** {len(resonances)} problematic resonances detected requiring surgical cuts")
            
            if sibilance_dry > sibilance_ref + guidelines['sibilance_threshold_rms']:
                excess = ((sibilance_dry - sibilance_ref) / sibilance_ref) * 100
                notes.append(f"**Sibilance:** {excess:.0f}% excess sibilance detected - de-essing recommended")
            else:
                notes.append("**Sibilance:** Levels are appropriate - no de-essing needed")
            
            notes.append("**Workflow:** Apply each step sequentially, A/B testing against reference after each stage")
            notes.append("**Final Step:** Manual level riding and automation for consistency")
            
            for note in notes:
                st.write(note)
                
        finally:
            # Cleanup
            os.unlink(dry_temp.name)
            os.unlink(ref_temp.name)

st.markdown("---")
st.caption("Professional vocal mixing assistant - generates engineer-ready mix chains based on spectral reference matching")
