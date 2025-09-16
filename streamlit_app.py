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

# Load professional engineer guidelines
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
    subdivisions = {'1/4': beat_ms, '1/8': beat_ms/2, '1/16': beat_ms/4, '1/32': beat_ms/8, '1/64': beat_ms/16}
    name, val = min(subdivisions.items(), key=lambda x: abs(x[1]-ms))
    return name, int(val)

def butter_bandpass(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfilt(sos, data)

def compute_spectrum_db(y, sr, seg_start=0.33, seg_end=0.66):
    a = int(len(y)*seg_start)
    b = int(len(y)*seg_end)
    seg = y[a:b] * windows.hann(b-a)
    fft = np.fft.rfft(seg)
    mag = np.abs(fft)
    mag_db = 20*np.log10(mag + 1e-10)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    return freqs, mag_db

def find_broad_tonal_differences(freqs, mag_dry, mag_ref):
    diff = mag_dry - mag_ref
    corrs = []
    for band,(fmin,fmax) in guidelines['freq_bands'].items():
        mask = (freqs>=fmin)&(freqs<=fmax)
        if not np.any(mask): continue
        avg = np.mean(diff[mask])
        if abs(avg)>=guidelines['broad_difference_threshold_db']:
            cf = np.sqrt(fmin*fmax)
            corrs.append({'band':band,'center_freq':cf,'gain_adjust':-avg,'q':guidelines['broad_eq_q']})
    return corrs

def detect_surgical_resonances(freqs, mag_dry, mag_ref):
    diff = mag_dry - mag_ref
    idxs,props = find_peaks(mag_dry, height=np.max(mag_dry)*0.05)
    res = []
    for i in idxs:
        if diff[i]>guidelines['resonance_threshold_db']:
            bw = estimate_bandwidth(freqs,mag_dry,i)
            q = np.clip(freqs[i]/bw,*guidelines['surgical_eq_q_range'])
            res.append({'freq':freqs[i],'q':q,'gain_cut':diff[i]})
    return res

def estimate_bandwidth(freqs, mag, i):
    hp=mag[i]-3
    l,r=i,i
    while l>0 and mag[l]>hp: l-=1
    while r<len(mag)-1 and mag[r]>hp: r+=1
    bw=freqs[r]-freqs[l] if r>l else freqs[i]/20
    return bw

def measure_sibilance_level(y, sr):
    bp=bandpass_filter(y,*guidelines['deesser_freq_range'],sr)
    return np.sqrt(np.mean(bp**2))

def detect_key_bpm(path):
    try:
        y, sr = librosa.load(path, sr=None, duration=30)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        key_names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key=key_names[chroma.mean(axis=1).argmax()]
        tempo,_=librosa.beat.beat_track(y=y,sr=sr)
        return key,int(tempo)
    except:
        return "Unknown",120

def generate_mix_chain(broad, res, sib_dry, sib_ref, bpm):
    chain=[]
    # Step 1
    chain.append({'step':'1. Cleaning','plugin':'Pro-Q3','settings':[
        f"High-Pass: {guidelines['highpass_cutoff']}Hz, {guidelines['highpass_slope']}dB/oct"]})
    # Step 2
    if broad:
        s=[f"{c['band']} Bell @ {c['center_freq']:.0f}Hz | Q {c['q']:.1f} | Gain {c['gain_adjust']:+.1f}dB" for c in broad]
        chain.append({'step':'2. Broad EQ','plugin':'Pro-Q3','settings':s})
    # Step 3
    if res:
        s=[f"Notch @ {r['freq']:.0f}Hz | Q {r['q']:.1f} | Cut -{r['gain_cut']:.1f}dB" for r in res]
        chain.append({'step':'3. Surgical Notches','plugin':'Pro-Q3','settings':s})
        atk_sub,atk_ms=ms_to_subdivision(guidelines['dynamic_eq_attack_ms'],bpm)
        rel_sub,rel_ms=ms_to_subdivision(guidelines['dynamic_eq_release_ms'],bpm)
        s2=[f"Dynamic Notch @ {r['freq']:.0f}Hz | Q {r['q']:.1f} | Thresh -{r['gain_cut']*0.75:.1f}dB | "+
            f"Atk {atk_sub}({atk_ms}ms) Rel {rel_sub}({rel_ms}ms)" for r in res]
        chain.append({'step':'4. Dynamic EQ','plugin':'Pro-Q3 Dynamic','settings':s2})
    # Step 5
    if sib_dry>sib_ref+guidelines['sibilance_threshold_rms']:
        excess=(sib_dry-sib_ref)/sib_ref*100
        th=guidelines['deesser_threshold_range'][0]+excess*0.1
        rat=guidelines['deesser_ratio_range'][0]+excess*0.02
        chain.append({'step':'5. De-Essing','plugin':'Pro-DS','settings':[
            f"{guidelines['deesser_freq_range'][0]}–{guidelines['deesser_freq_range'][1]}Hz",
            f"Thresh {th:.1f}dB | Ratio {rat:.1f}:1"]})
    # Step 6
    atk_sub,atk_ms=ms_to_subdivision(guidelines['dynamic_eq_attack_ms'],bpm)
    rel_sub,rel_ms=ms_to_subdivision(guidelines['dynamic_eq_release_ms']*2,bpm)
    s=[f"{p}: Ratio {r}:1 | Atk {atk_sub}({atk_ms}ms) Rel {rel_sub}({rel_ms}ms) | GR ~2dB"
       for r,p in [(c,p) for c,p in guidelines['compression_ratios']]]
    chain.append({'step':'6. Compression','plugin':'Serial','settings':s})
    # Step 7
    chain.append({'step':'7. Saturation','plugin':'Parallel Bus','settings':[
        f"Tape/Tube Sat Blend {int(guidelines['saturation_blend']*100)}%",
        "Optional: gentle high-shelf +0.5dB @10kHz"]})
    return chain

# Streamlit App

st.title("🎤 Vocal Reference Matching Assistant")

# Upload columns
c1,c2=st.columns(2)
with c1: dry=st.file_uploader("Dry Vocal WAV",type='wav')
with c2: ref=st.file_uploader("Reference WAV",type='wav')

# Key/BPM
if ref:
    tmp=tmp2=None
    tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    tmp.write(ref.read()); tmp.close()
    key,tempo=detect_key_bpm(tmp.name)
    os.unlink(tmp.name)
    st.write("Detected Reference Key:",key)
    st.write("Detected Reference BPM:",tempo)

if dry and ref and st.button("Analyze & Generate Mix Chain"):
    # Save
    dtmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    rtmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    dtmp.write(dry.read()); dtmp.close()
    rtmp.write(ref.read()); rtmp.close()

    # Load with SoundFile
    ydry, sr_dry = sf.read(dtmp.name)
    yref, sr_ref = sf.read(rtmp.name)

    # Spectra
    f_dry, m_dry = compute_spectrum_db(ydry, sr_dry)
    f_ref, m_ref = compute_spectrum_db(yref, sr_ref)
    m_ref_i = np.interp(f_dry, f_ref, m_ref)

    # Analysis
    broad = find_broad_tonal_differences(f_dry, m_dry, m_ref_i)
    res = detect_surgical_resonances(f_dry, m_dry, m_ref_i)
    sib_d = measure_sibilance_level(ydry, sr_dry)
    sib_r = measure_sibilance_level(yref, sr_ref)

    # Mix chain
    chain = generate_mix_chain(broad, res, sib_d, sib_r, tempo)

    # Plot
    st.subheader("Spectral Comparison")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=f_dry,y=m_dry,name="Dry",line=dict(color='red')))
    fig.add_trace(go.Scatter(x=f_dry,y=m_ref_i,name="Ref",line=dict(color='blue')))
    fig.update_layout(xaxis_type='log',yaxis_title='dB',height=350)
    st.plotly_chart(fig,use_container_width=True)

    # Display chain
    for step in chain:
        st.markdown(f"### {step['step']} — {step['plugin']}")
        for s in step['settings']:
            st.write("•",s)
        st.write("---")

    # Cleanup
    os.unlink(dtmp.name)
    os.unlink(rtmp.name)
