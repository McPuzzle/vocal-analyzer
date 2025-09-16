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
        "dynamic_eq": {"fabfilter_f6": {"attack_ms":[1,10],"release_ms":[50,250]}},
        "compressor": {"neve_33609": {"attack_ms":[10,30],"release_ms":[100,300]}},
        "deesser": {"default": {"threshold_db":[-20,-10],"ratio":[2,6]}}
    }
    try:
        with open('engineer_guidelines.json') as f:
            data = json.load(f)
            for k,v in default.items():
                if k not in data: data[k]=v
            return data
    except FileNotFoundError:
        return default

guidelines = load_guidelines()

def ms_to_subdivision(ms,bpm):
    beat_ms=60000/bpm
    subs={'1/4':beat_ms,'1/8':beat_ms/2,'1/16':beat_ms/4,'1/32':beat_ms/8,'1/64':beat_ms/16}
    name,val=min(subs.items(),key=lambda x:abs(x[1]-ms))
    return name,int(val)

st.title("🎤 Advanced Vocal EQ Analyzer")
st.write("Iterative subtractive-additive workflow with musically-aware, engineer-grade recommendations")

# Upload
dry_file=st.file_uploader("Upload Dry Vocal (WAV)",type='wav')
ref_file=st.file_uploader("Upload Reference Track (WAV)",type='wav')

# Session state
if 'chain_ready' not in st.session_state:
    st.session_state.chain_ready=False

# Key & BPM auto+override
st.subheader("🔑 Key & 🎵 Tempo Detection (Auto+Manual)")
def detect(path):
    try:
        y,sr=librosa.load(path,sr=None,mono=True,duration=30)
        chroma=librosa.feature.chroma_cens(y=y,sr=sr)
        key=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][chroma.mean(axis=1).argmax()]
        bpm,_=librosa.beat.beat_track(y=y,sr=sr)
        return key,int(bpm)
    except:
        return "Unknown",120

# detect on upload
if dry_file and 'auto_dry_key' not in st.session_state:
    tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    tmp.write(dry_file.read()); tmp.close()
    st.session_state.auto_dry_key,st.session_state.auto_dry_bpm=detect(tmp.name)
    os.unlink(tmp.name)
if ref_file and 'auto_ref_key' not in st.session_state:
    tmp=tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
    tmp.write(ref_file.read()); tmp.close()
    st.session_state.auto_ref_key,st.session_state.auto_ref_bpm=detect(tmp.name)
    os.unlink(tmp.name)

# defaults
auto_dry_key=st.session_state.get('auto_dry_key',"Unknown")
auto_dry_bpm=st.session_state.get('auto_dry_bpm',120)
auto_ref_key=st.session_state.get('auto_ref_key',"Unknown")
auto_ref_bpm=st.session_state.get('auto_ref_bpm',120)

# overrides
st.session_state.edit_dry=st.checkbox("Edit Dry Key/BPM?",value=st.session_state.get('edit_dry',False),key="chk_dry")
if st.session_state.edit_dry:
    col1,col2=st.columns(2)
    with col1:
        st.session_state.manual_dry_key=st.text_input("Dry Key",value=st.session_state.get('manual_dry_key',auto_dry_key),key="ti_dry")
    with col2:
        st.session_state.manual_dry_bpm=st.number_input("Dry BPM",min_value=0,max_value=300,value=st.session_state.get('manual_dry_bpm',auto_dry_bpm),key="ni_dry")
    dry_key,dry_bpm=st.session_state.manual_dry_key,st.session_state.manual_dry_bpm
else:
    dry_key,dry_bpm=auto_dry_key,auto_dry_bpm

st.session_state.edit_ref=st.checkbox("Edit Ref Key/BPM?",value=st.session_state.get('edit_ref',False),key="chk_ref")
if st.session_state.edit_ref:
    col1,col2=st.columns(2)
    with col1:
        st.session_state.manual_ref_key=st.text_input("Ref Key",value=st.session_state.get('manual_ref_key',auto_ref_key),key="ti_ref")
    with col2:
        st.session_state.manual_ref_bpm=st.number_input("Ref BPM",min_value=0,max_value=300,value=st.session_state.get('manual_ref_bpm',auto_ref_bpm),key="ni_ref")
    ref_key,ref_bpm=st.session_state.manual_ref_key,st.session_state.manual_ref_bpm
else:
    ref_key,ref_bpm=auto_ref_key,auto_ref_bpm

st.write(f"• Dry → Key: **{dry_key}**, BPM: **{dry_bpm}**")
st.write(f"• Ref → Key: **{ref_key}**, BPM: **{ref_bpm}**")

# Analyze
if dry_file and ref_file and st.button("Analyze Vocals"):
    # save temp for analysis
    tmp1=tempfile.NamedTemporaryFile(suffix='.wav',delete=False); tmp1.write(dry_file.read()); tmp1.close()
    tmp2=tempfile.NamedTemporaryFile(suffix='.wav',delete=False); tmp2.write(ref_file.read()); tmp2.close()
    dry_path,ref_path=tmp1.name,tmp2.name

    # spectrum
    seg_start,seg_end=0.33,0.66
    def spectrum(path):
        sr,data=wavfile.read(path)
        if data.ndim>1: data=data.mean(1)
        seg=data[int(len(data)*seg_start):int(len(data)*seg_end)]
        w=seg*windows.hann(len(seg));fft=np.fft.rfft(w);freqs=np.fft.rfftfreq(len(seg),1/sr)
        return freqs,np.abs(fft)
    freqs,mag_dry=spectrum(dry_path);_,mag_ref=spectrum(ref_path)
    fig=go.Figure();fig.add_trace(go.Scatter(x=freqs,y=mag_dry,name="Dry"));fig.add_trace(go.Scatter(x=freqs,y=mag_ref,name="Ref"))
    fig.update_layout(title="Spectrum",xaxis_type='log',yaxis_title='Mag');st.plotly_chart(fig,use_container_width=True)

    # find peaks
    bands={"Low":(20,600),"Mid":(600,1200),"High":(1200,12000)}
    peaks=[]
    for band,(fmin,fmax) in bands.items():
        idx=(freqs>=fmin)&(freqs<=fmax);bm,cf=mag_dry[idx],freqs[idx]
        p,props=find_peaks(bm,height=bm.max()*0.05)
        if not p.size:continue
        top=p[np.argsort(props['peak_heights'])[-3:]]
        for i in top:
            center=cf[i];half=bm[i]/np.sqrt(2)
            l,r=i,i
            while l>0 and bm[l]>half:l-=1
            while r<len(bm)-1 and bm[r]>half:r+=1
            bw=cf[r]-cf[l] if r<len(cf) and l>=0 else center/10
            Q=center/bw if bw>0 else 10
            cut=np.clip(10*np.log10(bm[i]/np.median(bm)),1,8)
            peaks.append({'Band':band,'Center':center,'Q':Q,'Cut':cut})
    df_ed=pd.DataFrame(peaks);st.subheader("Surgical Notches");edits=[]
    for i,row in df_ed.iterrows():
        st.markdown(f"**{row['Band']} @ {row['Center']:.1f} Hz**")
        c=st.slider("Cut dB",1.0,8.0,value=float(row['Cut']),step=0.1,key=f"c{i}")
        q=st.slider("Q",1.0,10.0,value=float(row['Q']),step=0.1,key=f"q{i}")
        edits.append({'Band':row['Band'],'Center':row['Center'],'Q':q,'Cut':c})
    df_ed=pd.DataFrame(edits);st.dataframe(df_ed)

    # store for chain
    st.session_state.chain_df=df_ed.copy();st.session_state.chain_ready=True

    # cleanup temps
    os.unlink(dry_path);os.unlink(ref_path)

# Chain builder
if st.session_state.chain_ready:
    st.subheader("🔧 Iterative Mix Chain Builder")
    if st.button("Generate Mix Chain Instructions"):
        s=st.session_state.chain_df.copy()
        for idx,pct in enumerate([0.10,0.20,0.10,0.30,0.20,0.10],start=1):
            s[f"P{idx}"]=(s['Cut']*pct).round(2)
        # Passes
        st.markdown("**Pass A (5–10% broad cuts)**")
        for _,r in s.iterrows():
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q1.5, Cut {r['P1']} dB")
        st.markdown("**Pass B (10–20% narrow notches)**")
        for _,r in s.iterrows():
            qv=3.0 if r['Band']=="Low" else 4.0
            st.write(f"• {r['Band']} @ {r['Center']:.1f} Hz — Q{qv}, Cut {r['P2']} dB")
        st.markdown("**Pass C (10–20% dynamic EQ)**")
        for _,r in s.iterrows():
            atk_sub,atk_ms=ms_to_subdivision(5,ref_bpm)
            rel_sub,rel_ms=ms_to_subdivision(120,ref_bpm)
            st.write(f"• {r['Band']} ~{r['Center']:.1f} Hz — Max {r['P4']} dB, Ratio3:1")
            st.write(f"  Attack~{atk_sub}({atk_ms}ms),Release~{rel_sub}({rel_ms}ms)")
        st.markdown("**Pass D (Serial Compression)**")
        atk_sub,atk_ms=ms_to_subdivision(15,ref_bpm)
        rel_sub,rel_ms=ms_to_subdivision(250,ref_bpm)
        st.write(f"• Comp A4:1—Attack~{atk_sub}({atk_ms}ms),Release~{rel_sub}({rel_ms}ms)")
        st.write(f"• Comp B2.5:1—Attack~{atk_sub}({atk_ms}ms),Release~{rel_sub}({rel_ms}ms)")
        st.markdown("**Pass E (Shelving & Parallel Sat)**")
        for _,r in s.iterrows():
            st.write(f"• Parallel Notch @ {r['Center']:.1f} Hz — Cut {r['P5']} dB")
        st.write("• High-Shelf @10kHz +1.5dB")
        st.write("• Parallel Sat Bus @10%")
        st.markdown("**Pass F (Final Touches)**")
        th_min,th_max=guidelines['deesser']['default']['threshold_db']
        r_min,r_max=guidelines['deesser']['default']['ratio']
        st.write(f"• De-Esser5–8kHz—Thresh{th_min}–{th_max},Ratio{r_min}:{r_max}")
        st.write(f"• A/B vs ref (Key:{ref_key},BPM:{ref_bpm})")
