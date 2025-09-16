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
    # Save temp
    for key, u in {'dry': dry_file, 'ref': ref_file}.items():
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(u.read()); tmp.close()
        locals()[f"{key}_path"] = tmp.name

    # Settings
    seg_start, seg_end = 0.33, 0.66
    bands = {"Low":(20,600),"Mid":(600,1200),"High":(1200,12000)}
    sr_dry, data = wavfile.read(dry_path := dry_path)
    sr_ref, data_ref = wavfile.read(ref_path := ref_path)
    if data.ndim>1: data = data.mean(1)
    if data_ref.ndim>1: data_ref = data_ref.mean(1)
    def get_spec(wav, sr):
        seg = wav[int(len(wav)*seg_start):int(len(wav)*seg_end)]
        w = seg * windows.hann(len(seg))
        fft = np.fft.rfft(w); freqs = np.fft.rfftfreq(len(seg),1/sr)
        return freqs, np.abs(fft)
    freqs, mag_dry = get_spec(data, sr_dry)
    _, mag_ref     = get_spec(data_ref, sr_ref)

    # Plot spectrum
    fig = go.Figure()
    fig.add_trace(go.Line(x=freqs, y=mag_dry, name="Dry", line_color='royalblue'))
    fig.add_trace(go.Line(x=freqs, y=mag_ref, name="Ref", line_color='orange'))
    fig.update_layout(title="Frequency Spectrum", xaxis_type='log', yaxis_title='Magnitude')
    st.plotly_chart(fig, use_container_width=True)

    # Surgical peaks detection
    peaks_list=[]
    for band,(fmin,fmax) in bands.items():
        idx=(freqs>=fmin)&(freqs<=fmax)
        p,props=find_peaks(mag_dry[idx],height=mag_dry[idx].max()*0.05)
        heights=props['peak_heights']; top=p[np.argsort(heights)[-3:]]
        for i in top:
            bin_i = np.where(idx)[0][i]
            cf=freqs[bin_i]; half=mag_dry[bin_i]/np.sqrt(2)
            l=bin_i; 
            while l>0 and mag_dry[l]>half: l-=1
            r=bin_i
            while r<len(mag_dry)-1 and mag_dry[r]>half: r+=1
            bw=freqs[r]-freqs[l]; Q=cf/bw if bw else np.nan
            peak_db=10*np.log10(mag_dry[bin_i]/np.median(mag_dry[idx]))
            peaks_list.append({"Band":band,"Center":cf,"Q":Q,"Cut":min(max(peak_db,1),8)})

    df_peaks=pd.DataFrame(peaks_list)
    st.subheader("🔧 Surgical Notch Settings")
    edited=[]
    for i,row in df_peaks.iterrows():
        st.markdown(f"**{row['Band']} Peak @ {row['Center']:.1f} Hz**")
        c=st.slider(f"Cut dB ({row['Center']:.1f}Hz)",1.0,8.0,value=row['Cut'],step=0.1,key=f"cut{i}")
        q=st.slider(f"Q ({row['Center']:.1f}Hz)",1.0,10.0,value=row['Q'],step=0.1,key=f"q{i}")
        edited.append({"Band":row['Band'],"Center":row['Center'],"Q":q,"Cut":c})
    df_edited=pd.DataFrame(edited)
    st.dataframe(df_edited)

    # Recompute broad bands with surgical applied (approx)
    # (Subtractive EQ reduces magnitude in each band)
    new_mag = mag_dry.copy()
    for _,r in df_edited.iterrows():
        g = 10**(-r['Cut']/20)
        # apply Gaussian notch
        new_mag *= 1 - (1-g)*np.exp(-0.5*((freqs-r['Center'])/(r['Center']/r['Q']))**2)
    broad=[]
    for band,(fmin,fmax) in bands.items():
        idx=(freqs>=fmin)&(freqs<=fmax)
        d_avg=np.mean(new_mag[idx]); r_avg=np.mean(mag_ref[idx])
        db=20*np.log10((r_avg+1e-6)/(d_avg+1e-6))
        broad.append({"Band":band,"Delta_dB":round(db,1)})
    st.subheader("🔄 Updated Broad-Band Delta_dB After Surgical EQ")
    st.table(pd.DataFrame(broad))

    # Cleanup
    os.unlink(dry_path); os.unlink(ref_path)
