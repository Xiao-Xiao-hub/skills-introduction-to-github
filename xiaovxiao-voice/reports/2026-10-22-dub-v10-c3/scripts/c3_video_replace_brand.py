"""Replace only the final short brand cue from a separately verified fresh take."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from c3_video_align import OUT, SR, read, write, sha, norm, dub

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);args=p.parse_args()
candidate=read(args.candidate)
audio,sr=sf.read(candidate['path'],dtype='float64')
assert sr==SR and audio.ndim==1
assert sha(candidate['path'])==candidate['sha256']
assert norm(candidate['paraformer']['text'])==norm('晓念。')
assert norm(candidate['sensevoice']['plain'])==norm('晓念。')
timestamps=candidate['paraformer']['timestamp']
start=max(0,timestamps[0][0]/1000-.05)
end=min(len(audio)/SR,timestamps[-1][1]/1000+.08)
piece=audio[round(start*SR):round(end*SR)].copy()
folder=OUT/'aligned';alignment=read(folder/'alignment.json')
wave=folder/'人声轨_48k24bit.wav'
assert sha(wave)==alignment['waveform_sha256']
track,sr=sf.read(wave,dtype='float64')
old=alignment['rows'][-1]
assert old['index']==55 and norm(old['synth'])==norm('晓念。')
old_piece=track[round(old['start']*SR):round(old['end']*SR)]
gain_db=dub.speech_rms_db(old_piece,SR)-dub.speech_rms_db(piece,SR)
piece*=10**(gain_db/20)
window=old['window'][1]-old['window'][0]
speed=max(1.,len(piece)/SR/(window-.01))
assert speed<=1.1,speed
source_piece=piece.copy()
if speed>1.005:
    piece=dub.atempo(source_piece,SR,speed)
    if len(piece)>round(window*SR):
        speed*=len(piece)/(window*SR)+.002
        assert speed<=1.1
        piece=dub.atempo(source_piece,SR,speed)
else:speed=1.
assert len(piece)<=round(window*SR)
piece=dub.fade(piece,SR,ms=2)
track[round(old['window'][0]*SR):round(old['window'][1]*SR)]=0
track[round(old['start']*SR):round(old['start']*SR)+len(piece)]=piece
assert np.max(np.abs(track))<.99
sf.write(old['cue_path'],piece,SR,subtype='PCM_24')
row={**old,'source_path':candidate['path'],'source_sha256':candidate['sha256'],
    'source_span':[start,end],'original_asr_speech_span':[timestamps[0][0]/1000,timestamps[-1][1]/1000],
    'end':old['start']+len(piece)/SR,'speed':speed,'closing_gain_db':gain_db,
    'cue_sha256':sha(old['cue_path']),'source_asr_mapping_disagreement':None,
    'brand_replacement':True,'brand_candidate_evidence':str(args.candidate.resolve())}
alignment['rows'][-1]=row
sf.write(wave,track,SR,subtype='PCM_24')
alignment.update(waveform_sha256=sha(wave),peak_dbfs=float(20*np.log10(np.max(np.abs(track)))),max_speed=max(r['speed']for r in alignment['rows']))
alignment['brand_replacement']={'previous_row':old,'candidate_evidence':candidate,'reason':'Make the short spoken brand unambiguous in the encoded final mix; other 54 cues unchanged.'}
write(folder/'alignment.json',alignment)
print({'waveform_sha256':alignment['waveform_sha256'],'brand_seconds':len(piece)/SR,'speed':speed,'gain_db':gain_db},flush=True)
