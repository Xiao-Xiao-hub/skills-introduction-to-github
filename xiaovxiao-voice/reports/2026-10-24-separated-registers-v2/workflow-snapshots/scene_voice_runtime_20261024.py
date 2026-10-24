"""Explicit formal/daily voice packages; one active adapter, no audio reference."""
from pathlib import Path
import os,json,sys,shutil,time,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['ATEN_CPU_CAPABILITY']='default'
import torch
import train_qwen_voice_20261024 as T
import split_voice_scenes_20261024 as S

def package(scene):
    src=S.BASE/scene;r=T.read(src/'training-result.json');assert r['completed'];p=T.ROOT/f'models/Qwen3-TTS-Xuan-{scene}-v2';p.mkdir(exist_ok=True)
    for name in ('adapter_model.safetensors','adapter_config.json'):shutil.copy2(Path(r['selected_checkpoint'])/name,p/name)
    shutil.copy2(src/'speaker-embedding.pt',p/'speaker-embedding.pt');shutil.copy2(Path('E:/Qwen3-TTS-training/LICENSE'),p/'LICENSE')
    meta=dict(scene=scene,speaker=r['speaker'],language=r['language'],base_model=str(T.ROOT/'models/Qwen3-TTS-Base'),adapter_sha256=r['adapter_sha256'],speaker_embedding_sha256=T.sha(p/'speaker-embedding.pt'),training_summary=T.read(src/'selection-summary.json'),snapshot_sha256=r['snapshot_sha256'],reference_audio_required_at_inference=False,merged=False,human_listening_accepted=False)
    T.write(p/'voice.json',meta);return p

def create_model(pack):
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig,inject_adapter_in_model
    meta=T.read(pack/'voice.json');v=Qwen3TTSModel.from_pretrained(meta['base_model'],device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa');v.model.requires_grad_(False);v.model=inject_adapter_in_model(LoraConfig.from_pretrained(pack),v.model);return v

def select(v,pack):
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    meta=T.read(pack/'voice.json');assert T.sha(pack/'adapter_model.safetensors')==meta['adapter_sha256'];state=load_file(str(pack/'adapter_model.safetensors'),device='cuda');set_peft_model_state_dict(v.model,state);actual=get_peft_model_state_dict(v.model);assert all(torch.equal(actual[k],state[k])for k in state)
    assert T.sha(pack/'speaker-embedding.pt')==meta['speaker_embedding_sha256']
    speaker=torch.load(pack/'speaker-embedding.pt',map_location='cuda',weights_only=True)
    with torch.no_grad():v.model.talker.model.codec_embedding.weight[3000]=speaker[0].to(v.model.dtype)
    name=meta['speaker'];v.model.config.tts_model_type='custom_voice';v.model.tts_model_type='custom_voice';v.model.config.talker_config.spk_id={name:3000};v.model.config.talker_config.spk_is_dialect={name:False};v.model.supported_speakers=[name];v.model.talker.config.use_cache=True;v.model.talker.code_predictor.config.use_cache=True;v.model.eval();return meta

def main():
    import soundfile as sf
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True,choices=['formal','daily']);parser.add_argument('--text',required=True);parser.add_argument('--out',required=True);parser.add_argument('--seed',type=int,default=201);args=parser.parse_args();T.runtime();pack=T.ROOT/f'models/Qwen3-TTS-Xuan-{args.scene}-v2';v=create_model(pack);meta=select(v,pack);T.seed(args.seed);t0=time.monotonic();wavs,sr=v.generate_custom_voice(text=args.text,language=meta['language'],speaker=meta['speaker'],temperature=.8,top_p=.9,max_new_tokens=640);p=Path(args.out).resolve()
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,wavs[0],sr,subtype='PCM_24');T.write(p.with_suffix('.json'),dict(scene=args.scene,speaker=meta['speaker'],text=args.text,sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-t0,sha256=T.sha(p),adapter_sha256=meta['adapter_sha256'],reference_audio_required_at_inference=False));print('GENERATED',str(p),flush=True)

if __name__=='__main__':main()
