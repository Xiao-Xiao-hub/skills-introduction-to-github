"""Run the validated personal Qwen adapter as CustomVoice, no reference audio."""
from pathlib import Path
import os,sys,json,argparse,time,random,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['OMP_NUM_THREADS']='1';os.environ['ATEN_CPU_CAPABILITY']='default'

def main():
    p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--out',required=True);p.add_argument('--model-path',default=str(ROOT/'models/Qwen3-TTS-Base'));p.add_argument('--adapter-path',default=str(ROOT/'models/Qwen3-TTS-Xuan-20261024-v1-adapter'));p.add_argument('--seed',type=int,default=201);args=p.parse_args()
    import numpy as np,soundfile as sf,torch
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig,inject_adapter_in_model,set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.version.cuda=='12.8'and torch.cuda.is_available()
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(out)
    model=Qwen3TTSModel.from_pretrained(args.model_path,device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    model.model.requires_grad_(False);model.model=inject_adapter_in_model(LoraConfig.from_pretrained(args.adapter_path),model.model)
    chosen=load_file(str(Path(args.adapter_path)/'adapter_model.safetensors'),device='cuda');set_peft_model_state_dict(model.model,chosen);restored=get_peft_model_state_dict(model.model);assert set(restored)==set(chosen)and all(torch.equal(restored[k],chosen[k])for k in chosen)
    embedding=torch.load(Path(args.adapter_path)/'speaker-embedding.pt',map_location='cuda',weights_only=True)
    with torch.no_grad():model.model.talker.model.codec_embedding.weight[3000]=embedding[0].to(model.model.dtype)
    model.model.config.tts_model_type='custom_voice';model.model.tts_model_type='custom_voice';model.model.config.talker_config.spk_id={'xuan':3000};model.model.config.talker_config.spk_is_dialect={'xuan':False};model.model.supported_speakers=['xuan'];model.model.talker.config.use_cache=True;model.model.talker.code_predictor.config.use_cache=True;model.model.eval()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);start=time.monotonic()
    wavs,sr=model.generate_custom_voice(text=args.text,language='Chinese',speaker='xuan',temperature=.8,top_p=.9,max_new_tokens=1024)
    out.parent.mkdir(parents=True,exist_ok=True);sf.write(out,wavs[0],sr,subtype='PCM_24')
    info=dict(text=args.text,path=str(out),sr=sr,duration=len(wavs[0])/sr,seconds=time.monotonic()-start,sha256=hashlib.file_digest(open(out,'rb'),'sha256').hexdigest(),model_path=str(Path(args.model_path).resolve()),adapter_path=str(Path(args.adapter_path).resolve()),speaker='xuan',seed=args.seed,reference_audio_required_at_inference=False)
    out.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
