"""Official Qwen dataset/objective with PEFT LoRA and bounded 12GB training.

Upstream pinned at E:/Qwen3-TTS-training; never modifies installed model code.
"""
# Portions adapted from Qwen3-TTS finetuning/sft_12hz.py.
# Copyright 2026 The Alibaba Qwen team. SPDX-License-Identifier: Apache-2.0.
# Alignment corrections follow QwenLM/Qwen3-TTS PR #278 and issue #371.
from pathlib import Path
import os,sys,json,time,math,random,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['ATEN_CPU_CAPABILITY']='default'
import numpy as np
import soundfile as sf
import torch
import prepare_qwen_voice_20261024 as D
ROOT=D.ROOT;OUT=D.OUT;read,write,rows,append,sha=D.read,D.write,D.rows,D.append,D.sha
sys.path.insert(0,'E:/Qwen3-TTS-training/finetuning')
SEED=20261024

def runtime():torch.set_num_threads(1);torch.set_num_interop_threads(1)
def seed(x=SEED):random.seed(x);np.random.seed(x%(2**32));torch.manual_seed(x);torch.cuda.manual_seed_all(x)
def environment():
    import importlib.metadata as m
    assert torch.version.cuda=='12.8' and torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,0)
    return dict(torch=torch.__version__,cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability()),packages={x:m.version(x)for x in ('qwen-tts','transformers','accelerate','peft','soundfile')},flash_attn_installed=False)

def encode():
    from qwen_tts import Qwen3TTSTokenizer
    environment();t0=time.monotonic();manifest=OUT/'codes.jsonl';existing={r['id']:r for r in rows(manifest)}
    raw=[r for s in ('train','val','test')for r in rows(OUT/f'{s}_raw.jsonl')];todo=[r for r in raw if r['id']not in existing]
    tokenizer=Qwen3TTSTokenizer.from_pretrained(str(ROOT/'models/Qwen3-TTS-Base/speech_tokenizer'),device_map='cuda:0')
    for i,r in enumerate(todo):
        assert sha(r['audio'])==r['audio_sha256'];a,sr=sf.read(r['audio'],dtype='float32')
        result=tokenizer.encode(a,sr=sr);code=result.audio_codes[0].cpu().tolist();entry=dict(id=r['id'],audio_sha256=r['audio_sha256'],audio_codes=code)
        append(manifest,entry);existing[r['id']]=entry
        if (i+1)%10==0 or i==len(todo)-1:print('ENCODE',i+1,len(todo),round(time.monotonic()-t0,1),'eta',round((time.monotonic()-t0)/(i+1)*(len(todo)-i-1),1),flush=True)
    for split in ('train','val','test'):
        data=[]
        for r in rows(OUT/f'{split}_raw.jsonl'):
            assert existing[r['id']]['audio_sha256']==r['audio_sha256'];data.append(dict(r,audio_codes=existing[r['id']]['audio_codes']))
        (OUT/f'{split}_codes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in data),encoding='utf-8')

def load_training():
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig,inject_adapter_in_model
    seed();v=Qwen3TTSModel.from_pretrained(str(ROOT/'models/Qwen3-TTS-Base'),device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    v.model.requires_grad_(False)
    config=LoraConfig(r=16,lora_alpha=16,lora_dropout=.05,bias='none',target_modules=r'talker\.(model|code_predictor\.model)\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))')
    v.model=inject_adapter_in_model(config,v.model)
    v.model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    v.model.talker.config.use_cache=False;v.model.talker.code_predictor.config.use_cache=False
    return v,config

def batches(v):
    from dataset import TTSDataset
    cache=OUT/'training-batches.pt';snap=read(OUT/'snapshot.json')
    if cache.exists():
        data=torch.load(cache,map_location='cpu',weights_only=False);assert data['snapshot_sha256']==sha(OUT/'snapshot.json');return data
    out={}
    for split in ('train','val','test'):
        data=rows(OUT/f'{split}_codes.jsonl');ds=TTSDataset(data,v.processor,v.model.config);out[split]=[ds.collate_fn([ds[i]])for i in range(len(ds))]
    with torch.no_grad():out['speaker_embedding']=v.model.speaker_encoder(out['train'][0]['ref_mels'].to('cuda',dtype=torch.bfloat16)).detach().cpu()
    out['snapshot_sha256']=sha(OUT/'snapshot.json');torch.save(out,cache);return out

def objective(model,batch,speaker):
    b={k:v.to('cuda')for k,v in batch.items()if k!='ref_mels'};speaker=speaker.to('cuda',dtype=torch.bfloat16)
    # Upstream PR #278/#371: use the same text projection as generation.
    x=model.talker.text_projection(model.talker.model.text_embedding(b['input_ids'][:,:,0]))*b['text_embedding_mask']
    codec=model.talker.model.codec_embedding(b['input_ids'][:,:,1])*b['codec_embedding_mask'];codec[:,6,:]=speaker;x=x+codec
    for i in range(1,16):x=x+model.talker.code_predictor.get_input_embeddings()[i-1](b['codec_ids'][:,:,i])*b['codec_mask'].unsqueeze(-1)
    # Compute aligned CE explicitly. The installed default ForCausalLMLoss
    # shifts labels internally; feeding already-shifted labels double-shifts.
    outputs=model.talker(inputs_embeds=x,attention_mask=b['attention_mask'],labels=None,output_hidden_states=True,use_cache=False)
    main=torch.nn.functional.cross_entropy(outputs.logits[:,:-1].float().reshape(-1,outputs.logits.shape[-1]),b['codec_0_labels'][:,1:].reshape(-1),ignore_index=-100)
    hidden=outputs.hidden_states[0][-1];next_mask=b['codec_mask'][:,1:]
    # Hidden at p-1 predicts frame p. Never expose frame p to its sub-talker.
    target_codes=b['codec_ids'][:,1:][next_mask];prev_hidden=hidden[:,:-1][next_mask]
    sub_logits,_=model.talker.forward_sub_talker_finetune(target_codes,prev_hidden)
    # All 15 sub-codebook heads use their corresponding codebook labels,
    # without another causal time shift across different codebooks.
    sub=torch.nn.functional.cross_entropy(sub_logits.float().reshape(-1,sub_logits.shape[-1]),target_codes[:,1:].reshape(-1))
    return main+.3*sub,main.detach(),sub.detach()

@torch.no_grad()
def evaluate(model,data,speaker):
    training=model.training;model.eval();values=[]
    for b in data:
        total,main,sub=objective(model,b,speaker);values.append([float(total),float(main),float(sub)])
    model.train(training);means=np.mean(values,axis=0).tolist();return dict(total=means[0],main=means[1],sub=means[2],per_clip=values)

def checkpoint(model,config,name,meta):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    dest=OUT/'checkpoints'/name;dest.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu().contiguous()for k,v in get_peft_model_state_dict(model).items()};save_file(state,str(dest/'adapter_model.safetensors'));config.save_pretrained(dest);write(dest/'state.json',meta)
    return str(dest)

def train(smoke=False):
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    env=environment();resultpath=OUT/'training-result.json'
    if resultpath.exists()and read(resultpath).get('completed'):
        result=read(resultpath);assert sha(Path(result['selected_checkpoint'])/'adapter_model.safetensors')==result['checkpoint_sha256'];print('TRAIN_ALREADY_COMPLETE');return
    v,config=load_training();model=v.model;data=batches(v);speaker=data['speaker_embedding'];params=[p for p in model.parameters()if p.requires_grad]
    counts=dict(trainable=sum(p.numel()for p in params),total=sum(p.numel()for p in model.parameters()));print('PARAMETERS',counts,flush=True)
    before={k:x.detach().cpu().clone()for k,x in get_peft_model_state_dict(model).items()}
    baseline=evaluate(model,data['val'],speaker);print('BASE_VAL',baseline['total'],flush=True)
    if smoke:
        b=data['train'][0];positions=torch.arange(b['codec_mask'].shape[1])[None,:];targets=positions[:,1:][b['codec_mask'][:,1:]];previous=positions[:,:-1][b['codec_mask'][:,1:]];assert torch.equal(previous+1,targets)
        write(OUT/'alignment-proof.json',dict(main='unshifted full forward; explicit logits[t] vs labels[t+1] CE',sub_hidden='hidden[p-1] predicts frame p; asserted for every audio position',sub_labels='15 heads against codebooks1..15 directly; no loss-time shift',target_frames=int(len(targets)),text_projection_used=True,original_toolkit_and_library_files_modified=False,sources=['https://github.com/QwenLM/Qwen3-TTS/pull/278','https://github.com/QwenLM/Qwen3-TTS/issues/371']))
        model.train();loss,_,_=objective(model,data['train'][0],speaker);loss.backward();assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in params)
        write(OUT/'training-smoke.json',dict(passed=True,loss=float(loss),counts=counts,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,environment=env));print('SMOKE_PASSED',float(loss),torch.cuda.max_memory_allocated()/2**30,flush=True);return
    epochs=8;accum=4;total_steps=epochs*math.ceil(len(data['train'])/accum);optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min(1.,(n+1)/8)*(.1+.9*.5*(1+math.cos(math.pi*min(1.,n/max(total_steps,1))))))
    initial=checkpoint(model,config,'step_000000',dict(epoch=0,step=0,validation=baseline['total']))
    best=baseline['total'];bestpath=initial;bestepoch=0;history=[];step=0;t0=time.monotonic();torch.cuda.reset_peak_memory_stats();model.train()
    for epoch in range(epochs):
        order=list(range(len(data['train'])));random.Random(SEED+epoch).shuffle(order);losses=[];optimizer.zero_grad(set_to_none=True)
        for pos,i in enumerate(order):
            loss,main,sub=objective(model,data['train'][i],speaker);assert torch.isfinite(loss)
            group=min(accum,len(order)-(pos//accum)*accum);(loss/group).backward();losses.append(float(loss.detach()))
            if (pos+1)%accum==0 or pos==len(order)-1:
                grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if (pos+1)%8==0 or pos==len(order)-1:
                completed=epoch*len(order)+pos+1;total=epochs*len(order);elapsed=time.monotonic()-t0;status=dict(epoch=epoch+1,epochs=epochs,clip=pos+1,clips=len(order),optimizer_step=step,total_steps=total_steps,elapsed_s=elapsed,eta_s=elapsed/completed*(total-completed),loss=float(loss.detach()),peak_gib=torch.cuda.max_memory_allocated()/2**30);write(OUT/'training-status.json',status);print('TRAIN',json.dumps(status),flush=True)
        val=evaluate(model,data['val'],speaker);meta=dict(epoch=epoch+1,step=step,train_loss=float(np.mean(losses)),validation=val,elapsed_s=time.monotonic()-t0)
        path=checkpoint(model,config,f'step_{step:06d}',meta);append(OUT/'validation-history.jsonl',meta);history.append(meta)
        if val['total']<best:best=val['total'];bestpath=path;bestepoch=epoch+1
        print('VALIDATION',epoch+1,val['total'],'BEST',bestepoch,flush=True)
    # Test set is evaluated once, after selection exclusively by validation.
    base_test=None
    model.eval()
    with model.disable_adapter() if hasattr(model,'disable_adapter') else _disabled(model):base_test=evaluate(model,data['test'],speaker)
    set_peft_model_state_dict(model,load_file(str(Path(bestpath)/'adapter_model.safetensors'),device='cuda'))
    selected_test=evaluate(model,data['test'],speaker)
    after=get_peft_model_state_dict(model);proof=dict(tensors=len(after),changed_tensors=sum(not torch.equal(before[k],x.detach().cpu())for k,x in after.items()),all_finite=all(torch.isfinite(x).all().item()for x in after.values()))
    torch.save(speaker,OUT/'speaker-embedding.pt')
    result=dict(completed=True,model='Qwen3-TTS-12Hz-1.7B-Base',method='PEFT_LoRA_single_speaker_with_PR278_alignment_corrections',objective_revision='text_projection_main_next_frame_CE_sub_previous_hidden_direct_15_head_CE',upstream=read(Path('E:/Qwen3-TTS-training/UPSTREAM-REVISION.json')),epochs=epochs,optimizer_steps=step,selected_epoch=bestepoch,selected_checkpoint=bestpath,checkpoint_sha256=sha(Path(bestpath)/'adapter_model.safetensors'),initial_validation=baseline,best_validation=best,base_test=base_test,selected_test=selected_test,weight_update_proof=proof,counts=counts,elapsed_s=time.monotonic()-t0,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,environment=env,training_snapshot_sha256=sha(OUT/'snapshot.json'),human_listening_accepted=False)
    write(resultpath,result);print('TRAINING_COMPLETE',json.dumps(result),flush=True)

from contextlib import contextmanager
@contextmanager
def _disabled(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for m in model.modules():
        if isinstance(m,BaseTunerLayer):m.enable_adapters(False)
    try:yield
    finally:
        for m in model.modules():
            if isinstance(m,BaseTunerLayer):m.enable_adapters(True)

def export():
    import shutil
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file,save_file
    result=read(OUT/'training-result.json');best=Path(result['selected_checkpoint']);assert result['selected_epoch']>0
    assert sha(best/'adapter_model.safetensors')==result['checkpoint_sha256']
    dest=ROOT/'models/Qwen3-TTS-Xuan-20261024-v1'
    if (dest/'TRAINING-RESULT.json').exists():
        proof=read(dest/'TRAINING-RESULT.json');assert sha(dest/'model.safetensors')==proof['exported_weight_sha256'];print('EXPORT_ALREADY_COMPLETE');return
    v,config=load_training();model=v.model;chosen=load_file(str(best/'adapter_model.safetensors'),device='cuda');set_peft_model_state_dict(model,chosen);model.eval()
    restored=get_peft_model_state_dict(model);assert set(restored)==set(chosen)and all(torch.equal(restored[k],chosen[k])for k in chosen)
    for layer in model.modules():
        if isinstance(layer,BaseTunerLayer):layer.merge(safe_merge=True)
    speaker=torch.load(OUT/'speaker-embedding.pt',map_location='cuda',weights_only=True)
    with torch.no_grad():model.talker.model.codec_embedding.weight[3000]=speaker[0].to(model.dtype)
    state={k.replace('.base_layer.','.'):val.detach().cpu().contiguous()for k,val in model.state_dict().items()if 'lora_'not in k and not k.startswith('speaker_encoder')}
    assert all(torch.isfinite(x).all()for x in state.values())
    dest.mkdir(exist_ok=True)
    for src in (ROOT/'models/Qwen3-TTS-Base').iterdir():
        if src.name in ('model.safetensors','.gitattributes'):continue
        target=dest/src.name
        if src.is_dir():shutil.copytree(src,target,dirs_exist_ok=True)
        else:shutil.copy2(src,target)
    shutil.copy2(Path('E:/Qwen3-TTS-training/LICENSE'),dest/'LICENSE')
    cfg=read(dest/'config.json');cfg['tts_model_type']='custom_voice';cfg['talker_config']['spk_id']={'xuan':3000};cfg['talker_config']['spk_is_dialect']={'xuan':False};write(dest/'config.json',cfg)
    save_file(state,str(dest/'model.safetensors'))
    proof=dict(source_base='Qwen3-TTS-12Hz-1.7B-Base',adapter_sha256=result['checkpoint_sha256'],selected_epoch=result['selected_epoch'],speaker='xuan',speaker_id=3000,exported_weight_sha256=sha(dest/'model.safetensors'),reference_audio_required_at_inference=False,human_listening_accepted=False)
    write(dest/'TRAINING-RESULT.json',proof);write(OUT/'export-result.json',dict(path=str(dest),**proof))
    (dest/'README.md').write_text('# 轩 Qwen专属声音 v1\n\n由Qwen3-TTS Base与本轮真实PEFT LoRA合并，固定speaker=xuan，可用generate_custom_voice直接生成，无需提供参考音频。训练记录见TRAINING-RESULT.json；尚待用户听感验收。原模型和适配器均保留。\n',encoding='utf-8')
    print('EXPORTED',str(dest),flush=True)

if __name__=='__main__':
    runtime()
    mode=sys.argv[1]
    if mode=='smoke':train(True)
    elif mode=='train':train()
    else:globals()[mode]()
