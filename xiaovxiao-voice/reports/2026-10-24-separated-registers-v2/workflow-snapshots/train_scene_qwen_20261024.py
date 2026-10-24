"""Two independent Qwen LoRAs, each trained from original Base only.

Data layout/objective adapt Alibaba Qwen's Apache-2.0 trainer, retaining the
PR278/issue371 text-projection and next-frame/sub-codebook alignment fixes.
"""
import sys,time,math,random,json
from pathlib import Path
import torch,numpy as np
import train_qwen_voice_20261024 as T
import split_voice_scenes_20261024 as S

def configure(scene):
    assert scene in ('formal','daily');T.OUT=S.BASE/scene;T.SEED=20261024+(0 if scene=='formal'else 7);return T.OUT

def prepared(v,scene):
    data=T.batches(v)
    if scene=='formal':
        # Match the explicit Chinese prefix used for formal inference. The
        # upstream collator is Auto-only and places the speaker at position 6.
        for split in ('train','val','test'):
            out=[]
            for batch in data[split]:
                b={}
                for k,x in batch.items():
                    if k=='ref_mels':b[k]=x;continue
                    shape=list(x.shape);shape[1]=1;fill=-100 if k=='codec_0_labels'else 1 if k in ('text_embedding_mask','codec_embedding_mask','attention_mask')else 0;item=torch.full(shape,fill,dtype=x.dtype)
                    if k=='input_ids':item[:,:,0]=v.model.config.tts_pad_token_id;item[:,:,1]=v.model.config.talker_config.codec_language_id['chinese']
                    b[k]=torch.cat([x[:,:5],item,x[:,5:]],dim=1)
                b['input_ids'][:,3,1]=v.model.config.talker_config.codec_think_id;out.append(b)
            data[split]=out
    return data

def objective(model,batch,speaker,scene):
    b={k:v.to('cuda')for k,v in batch.items()if k!='ref_mels'}
    text=model.talker.text_projection(model.talker.model.text_embedding(b['input_ids'][:,:,0]))*b['text_embedding_mask'];codec=model.talker.model.codec_embedding(b['input_ids'][:,:,1])*b['codec_embedding_mask'];codec[:,7 if scene=='formal'else 6,:]=speaker.to('cuda',dtype=model.dtype);x=text+codec
    for i in range(1,16):x=x+model.talker.code_predictor.get_input_embeddings()[i-1](b['codec_ids'][:,:,i])*b['codec_mask'].unsqueeze(-1)
    out=model.talker(inputs_embeds=x,attention_mask=b['attention_mask'],labels=None,output_hidden_states=True,use_cache=False)
    main=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.shape[-1]),b['codec_0_labels'][:,1:].reshape(-1),ignore_index=-100);mask=b['codec_mask'][:,1:];target=b['codec_ids'][:,1:][mask];hidden=out.hidden_states[0][-1][:,:-1][mask];logits,_=model.talker.forward_sub_talker_finetune(target,hidden);sub=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),target[:,1:].reshape(-1));return main+.3*sub

@torch.no_grad()
def evaluate(model,data,speaker,scene):
    old=model.training;model.eval();values=[float(objective(model,b,speaker,scene))for b in data];model.train(old);return dict(mean=float(np.mean(values)),per_clip=values)

def train(scene,smoke=False):
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    out=configure(scene);resultpath=out/'training-result.json'
    if resultpath.exists()and T.read(resultpath)['completed']:print('ALREADY_COMPLETE',scene);return
    env=T.environment();v,config=T.load_training();model=v.model;data=prepared(v,scene);speaker=data['speaker_embedding'];params=[p for p in model.parameters()if p.requires_grad];before={k:v.detach().cpu().clone()for k,v in get_peft_model_state_dict(model).items()}
    first=data['train'][0];prefix=first['input_ids'][0,3:9 if scene=='formal'else 8,1].tolist();positions=torch.arange(first['codec_mask'].shape[1])[None,:];mask=first['codec_mask'][:,1:];assert torch.equal(positions[:,:-1][mask]+1,positions[:,1:][mask]);T.write(out/'alignment-proof.json',dict(scene=scene,codec_prefix=prefix,speaker_position=7 if scene=='formal'else 6,inference_language='Chinese'if scene=='formal'else'Auto',next_frame_hidden_alignment_asserted=True,text_projection_used=True,sub_labels_direct_15_heads=True))
    base=evaluate(model,data['val'],speaker,scene);print('BASE',scene,base['mean'],flush=True)
    if smoke:
        model.train();loss=objective(model,data['train'][0],speaker,scene);loss.backward();assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in params);T.write(out/'smoke.json',dict(passed=True,scene=scene,loss=float(loss.detach()),peak_gib=torch.cuda.max_memory_allocated()/2**30,environment=env));print('SMOKE_PASS',scene,float(loss.detach()),flush=True);return
    pool=list(range(len(data['train'])));raw=T.rows(out/'train_raw.jsonl');core=[i for i,r in enumerate(raw)if r.get('dialect_core')]
    if scene=='daily'and core:
        def core_fraction():return sum(raw[i]['duration']for i in pool if i in core)/sum(raw[i]['duration']for i in pool)
        while core_fraction()<.7 and len(pool)<3*len(raw):pool+=core
    T.write(out/'sampling.json',dict(unique_train_clips=len(raw),exposures_per_epoch=len(pool),dialect_core_indices=core,additional_exposures_are_not_new_recordings=True))
    epochs=6 if scene=='formal'else 12;accum=2 if scene=='formal'else 4;lr=1e-5 if scene=='formal'else 5e-5;steps_per=math.ceil(len(pool)/accum);total_steps=steps_per*epochs;opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/max(2,steps_per))*(.1+.45*(1+math.cos(math.pi*min(1.,s/max(total_steps,1))))));best=float('inf');bestpath=None;bestepoch=None;step=0;t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train();order=list(pool);random.Random(T.SEED+epoch).shuffle(order);opt.zero_grad(set_to_none=True)
        for pos,i in enumerate(order):
            loss=objective(model,data['train'][i],speaker,scene);assert torch.isfinite(loss);group=min(accum,len(order)-(pos//accum)*accum);(loss/group).backward()
            if (pos+1)%accum==0 or pos==len(order)-1:
                grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad);opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
            if (pos+1)%8==0 or pos==len(order)-1:
                n=epoch*len(order)+pos+1;elapsed=time.monotonic()-t0;status=dict(scene=scene,epoch=epoch+1,epochs=epochs,clip=pos+1,clips=len(order),step=step,total_steps=total_steps,elapsed_s=elapsed,eta_s=elapsed/n*(epochs*len(order)-n),loss=float(loss.detach()));T.write(out/'training-status.json',status);print('PROGRESS',json.dumps(status),flush=True)
        val=evaluate(model,data['val'],speaker,scene);meta=dict(epoch=epoch+1,step=step,validation=val,elapsed_s=time.monotonic()-t0);path=T.checkpoint(model,config,f'step_{step:06d}',meta);T.append(out/'validation-history.jsonl',meta)
        if val['mean']<best:best=val['mean'];bestpath=path;bestepoch=epoch+1
        print('VAL',scene,epoch+1,val['mean'],'BEST',bestepoch,flush=True)
    chosen=load_file(str(Path(bestpath)/'adapter_model.safetensors'),device='cuda');set_peft_model_state_dict(model,chosen);actual=get_peft_model_state_dict(model);assert all(torch.equal(actual[k],chosen[k])for k in chosen)
    model.eval()
    with T._disabled(model):base_test=evaluate(model,data['test'],speaker,scene)
    test=evaluate(model,data['test'],speaker,scene);proof=dict(tensors=len(actual),changed=sum(not torch.equal(before[k],x.detach().cpu())for k,x in actual.items()),all_finite=all(torch.isfinite(x).all().item()for x in actual.values()));torch.save(speaker,out/'speaker-embedding.pt')
    result=dict(completed=True,scene=scene,started_from='original_Qwen3-TTS-Base_no_previous_adapter',epochs=epochs,optimizer_steps=step,selected_epoch=bestepoch,selected_checkpoint=bestpath,adapter_sha256=T.sha(Path(bestpath)/'adapter_model.safetensors'),base_validation=base,best_validation=best,base_test=base_test,selected_test=test,weight_update_proof=proof,elapsed_s=time.monotonic()-t0,peak_gib=torch.cuda.max_memory_allocated()/2**30,environment=env,language='Chinese'if scene=='formal'else'Auto',speaker='xuan_'+scene,snapshot_sha256=T.sha(out/'snapshot.json'),human_listening_accepted=False);T.write(resultpath,result);print('COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    T.runtime();mode,scene=sys.argv[1:3];configure(scene)
    if mode=='encode':T.encode()
    elif mode=='smoke':train(scene,True)
    elif mode=='train':train(scene)
