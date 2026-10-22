"""Identity diagnostics; these criteria do not measure emotional naturalness."""
import numpy as np

def evaluate(vectors, cache, plan):
    ids=list(vectors);v=np.array([vectors[g] for g in ids]);pair=v@v.T
    pairs=pair[np.triu_indices(len(ids),1)]
    calm=next(p['sha256'] for p in plan['profiles'] if p['id']=='warm')
    anchor=np.array(cache[calm]['embedding'])
    expression={g:float(vectors[g]@np.array(cache[plan['expression_bank'][plan['group_bank'][g]]['sha256']]['embedding'])) for g in ids}
    medians={g:float(np.median(np.delete(pair[i],i))) for i,g in enumerate(ids)}
    failed=[]
    for g in ids:
        if expression[g]<.85:failed.append({'group':g,'check':'expression_prompt','value':expression[g]})
        if medians[g]<.85:failed.append({'group':g,'check':'peer_median','value':medians[g]})
    for i,j in zip(*np.triu_indices(len(ids),1)):
        if pair[i,j]<.80:failed.append({'groups':[ids[i],ids[j]],'check':'pairwise_min','value':float(pair[i,j])})
    if float(np.percentile(pairs,10))<.85:failed.append({'check':'pairwise_p10','value':float(np.percentile(pairs,10))})
    return dict(passed=not failed,expression_prompt_scores=expression,peer_medians=medians,
                calm_reference_scores={g:float(vectors[g]@anchor) for g in ids},
                pairwise_min=float(pairs.min()),pairwise_p10=float(np.percentile(pairs,10)),pairwise_mean=float(pairs.mean()),
                criteria=dict(expression_prompt_min=.85,peer_median_min=.85,pairwise_p10_min=.85,pairwise_min=.80),
                failures=failed,human_identity_approval=False,naturalness_evaluated=False)
