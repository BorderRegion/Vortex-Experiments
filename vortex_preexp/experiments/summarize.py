"""Flatten authoritative raw results and draw a small set of diagnostic figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,write_json

COLORS={'rope_safe':'#3563a6','vortex':'#d46a32','ring_plain':'#479377','spectral_rope':'#8a5caf','vortex_post':'#d46a32','vortex_pre':'#bb986a'}


def load(name):return json.loads((ROOT/'results/raw'/name).read_text())


def main():
    out=ROOT/'results/plots';out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    rows=[]
    p1files=sorted((ROOT/'results/raw').glob('p1*.json'))
    for file in p1files:
        data=json.loads(file.read_text())
        for r in data['rows']:
            base={k:v for k,v in r.items() if k not in ['per_delta','candidate_distances']}
            rows.append(dict(stage='P1',source=file.name,**base))
            for delta,v in r.get('per_delta',{}).items():
                rows.append(dict(stage='P1_by_distance',source=file.name,**{k:v for k,v in base.items() if k not in ['logit_mae','logit_nrmse','softmax_kl','top1_change','output_nrmse']},distance=delta,**v))
        rows.extend(dict(stage='P1_commutation',source=file.name,**r) for r in data['commutation'])
    p2=[load(f'p2_{task}_{method}_s2026.json') for task in ['relative','associative'] for method in ['rope_safe','vortex','ring_plain']]
    extended=[json.loads(p.read_text()) for p in sorted((ROOT/'results/raw').glob('p2b_associative_*_s2026.json'))]
    calibrated=[json.loads(p.read_text()) for p in sorted((ROOT/'results/raw').glob('p2c_associative_*_s2026.json'))]
    staged=[('P2',p2),('P2B',extended),('P2C',calibrated)]
    for stage,variants in staged:
      for data in variants:
        baselines={(r['length'],r['distance']):r for r in data['bf16']}
        rope=next(d for d in variants if d['config']['method']=='rope_safe' and d['config']['task']==data['config']['task'])
        rope_baselines={(r['length'],r['distance']):r for r in rope['bf16']}
        spectral_baselines={(r['length'],r['distance']):r for r in data['spectral'] if r['group'] is None}
        common=dict(train_method=data['config']['method'],seed=data['seed'],
                    train_tokens=data['train_tokens'],optimizer_steps=data['optimizer_steps'],
                    additional_tokens=data.get('additional_tokens'),layers=data['model_config']['layers'],
                    parameters=data['parameters'],train_seconds=data['train_seconds'],
                    tokens_per_second=data['tokens_per_second'],peak_memory_bytes=data['peak_memory_bytes'],
                    train_data_sha256=data['train_data_sha256'],code_sha256=data['code_sha256'])
        for kind in ['bf16','ptq','spectral']:
            for r in data[kind]:
                key=r['length'],r['distance']
                base=(spectral_baselines if kind=='spectral' else baselines)[key]
                ref=rope_baselines[key]
                rows.append(dict(stage=stage,precision_set=kind,**common,**r,
                    quant_penalty_nll=r['nll']-base['nll'],accuracy_drop=base['accuracy']-r['accuracy'],
                    bf16_accuracy_delta_vs_rope=baselines[key]['accuracy']-ref['accuracy'],
                    bf16_nll_delta_vs_rope=baselines[key]['nll']-ref['nll']))
    rows.extend(dict(stage='P4',**{k:v for k,v in r.items() if not k.endswith('samples_ms')}) for r in load('p4.json')['rows'])
    replay=load('attention_replay.json')
    rows.extend(dict(stage='P1_actual_attention_replay',**r) for r in replay['rows'])
    fieldnames=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/'reports/RESULTS_TABLE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    corr=load('p0_correlation.json');fig,axs=plt.subplots(2,1,figsize=(10,6))
    axs[0].plot(corr['curve'],lw=.75,color=COLORS['vortex']);axs[0].set(xlabel='Relative distance',ylabel='tr(T(delta))/128',title='Signed rings: strong partial recurrences remain')
    for d in [1333,1334,1335]:axs[0].axvline(d,lw=.5,alpha=.4,color='red')
    frequencies=[0.]*16
    for length in corr['ring_lengths']:frequencies.extend([((2*k+1)/(2*length)) for k in range(length)])
    axs[1].hist(frequencies,bins=np.linspace(0,1,65),color=COLORS['vortex'],alpha=.85)
    axs[1].set(xlabel='Eigenphase / 2pi',ylabel='Multiplicity',title='Exact eigenphase spectrum (112 signed-ring + 16 identity modes)')
    fig.tight_layout();fig.savefig(out/'p0_correlation_spectrum.png');plt.close(fig)
    datasets=[]
    for file in p1files:
        data=json.loads(file.read_text())
        datasets.extend((file.name,name,[r for r in data['rows'] if r['distribution']==name]) for name in sorted({r['distribution'] for r in data['rows']}))
    fig,axs=plt.subplots(len(datasets),2,figsize=(10,2.7*len(datasets)),squeeze=False)
    for i,(file,name,data) in enumerate(datasets):
        for j,group in enumerate([128,16]):
            methods=['rope_safe','vortex_post','spectral_rope']
            vals=[next(r['logit_nrmse'] for r in data if r['method']==m and r['kind']=='int4' and r['group']==group and r['scope']=='qk' and r['offset']==0) for m in methods]
            ax=axs[i,j];ax.bar(['RoPE','Vortex','Spectral'],vals,color=[COLORS[m] for m in methods]);ax.set_ylabel('Logit NRMSE')
            label=name if '/' not in name else ('Relative trained Q/K' if 'relative' in name else 'Associative Q/K: '+('4-layer calibration' if 'p2c_' in name else 'extended' if 'p2b_' in name else 'original'))
            ax.set_title(f'{label} | group={group}',fontsize=9)
    fig.tight_layout();fig.savefig(out/'p1_int4_errors.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for data in p2:
        cfg=data['config'];task,method=cfg['task'],cfg['method']
        curve=[json.loads(s) for s in (ROOT/f'results/raw/p2_{task}_{method}_s2026_curve.jsonl').read_text().splitlines()]
        ax=axs[int(task=='associative')];ax.plot([x['tokens']/1e6 for x in curve],[max(x['loss'],1e-8) for x in curve],label=method,color=COLORS[method],lw=1)
        ax.set(yscale='log',xlabel='Training tokens (M)',ylabel='Query cross-entropy',title=task)
    axs[1].set(yscale='linear',ylim=(0,4.8))
    axs[1].axhline(np.log(32),color='#777777',lw=.8,ls='--',label='Uniform 32-value NLL')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'p2_training_curves.png');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for i,task in enumerate(['relative','associative']):
        for data in [d for d in p2 if d['config']['task']==task]:
            method=data['config']['method'];base=data['bf16']
            for group,style in [(None,'-'),(128,'--'),(16,':')]:
                source=base if group is None else [r for r in data['ptq'] if r['method']==method and r['group']==group and r['scope']=='qk']
                vals=[np.mean([r['accuracy'] for r in source if r['length']==length and (task=='relative' or r['distance'] in [8,32,128])])*100 for length in [256,512,1024,2048]]
                axs[i,0].plot([256,512,1024,2048],vals,style,marker='.',color=COLORS[method],label=f'{method} '+('BF16' if group is None else f'INT4 g{group}'))
            target=[r for r in base if r['length']==2048 and r['distance'] in ([1,2,8,32,128] if task=='relative' else [512,1333,1334,1335])]
            xs=[r['distance'] for r in target] if task=='relative' else [str(r['distance']) for r in target]
            axs[i,1].plot(xs,[100*r['accuracy'] for r in target],marker='o',color=COLORS[method],label=method)
        axs[i,0].set(title=task+' | common trained distances',xlabel='Evaluation length',ylabel='Accuracy (%)',ylim=(-2,102))
        axs[i,1].set(title=task+' | length=2048',xlabel='Target distance',ylabel='BF16 accuracy (%)',ylim=(-2,102))
    for ax in axs.flat:ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'p2_accuracy.png');plt.close(fig)
    if extended:
        fig,axs=plt.subplots(1,3,figsize=(15,4))
        for data in extended:
            method=data['config']['method']
            curve=[json.loads(s) for s in (ROOT/f'results/raw/p2b_associative_{method}_s2026_curve.jsonl').read_text().splitlines()]
            axs[0].plot([r['tokens']/1e6 for r in curve],[max(r['loss'],1e-8) for r in curve],label=method,color=COLORS[method])
            for group,style in [(None,'-'),(128,'--'),(16,':')]:
                source=data['bf16'] if group is None else [r for r in data['ptq'] if r['group']==group and r['scope']=='qk']
                vals=[np.mean([r['accuracy'] for r in source if r['length']==length and r['distance'] in [8,32,128]])*100 for length in [256,512,1024,2048]]
                axs[1].plot([256,512,1024,2048],vals,style,marker='.',color=COLORS[method],label=f'{method} '+('BF16' if group is None else f'g{group}'))
            source=[r for r in data['bf16'] if r['length']==2048 and r['distance'] in [512,1333,1334,1335]]
            axs[2].plot([str(r['distance']) for r in source],[r['accuracy']*100 for r in source],marker='o',label=method,color=COLORS[method])
        axs[0].set(xlabel='Cumulative tokens (M)',ylabel='Query cross-entropy',ylim=(0,4.8),title='Common continuation: original states retained')
        axs[0].axhline(np.log(32),color='#777777',lw=.8,ls='--',label='Uniform 32-value NLL')
        axs[1].set(xlabel='Length',ylabel='Accuracy (%)',ylim=(-2,102),title='Associative: same trained distances')
        axs[2].set(xlabel='Target distance',ylabel='BF16 accuracy (%)',ylim=(-2,102),title='Associative: length 2048')
        for ax in axs:ax.legend(fontsize=6);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'p2b_associative.png');plt.close(fig)
    if calibrated:
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for data in calibrated:
            method=data['config']['method']
            curve=[json.loads(s) for s in (ROOT/f'results/raw/p2c_associative_{method}_s2026_curve.jsonl').read_text().splitlines()]
            axs[0].plot([r['tokens']/1e6 for r in curve],[r['loss'] for r in curve],label=method,color=COLORS[method])
            for group,style in [(None,'-'),(128,'--'),(16,':')]:
                source=data['bf16'] if group is None else [r for r in data['ptq'] if r['group']==group and r['scope']=='qk' and r['method']==method]
                vals=[np.mean([r['accuracy'] for r in source if r['length']==length and r['distance'] in [8,32,128]])*100 for length in [256,512,1024,2048]]
                axs[1].plot([256,512,1024,2048],vals,style,marker='.',color=COLORS[method],label=f'{method} '+('BF16' if group is None else f'g{group}'))
        gate=load('p2c_gate.json')
        axs[0].axhline(np.log(32),color='#777777',lw=.8,ls='--',label='Uniform 32-value NLL')
        axs[0].set(xlabel='Training tokens (M)',ylabel='Query cross-entropy',ylim=(0,4.8),title='Four-layer RoPE control: '+('PASS' if gate['passed'] else 'FAIL'))
        axs[1].set(xlabel='Length',ylabel='Accuracy (%)',ylim=(-2,102),title='Associative: distances 8 / 32 / 128')
        for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'p2c_calibration.png');plt.close(fig)
    p4=load('p4.json')['rows'];fig,axs=plt.subplots(1,2,figsize=(10,4))
    for method in ['rope_safe','vortex']:
        for j,path in enumerate(['prefill_attention','decode_attention']):
            data=[r for r in p4 if r['method']==method and r['path']==path]
            axs[j].plot([r['sequence'] for r in data],[r['median_ms'] for r in data],marker='o',label=method,color=COLORS[method])
            axs[j].set(title=path,xlabel='Sequence / cache length',ylabel='Synchronized wall median (ms)')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'p4_latency.png');plt.close(fig)
    print('Wrote RESULTS_TABLE.csv and diagnostic figures')


if __name__=='__main__':main()
