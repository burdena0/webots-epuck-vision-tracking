import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    report=json.loads((ROOT/'reports/perception_audit/results.json').read_text())
    rows=[r for r in report['results'] if r['case']!='missing']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(rows))
    for method,offset,color,label in [('cnn',-0.18,'#165C91','Catalog CNN'),('hsv',0.18,'#D97926','Calibrated HSV')]:
        bars=axes[0].bar(x+offset,[r[method]['macro_iou']*100 for r in rows],0.36,color=color,label=label)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
        axes[1].plot(x,[r[method]['latency_ms']['median'] for r in rows],marker='o',color=color,label=label)
    labels=[r['case'].replace('_',' ') for r in rows]
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,110),ylabel='Macro foreground IoU (%)',title='600 matched synthetic scenes per condition')
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,1.6),ylabel='Median selected-class detector latency (ms)',title='CPU preprocessing + inference + postprocessing')
    for ax in axes:
        ax.legend(loc='lower left' if ax==axes[0] else 'upper right',frameon=False)
        ax.grid(axis='y',alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('CNN / HSV tradeoffs with frozen model weights',fontsize=15,fontweight='bold')
    fig.savefig(ROOT/'reports/perception_audit/comparison.png',dpi=180)
    fig.savefig(ROOT/'reports/perception_audit/comparison.svg')
    trial_path=ROOT/'reports/closed_loop_verified/summary.json'
    if trial_path.exists():
        trial_report=json.loads(trial_path.read_text())
        rows=trial_report['trials']
        if len(rows)==108:
            plot_trials(trial_report)

def plot_trials(report):
    import statistics
    from visionlab.evaluation import wilson
    cases=report['protocol']['cases']
    x=np.arange(len(cases))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method,offset,color,label in [('catalog',-0.18,'#165C91','Catalog CNN'),('hsv',0.18,'#D97926','Calibrated HSV')]:
        groups=[[r for r in report['trials'] if r['case']==case and r['backend']==method] for case in cases]
        successes=[sum(r['success'] for r in group) for group in groups]
        totals=[len(group) for group in groups]
        values=np.array([s/n*100 for s,n in zip(successes,totals)])
        intervals=np.array([wilson(s,n) for s,n in zip(successes,totals)])*100
        error=np.vstack([np.maximum(0,values-intervals[:,0]),np.maximum(0,intervals[:,1]-values)])
        bars=axes[0].bar(x+offset,values,0.36,color=color,label=label,yerr=error,capsize=3)
        axes[0].bar_label(bars,labels=[f'{s}/{n}' for s,n in zip(successes,totals)],padding=3,fontsize=8)
        axes[1].plot(x,[statistics.median(r['inference_median_ms'] for r in group) for group in groups],marker='o',color=color,label=label)
    labels=[case.replace('_',' ') for case in cases]
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,118),ylabel='Successful fixed configurations (%)',title='Nine paired configurations per condition')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Median of trial perception medians (ms)',title='320 x 240 Webots camera input')
    for ax in axes:
        ax.legend(loc='lower left' if ax==axes[0] else 'upper right',frameon=False)
        ax.grid(axis='y',alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('Closed-loop success with verified contact and target scoring',fontsize=14,fontweight='bold')
    fig.savefig(ROOT/'reports/closed_loop_verified/comparison.png',dpi=180)
    fig.savefig(ROOT/'reports/closed_loop_verified/comparison.svg')

if __name__=='__main__':
    main()
