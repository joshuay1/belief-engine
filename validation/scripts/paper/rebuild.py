"""Rebuild the camera-ready numerical results from saved inputs, without model calls."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'validation/camera_ready'
sys.path.insert(0, str(Path(__file__).parent / 'frozen'))


def compare(actual, expected, keys, columns=None, atol=1e-10):
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(e), (len(a), len(e))
    pd.testing.assert_frame_equal(a[keys].astype(str), e[keys].astype(str), check_dtype=False)
    for col in columns or list(e.select_dtypes(include='number').columns):
        np.testing.assert_allclose(pd.to_numeric(a[col]), pd.to_numeric(e[col]), atol=atol, rtol=1e-9, equal_nan=True, err_msg=col)


def save(out, name, data):
    if isinstance(data, pd.DataFrame):
        data.to_csv(out / (name + '.csv'), index=False)
    else:
        (out / (name + '.json')).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def stance(out):
    import stance_alignment as code
    src = DATA / 'stance_alignment'
    config = yaml.safe_load((src / 'resolved_config.yaml').read_text())
    result = code.analyse_results(src / 'results.csv', out, config['analysis'], config['study']['random_seed'])
    expected = json.loads((src / 'expected/metrics.json').read_text())
    for name in ['pearson_r', 'slope', 'mae', 'rmse', 'pearson_r_ci95', 'slope_ci95']:
        np.testing.assert_allclose(result['pooled'][name], expected['pooled'][name], atol=1e-12)
    compare(pd.read_csv(out/'topic_metrics.csv'), pd.read_csv(src/'expected/topic_metrics.csv'), ['topic'])
    assert result['n_successful_raw_rows'] == 630 and result['n_cells'] == 210


def baselines(out):
    import analyze_matched_baseline as code
    import analyze_matched_baseline_primary_multiplicity as multi
    src = DATA / 'baselines'
    trajectories = code.validate_trajectories(pd.read_csv(src/'trajectories.csv'), expected_final_round=15)
    runs = code.compute_run_metrics(trajectories, expected_final_round=15)
    code.require_complete_pairs(runs)
    saved = pd.read_csv(src/'runs.csv')
    assert len(runs) == len(saved) == 150
    with gzip.open(src/'generated_dialogues.jsonl.gz','rt') as handle:
        dialogues=[json.loads(line) for line in handle]
    assert len(dialogues)==150 and {d['run_id'] for d in dialogues}==set(saved.run_id)
    for dialogue in dialogues:
        assert len(dialogue['transcript'])==30
        final=dialogue['stance_history'][-1]
        row=saved[saved.run_id==dialogue['run_id']].iloc[0]
        np.testing.assert_allclose([final['Pro_evaluated'],final['Con_evaluated']],[row.pro_final,row.con_final],atol=1e-12)
    compare(runs, saved.rename(columns={'final_distance':'final_gap'}), ['run_id'], ['final_gap'])
    # Preserve the original per-condition child-seed allocation, including the
    # positions assigned to other metrics, while rebuilding only final gap.
    children = np.random.SeedSequence(20260723).spawn(len(code.EXPECTED_CONDITIONS)*len(code.METRICS))
    rows=[]
    for i, condition in enumerate(code.EXPECTED_CONDITIONS):
        subset=runs[runs.condition==condition]
        rng=np.random.default_rng(children[i*len(code.METRICS)+code.METRICS.index('final_gap')])
        draws=code.hierarchical_bootstrap(subset,'final_gap',replicates=10000,rng=rng)
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(condition=condition,estimate=subset.groupby('topic').final_gap.mean().mean(),ci_lower=lo,ci_upper=hi))
    conditions=pd.DataFrame(rows)
    compare(conditions,pd.read_json(src/'expected/condition_intervals.json'),['condition'],['estimate','ci_lower','ci_upper'])
    save(out,'condition_intervals',conditions)
    topic_table=runs.pivot_table(index='topic',columns='condition',values='final_gap',aggfunc='mean').reset_index()
    save(out,'topic_gaps',topic_table)
    spec=yaml.safe_load((src/'MATCHED_BASELINE_PRIMARY_MULTIPLICITY_POSTHOC_SPEC.yaml').read_text())
    topics=spec['expected_design']['topics']
    # Recreate the locked topic/seed order in the original multiplicity analysis.
    cube=np.empty((5,10,2))
    for t,topic in enumerate(topics):
        part=saved[saved.topic_slug==topic['slug']]
        seeds=sorted(part.trial.unique())
        for j,seed in enumerate(seeds):
            block=part[part.trial==seed].set_index('condition').final_distance
            for k,other in enumerate(code.COMPARATORS):cube[t,j,k]=block['be_gpt54mini']-block[other]
    draws=multi.hierarchical_bootstrap(cube,replicates=100000,seed=2026072507,chunk_size=2000)
    intervals=multi.bootstrap_table(cube,draws,spec)
    compare(intervals,pd.read_csv(src/'expected/bootstrap_intervals.csv'),['comparison'])
    save(out,'contrast_intervals',intervals)
    _,tests,_=multi.exact_sign_flip_tables(cube,topics)
    compare(tests,pd.read_csv(src/'expected/exact_topic_sign_flip_tests.csv'),['comparison'])
    save(out,'topic_sign_tests',tests)
    leave,_=multi.leave_one_topic_out_tables(cube,topics)
    compare(leave,pd.read_csv(src/'expected/leave_one_topic_out.csv'),['comparison','omitted_topic_slug'])
    save(out,'leave_one_topic_out',leave)


def endpoints(out):
    import analyze_exact_endpoint_trajectories as code
    code.validate_spec(DATA/'endpoints/spec.yaml')
    participants,_=code.load_blind_participants(DATA/'replay')
    _,denominator,matches,flow=code.analyze_participants(participants)
    assert len(denominator)==2441 and len(matches)==958
    summary=pd.DataFrame(code.summarize(denominator))
    compare(summary,pd.read_csv(DATA/'endpoints/expected/summary.csv'),['dimension','level'])
    save(out,'summary',summary)
    print('  endpoint roots verified; rebuilding clustered intervals',flush=True)
    intervals=pd.DataFrame(code.bootstrap_intervals(denominator,100000))
    expected=pd.read_csv(DATA/'endpoints/expected/bootstrap_intervals.csv')
    keys=[c for c in ['resampling_scheme','scheme','metric'] if c in expected]
    compare(intervals,expected,keys)
    save(out,'bootstrap_intervals',intervals)
    _,traces=code.select_examples(denominator,{p.participant_record_id:p for p in participants})
    traces=pd.DataFrame(traces)
    compare(traces,pd.read_csv(DATA/'endpoints/expected/selected_example_trajectories.csv'),['orientation','step'])
    save(out,'example_trajectories',traces)
    save(out,'flow',flow)
    plot_endpoints(traces,out)


def plot_endpoints(traces,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7,3),sharey=True,layout='constrained')
    for ax,which in zip(axes,['prior_consistent','prior_opposing']):
        d=traces[traces.orientation==which]
        for prefix,color in [('reference','#167783'),('variable','#d67b1c')]:
            ax.plot(d.step,d[prefix+'_stance'],color=color,label=prefix)
        ax.set(title=which.replace('_',' '),xlabel='Accepted evidence items',ylim=(-1.05,1.05))
        ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Maintained stance');fig.savefig(out/'endpoint_paths.pdf');plt.close(fig)


def components(out):
    import analyze_component_necessity_factorial as code
    src=DATA/'components';frame=pd.read_csv(src/'cells.csv')
    assert len(frame)==500 and frame.groupby(['topic_slug','trial']).size().eq(10).all()
    np.testing.assert_allclose(frame.final_gap,abs(frame.pro_final-frame.con_final),atol=1e-12)
    # Only final-gap contrasts are reported in this appendix.
    original_metrics=code.METRICS;code.METRICS=('final_gap',)
    units=code.unit_contrasts(frame.to_dict('records'),code.PRIMARY)
    summaries,topics=code.summarize_contrasts(units,100000,20260725)
    code.METRICS=original_metrics
    summaries=pd.DataFrame(summaries);expected=pd.read_csv(src/'expected/primary_contrasts.csv')
    expected=expected[expected.metric=='final_gap']
    compare(summaries,expected,['contrast','metric'])
    save(out,'contrasts',summaries);save(out,'topic_contrasts',pd.DataFrame(topics))
    full=frame[(frame.S==1)&frame.condition.isin(code.CORE_CONDITIONS)]
    flat=frame[(frame.S==0)&frame.condition.isin(code.CORE_CONDITIONS)]
    mean=full.extraction_records_emitted.sum()/full.extraction_event_count.sum()
    assert round(mean,2)==6.34
    assert flat.extraction_records_emitted.sum()==flat.extraction_event_count.sum()
    save(out,'decomposition',{'structured_claims_per_message':mean,'whole_message_records_per_message':1.0})


def aqr(out):
    d=pd.read_csv(DATA/'aqr/selected_100.csv')
    assert len(d)==100 and d.text_hash.is_unique and d.classifier_audit_row.tolist()==list(range(100))
    np.testing.assert_allclose(d.quality_target,(d.WA+d['MACE-P'])/2,atol=1e-12)
    extracted=d[d.n_extracted>0]
    correct=(extracted.human_polarity==extracted.pred_polarity_strongest).sum()
    result={'inputs':len(d),'extracted':len(extracted),'single_claim':int((d.n_extracted==1).sum()),'multiple_claims':int((d.n_extracted>1).sum()),'correct_polarity':int(correct),'polarity_accuracy':correct/len(extracted),'classifier_r':d.classifier_strength.corr(d.quality_target),'llm_r':extracted.strongest_strength_llm.corr(extracted.quality_target)}
    assert result['extracted']==97 and result['correct_polarity']==94
    np.testing.assert_allclose(result['classifier_r'],.7195065558642759,atol=1e-12)
    np.testing.assert_allclose(result['llm_r'],.0892009353,atol=1e-10)
    save(out,'metrics',result)


def checks(out):
    import replay_uncertainty as code
    predictions=pd.read_csv(DATA/'replay/predictions.csv.gz').rename(columns={'group_id':'group_file','topic_id':'topic'})
    actual=code.paired_cluster_bootstrap(predictions)
    expected=pd.read_csv(DATA/'replay/expected/paired_cluster_bootstrap.csv')
    compare(actual,expected,['split','contrast','metric'],['observed_delta'])
    # Released cluster IDs are hashed. Sorting them changes bootstrap draw
    # assignments, so compare point estimates exactly and the reported
    # interval conclusions rather than claiming identical Monte Carlo draws.
    for metric in ['rmse','mae']:
        for contrast in ['classifier_stance_linear_minus_classifier_log1p','uniform1_log1p_minus_classifier_log1p']:
            a=actual[(actual.metric==metric)&(actual.contrast==contrast)].sort_values('split')
            e=expected[(expected.metric==metric)&(expected.contrast==contrast)].sort_values('split')
            assert ((a.ci_lower.to_numpy()<=0)&(a.ci_upper.to_numpy()>=0)).tolist()==((e.ci_lower.to_numpy()<=0)&(e.ci_upper.to_numpy()>=0)).tolist()
    save(out,'scoring_updater_intervals',actual)
    # Natural deduplication: recompute the paired loss contrasts reported in prose
    # from frozen out-of-fold predictions. Saved uncertainty accompanies the data.
    p=pd.read_csv(DATA/'checks/natural_dedup_predictions.csv.gz')
    rows=[]
    for (split,strategy),d in p.groupby(['split','parameter_strategy']):
        reference=d[d.stream_cell=='registered_classifier_dedup'].set_index('participant_record_id')
        for cell,dd in d.groupby('stream_cell'):
            if cell=='registered_classifier_dedup':continue
            dd=dd.set_index('participant_record_id').loc[reference.index]
            np.testing.assert_allclose(dd.opinion_change_actual,reference.opinion_change_actual)
            e=dd.opinion_change_predicted-dd.opinion_change_actual
            r=reference.opinion_change_predicted-reference.opinion_change_actual
            for metric,v,w in [('squared_error',(e**2).mean(),(r**2).mean()),('rmse',np.sqrt((e**2).mean()),np.sqrt((r**2).mean())),('absolute_error',abs(e).mean(),abs(r).mean()),('mae',abs(e).mean(),abs(r).mean())]:
                rows.append(dict(split=split,parameter_strategy=strategy,stream_cell=cell,stratum='all',metric=metric,observed_delta_stream_minus_registered=v-w))
    actual=pd.DataFrame(rows);expected=pd.read_csv(DATA/'checks/expected/natural_paired_contrasts.csv')
    expected=expected[(expected.stratum=='all')&expected.metric.isin(actual.metric.unique())]
    compare(actual,expected,['split','parameter_strategy','stream_cell','stratum','metric'],['observed_delta_stream_minus_registered'])
    save(out,'natural_dedup_contrasts',actual)
    # Controlled test: independently reconstruct the recorded active-set sum.
    fixture=json.loads((DATA/'checks/controlled_results.json').read_text())
    expected=pd.read_csv(DATA/'checks/controlled_result_table.csv');rows=[]
    for r in fixture['rows']:
        multiplicity=r['paraphrase_multiplicity']
        polarity=-1 if r['trace_direction']=='pro_control_then_anti_paraphrases' else 1
        contribution=r['single_record_log_odds_contribution']
        log_odds=0.0 if r['condition']=='full_dedup' else polarity*(multiplicity-1)*contribution
        rows.append({'trace_direction':r['trace_direction'],'paraphrase_multiplicity':multiplicity,'condition':r['condition'],'signed_log_odds':log_odds,'signed_stance':np.tanh(log_odds/2)})
    actual=pd.DataFrame(rows)
    compare(actual,expected,['trace_direction','paraphrase_multiplicity','condition'],['signed_log_odds','signed_stance'])
    save(out,'controlled_dedup',actual)


STUDIES={'stance':stance,'baselines':baselines,'endpoints':endpoints,'components':components,'aqr':aqr,'checks':checks}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=['all',*STUDIES],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/paper')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for record in json.loads((DATA/'provenance.json').read_text())['sources']:
        path=ROOT/record['destination']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['destination_sha256'], str(path)
    completed=[]
    for name,fn in STUDIES.items():
        if args.study!='all' and args.study!=name:continue
        out=args.output_dir/name;out.mkdir(parents=True,exist_ok=True)
        print('Rebuilding '+name+' ...',flush=True);fn(out);completed.append(name)
        print('PASS '+name,flush=True)
    save(args.output_dir,'verification',{'verified_studies':completed,'mode':'offline saved-output rebuild; no model calls'})

if __name__=='__main__':main()
