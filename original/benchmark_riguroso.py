"""Benchmark reproducible: no altera los módulos de la baseline.
Tiempo de consulta cifrada y tiempos de preparación medidos por Main instrumentado.
"""
import argparse, csv, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'instrumentada'
CASES = [(12,3,128,4),(24,3,128,4),(48,3,128,4),(96,3,128,4),
         (48,1,128,4),(48,5,128,4),(48,3,256,4),(48,3,512,4)]

def run_case(n,k,bits,leaf,rep,timeout):
    name=f'n{n}_k{k}_b{bits}_f{leaf}_r{rep}'
    output=ROOT/'runs'/f'{name}.csv'
    output.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,'Main.py','--source','csv','--csv-path','test_2d.csv',
         '--db-rows',str(n),'--queries','2','--k',str(k),'--key-bits',str(bits),
         '--leaf-capacity',str(leaf),'--output-csv',str(output)]
    start=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=TARGET,capture_output=True,text=True,timeout=timeout)
        elapsed=time.perf_counter()-start
        (ROOT/'runs'/f'{name}.log').write_text(p.stdout+'\n'+p.stderr)
        path=output.with_suffix('.timings.json')
        timings=json.loads(path.read_text()) if path.exists() else {}
        q=timings.get('queries',[])
        matches=re.findall(r'Consultas correctas:\s*(\d+)/(\d+)',p.stdout)
        correct,total=tuple(map(int,matches[-1])) if matches else (0,0)
        avg=lambda key: statistics.mean(x[key] for x in q) if q else None
        result=dict(n=n,k=k,bits=bits,leaf=leaf,repeat=rep,wall_s=elapsed,
                    key_generation_s=timings.get('stages',{}).get('key_generation_s'),
                    outsourcing_s=timings.get('stages',{}).get('outsourcing_s'),
                    encrypted_query_mean_s=avg('encrypted_query_s'),
                    reference_mean_s=avg('reference_s'),
                    mean_distance_evaluations=avg('distance_evaluations'),
                    correct=correct,queries=total,
                    status='OK' if p.returncode==0 and correct==total and total==2 and all(x['correct'] for x in q) else 'FAIL')
    except subprocess.TimeoutExpired:
        result=dict(n=n,k=k,bits=bits,leaf=leaf,repeat=rep,wall_s=timeout,
                    status='TIMEOUT',correct=0,queries=0)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--timeout',type=int,default=90)
    a=ap.parse_args()
    rows=[]
    for n,k,bits,leaf in CASES:
        for repeat in range(1,a.repeats+1):
            row=run_case(n,k,bits,leaf,repeat,a.timeout)
            rows.append(row)
            print(row,flush=True)
            fields=['n','k','bits','leaf','repeat','wall_s','key_generation_s','outsourcing_s',
                    'encrypted_query_mean_s','reference_mean_s','mean_distance_evaluations','correct','queries','status']
            with (ROOT/'mediciones.csv').open('w',newline='') as file:
                w=csv.DictWriter(file,fieldnames=fields);w.writeheader();w.writerows(rows)
    return 0 if all(r['status']=='OK' for r in rows) else 1
if __name__=='__main__': sys.exit(main())
