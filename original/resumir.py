from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'mediciones.csv')
g=raw.groupby(['n','k','bits','leaf'],as_index=False).agg(
    query_mean_s=('encrypted_query_mean_s','mean'),query_std_s=('encrypted_query_mean_s','std'),
    wall_mean_s=('wall_s','mean'),wall_std_s=('wall_s','std'),
    outsource_mean_s=('outsourcing_s','mean'),correct=('correct','sum'),queries=('queries','sum'),repetitions=('repeat','count'))
g.to_csv(ROOT/'resumen.csv',index=False)
for filename,sub,x,title in [
    ('grafica_n.png',g[(g.k==3)&(g.bits==128)],'n','Costo de consulta cifrada vs. registros'),
    ('grafica_k.png',g[(g.n==48)&(g.bits==128)],'k','Costo de consulta cifrada vs. vecinos k'),
    ('grafica_claves.png',g[(g.n==48)&(g.k==3)],'bits','Costo de consulta cifrada vs. bits de clave')]:
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    sub=sub.sort_values(x)
    ax.errorbar(sub[x],sub.query_mean_s,yerr=sub.query_std_s,fmt='o-',capsize=5)
    ax.set(xlabel={'n':'Registros (n)','k':'Vecinos (k)','bits':'Bits de clave Paillier'}[x],ylabel='Segundos por consulta cifrada (media ± DE)',title=title)
    ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(ROOT/filename,dpi=150);plt.close(fig)
print(g.to_string(index=False))
