
import numpy as np
MESH="reports/meshes_acdc/meshes/patient001_coarse5"
with open(f"{MESH}.pts") as f:
    n=int(f.readline()); nd=np.array([list(map(float,f.readline().split())) for _ in range(n)])
with open(f"{MESH}.elem") as f:
    ne=int(f.readline()); el=np.array([list(map(int,f.readline().split()[1:5])) for _ in range(ne)],dtype=np.int64)
v=nd[el]; J=np.linalg.det(v[:,1:]-v[:,0:1])/6.0
edges=np.concatenate([v[:,i]-v[:,j] for i in range(4) for j in range(i+1,4)],axis=0).reshape(6,-1,3)
h=np.linalg.norm(edges,axis=2).mean(0)
qual=np.abs(J)/(h**3/6.0+1e-12)
print("=== QUALITE MAILLAGE AU REPOS ===")
print(f"tets={ne}  volume signe min={J.min():.4f}")
print(f"qualite: min={qual.min():.4f} p5={np.percentile(qual,5):.4f} median={np.median(qual):.4f}")
print(f"tets <0.1: {int((qual<0.1).sum())}   <0.2: {int((qual<0.2).sum())}")
a_kPa=0.496; b=7.209; mu=a_kPa*1000*b
print("\n=== RIGIDITE vs PRESSION ===")
print(f"mu~a*b={mu:.0f} Pa = {mu/1000:.2f} kPa   pression=15 kPa   ratio P/mu={15000/mu:.1f}")
