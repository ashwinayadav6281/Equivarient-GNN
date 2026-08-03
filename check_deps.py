"""Check available libraries and sample structure properties."""
try:
    import matminer
    print('matminer:', matminer.__version__)
except:
    print('matminer: NOT installed')

try:
    import sklearn
    print('sklearn:', sklearn.__version__)
except:
    print('sklearn: NOT installed')

from pymatgen.core import Structure, Element
import numpy as np

s = Structure.from_file('cgcnn_dataset_full/mp-aaaaaaab.cif')
print(f'\nSample structure: {s.formula}')
print(f'  Sites: {len(s)}')
print(f'  Volume: {s.volume:.2f} A^3')
print(f'  Density: {s.density:.2f} g/cm^3')
print(f'  Lattice: a={s.lattice.a:.3f} b={s.lattice.b:.3f} c={s.lattice.c:.3f}')
print(f'  Angles: alpha={s.lattice.alpha:.1f} beta={s.lattice.beta:.1f} gamma={s.lattice.gamma:.1f}')
print(f'  Species: {[str(sp) for sp in set(s.species)]}')

# Check what elemental properties are available
el = Element('Fe')
print(f'\nSample Element properties for Fe:')
print(f'  atomic_mass: {el.atomic_mass}')
print(f'  atomic_radius: {el.atomic_radius}')
print(f'  X (electronegativity): {el.X}')
print(f'  group: {el.group}')
print(f'  row: {el.row}')

# CGCNN shear metrics
import csv, statistics
rows = [r for r in csv.reader(open('predictions_shear.csv')) if r]
vals = [(float(r[1]), float(r[2])) for r in rows]
errs = [abs(a-p) for a,p in vals]
sq_errs = [(a-p)**2 for a,p in vals]
mean_a = statistics.mean([a for a,_ in vals])
ss_res = sum((a-p)**2 for a,p in vals)
ss_tot = sum((a-mean_a)**2 for a,_ in vals)
print(f'\nCGCNN Shear: MAE={statistics.mean(errs):.3f}, RMSE={statistics.mean(sq_errs)**0.5:.3f}, R2={1-ss_res/ss_tot:.4f}')
