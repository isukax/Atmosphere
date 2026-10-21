"""Compare compact_dump and compact_reference_dump outputs (requires NumPy)."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference',type=Path)
parser.add_argument('compact',type=Path)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
expected={'scene','transmittance','ms','sky','sky_mie','ap','ap_mie','ap_ms','irradiance','depth'}
results=[]
ref_moments=np.fromfile(args.reference/'phase_moments.f32',dtype='<f4')
got_moments=np.fromfile(args.compact/'phase_moments.f32',dtype='<f4')
if ref_moments.size!=9 or not np.array_equal(ref_moments,got_moments):
    raise SystemExit('Embedded phase moments differ from the original float values')
def compare(ref,got,label,fp32):
    finite=bool(np.isfinite(ref).all() and np.isfinite(got).all())
    scale=max(float(np.sqrt(np.mean(ref*ref))),1e-30)
    peak=max(float(np.max(np.abs(ref))),1e-30)
    rmse=float(np.sqrt(np.mean((ref-got)**2)))/scale
    maximum=float(np.max(np.abs(ref-got)))/peak
    passed=finite and rmse<=(1e-4 if fp32 else 1e-3) and maximum<=(1e-3 if fp32 else 1e-2)
    results.append({'case':label,'nrmse':rmse,'normalized_max':maximum,'finite':finite,'pass':passed})
folders=sorted(p for p in args.reference.glob('fp*/*') if p.is_dir())
if not folders:raise SystemExit('No reference cases found')
if {p.relative_to(args.reference) for p in folders}!={p.relative_to(args.compact) for p in args.compact.glob('fp*/*') if p.is_dir()}:
    raise SystemExit('Case sets differ')
for folder in folders:
    relative=folder.relative_to(args.reference)
    target=args.compact/relative
    if {p.stem for p in folder.glob('*.f32')}!=expected or {p.stem for p in target.glob('*.f32')}!=expected:
        raise SystemExit(f'Artifact set differs: {relative}')
    depth=np.fromfile(folder/'depth.f32',dtype='<f4')
    for stem in sorted(expected):
        ref=np.fromfile(folder/(stem+'.f32'),dtype='<f4').astype(np.float64)
        got=np.fromfile(target/(stem+'.f32'),dtype='<f4').astype(np.float64)
        if ref.shape!=got.shape or not ref.size:raise SystemExit(f'Shape mismatch: {relative}/{stem}')
        if stem=='depth':
            if not np.array_equal(ref,got):raise SystemExit(f'Depth mismatch: {relative}')
            continue
        ref=ref.reshape(-1,4);got=got.reshape(-1,4)
        if stem=='scene':
            for name,mask in [('sky',depth==0),('ground',depth>0)]:
                if np.any(mask):compare(ref[mask,:3],got[mask,:3],str(relative/stem)+'/'+name,relative.parts[0]=='fp32')
        else:
            compare(ref[:,:3],got[:,:3],str(relative/stem),relative.parts[0]=='fp32')
            if stem=='ap':compare(ref[:,3],got[:,3],str(relative/stem)+'/mean_T',relative.parts[0]=='fp32')
        if not np.isfinite(got).all():raise SystemExit(f'Nonfinite channel: {relative}/{stem}')
report={'cases':len(folders),'checks':len(results),'passed':sum(r['pass'] for r in results),'results':results}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
failed=[r for r in results if not r['pass']]
print(f"{report['passed']}/{len(results)} checks passed in {len(folders)} cases")
for r in failed:print(r)
raise SystemExit(bool(failed))
