"""Portable entry points for the RevRender research implementation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,runpy,sys
ROOT=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('public_'+name[:3],ROOT/'scripts'/name)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def forward(name,args):
    sys.argv=[str(ROOT/'scripts'/name),*args];runpy.run_path(sys.argv[0],run_name='__main__')

def read_image(path):
    import numpy as np
    if path.suffix.lower()=='.npy':image=np.load(path,allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(path) as im:
            if im.mode!='L':raise ValueError('Supply a grayscale L image; automatic color conversion/resizing is not applied')
            image=np.asarray(im)
    if image.dtype!=np.uint8 or image.shape!=(96,768):raise ValueError('Expected uint8 grayscale shape (96, 768); prepare the line explicitly')
    return image

def predict(args):
    import numpy as np
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('Create-only output already exists')
    image=read_image(args.image);frozen=json.loads(args.freeze.read_text())
    if frozen.get('model')!='point_context' or frozen.get('test_touched') is not False:raise ValueError('Expected a validation-only RevRender freeze record')
    if digest(args.checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Checkpoint differs from the freeze record')
    if digest(args.backbone)!='16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6':raise ValueError('Unexpected pretrained ResNet-50 file')
    os.environ['HME_TUNE']='control'
    import torch
    from types import SimpleNamespace
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    mode=checkpoint['profile']['configuration'].get('tuning880',{}).get('mode','control')
    if mode!='control':raise ValueError('This entry reproduces the paper control model; use original architecture entry for ablations')
    ops=load('668_revision_deterministic_ops.py');ops.configure();torch.autograd.set_multithreading_enabled(False)
    torch.manual_seed(42)
    architecture=load('880_revision_tuning.py');ops.install(architecture.CORE)
    model=architecture.build_model(load('422_revision_paper_baselines.py'),SimpleNamespace(arm='d_region_roi_point',segformer_b0_weights=args.backbone)).to(args.device).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    tensor=torch.from_numpy(image[None,None].astype(np.float32)/127.5-1).to(args.device)
    with torch.no_grad():masks,objects,_=model(tensor)
    masks=masks[0].sigmoid().cpu().numpy()>=float(frozen['mask_threshold']);scores=objects[0].sigmoid().cpu().numpy().astype(np.float64)
    keep=load('702_revision_suppression_protocol.py').suppressed(masks,scores,float(frozen['nms']))
    keep=[i for i in keep if masks[i].any() and scores[i]>=float(frozen['score_threshold'])]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:np.savez_compressed(f,masks=masks[keep],scores=scores[keep],query_indices=np.array(keep,dtype=np.int64))
    with args.output.with_suffix('.json').open('x') as f:json.dump({'image_sha256':digest(args.image),'checkpoint_sha256':digest(args.checkpoint),'freeze_sha256':digest(args.freeze),'source_sha256':digest(Path(__file__)),'device':args.device,'torch':torch.__version__,'mask_threshold':frozen['mask_threshold'],'score_threshold':frozen['score_threshold'],'nms':frozen['nms'],'instances':len(keep)},f,indent=2)
    print(args.output)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train',help='Run the paper training recipe; explicit data and weight paths required')
    for n in ['bundle','corrected-bundle','backbone','output-dir']:train.add_argument('--'+n,type=Path,required=True)
    for n in ['bundle-sha','corrected-sha']:train.add_argument('--'+n,required=True)
    train.add_argument('--stage',choices=['smoke','learning','formal'],required=True)
    train.add_argument('--replay-of',type=Path);train.add_argument('--gate-root',type=Path)
    for command,script in [('freeze','867_revision_mask_threshold_freeze.py'),('evaluate','883_revision_mask_ap.py')]:
        s=sub.add_parser(command,add_help=False,help='Use the original '+command+' protocol (--help for arguments)');s.add_argument('arguments',nargs=argparse.REMAINDER);s.set_defaults(script=script)
    pred=sub.add_parser('predict',help='Infer one uint8 grayscale 96x768 answer line')
    for n in ['image','checkpoint','freeze','backbone','output']:pred.add_argument('--'+n,type=Path,required=True)
    pred.add_argument('--device',default='cuda',choices=['cuda','cpu']);pred.set_defaults(function=predict)
    words=list(sys.argv[1:] if argv is None else argv)
    # Forward switches unchanged rather than letting the wrapper consume them.
    if words and words[0] in ['freeze','evaluate']:
        return forward({'freeze':'867_revision_mask_threshold_freeze.py','evaluate':'883_revision_mask_ap.py'}[words[0]],words[1:])
    args=p.parse_args(words)
    if args.command=='train':
        extra=[]
        for field in ['replay_of','gate_root']:
            if getattr(args,field):extra += ['--'+field.replace('_','-'),str(getattr(args,field))]
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        forward('881_revision_tuning_train.py',['--tune','control','--bundle',str(args.bundle),'--bundle-sha',args.bundle_sha,'--corrected-bundle',str(args.corrected_bundle),'--corrected-sha',args.corrected_sha,'--arm','d_region_roi_point','--seed','42','--segformer-b0-weights',str(args.backbone),'--clip-groups','find.mask_iou','--speed-v1','--early-stop-patience','6','--stage',args.stage,'--output-dir',str(args.output_dir),*extra])
    else:args.function(args)

if __name__=='__main__':main()
