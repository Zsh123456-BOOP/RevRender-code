"""Three trainable parts: see details, find instances, draw instance masks.

No frozen teacher, hooks, takeover slots, second decoder or feedback cascade.
All four arms share the same front end. The renderer is the ablation factor.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('a_coarse', 'b_point', 'c_identity', 'd_competition')

def sample_points(features, coordinates, mode="bilinear"):
    """B,C,H,W and B,P,2 normalized x,y pixel-centre coordinates -> B,C,P."""
    return F.grid_sample(features, coordinates[:, None] * 2 - 1,
                         mode=mode, padding_mode="border", align_corners=False)[:, :, 0]

def index_coordinates(indices, height, width):
    return torch.stack(((indices % width + .5) / width,
                        (indices // width + .5) / height), -1)

class ConvNormAct(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU())

class PixelDecoder(nn.Module):
    """480's learned full/half-resolution image skips, with 32 mask channels."""
    def __init__(self, dimension, channels):
        super().__init__()
        self.skip_full = ConvNormAct(1, 32)
        self.skip_half = ConvNormAct(1, 64)
        self.reduce_half = nn.Conv2d(dimension, 64, 1, bias=False)
        self.refine_half = ConvNormAct(64, 64)
        self.reduce_full = nn.Conv2d(64, 32, 1, bias=False)
        self.refine_full = ConvNormAct(32, 32)
        self.output = nn.Conv2d(32, channels, 1)

    def forward(self, quarter, image):
        half = F.interpolate(quarter, size=(image.shape[-2] // 2, image.shape[-1] // 2),
                             mode="bilinear", align_corners=False)
        half = self.refine_half(self.reduce_half(half) + self.skip_half(F.avg_pool2d(image, 2)))
        full = F.interpolate(half, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.refine_full(self.reduce_full(full) + self.skip_full(image)))

def select_shared_points(coarse, objects, dense, count, training):
    """Shared query-point budget: 75% predicted ambiguity, 25% row coverage."""
    batch, queries, height, width = coarse.shape
    count = min(count, height * width)
    candidates = (torch.rand(batch, count * 3, 2, device=coarse.device) if training else
                  index_coordinates(torch.arange(height * width, device=coarse.device)[None].expand(batch, -1), height, width))
    p = sample_points(coarse, candidates).sigmoid()
    scores = objects.sigmoid()[..., None]
    uncertainty = (4 * p * (1 - p) * scores).amax(1)
    competing = (p * scores).topk(min(2, queries), dim=1).values[:, -1]
    support = sample_points(dense, candidates).sigmoid()[:, 0]
    priority = uncertainty + competing + .25 * support
    top_count = int(count * .75)
    top = priority.topk(top_count, 1).indices
    available = torch.ones_like(priority, dtype=torch.bool).scatter(1, top, False)
    if training:
        coverage = torch.rand_like(priority)
    else:
        # Fixed low-discrepancy rank, without duplicate selected pixel locations.
        k = torch.arange(priority.shape[1], device=priority.device, dtype=priority.dtype) + .5
        coverage = (k * .61803398875).remainder(1)[None].expand(batch, -1)
    rest = coverage.masked_fill(~available, -torch.inf).topk(count - top_count, 1).indices
    chosen = torch.cat((top, rest), 1)
    return candidates.gather(1, chosen[..., None].expand(-1, -1, 2))

def mask_geometry(coarse):
    """Image-normalized soft mask centres and spreads, never ground truth boxes."""
    batch, queries, height, width = coarse.shape
    with torch.no_grad():
        p = coarse.sigmoid().flatten(2)
        grid = index_coordinates(torch.arange(height * width, device=coarse.device)[None], height, width)[0]
        mass = p.sum(-1, keepdim=True).clamp_min(1e-6)
        centre = torch.einsum("bqn,nc->bqc", p, grid) / mass
        second = torch.einsum("bqn,nc->bqc", p, grid.square()) / mass
        spread = (second - centre.square()).clamp_min(0).sqrt()
        minimum = coarse.new_tensor([1 / width, 1 / height])
        spread = torch.maximum(spread, minimum)
    return centre, spread


class SeeDetails(nn.Module):
    """Image encoder, scale fusion and full-resolution detail features."""
    def __init__(self, encoder, decoder, decoder_channels, dimension, channels):
        super().__init__()
        self.encoder = encoder
        self.dense_decoder = decoder
        self.shared_projection = ConvNormAct(decoder_channels, dimension)
        self.pixel_decoder = PixelDecoder(dimension, channels)
        self.dense_head = nn.Conv2d(channels, 1, 1)

    def forward(self, image):
        quarter = self.shared_projection(self.dense_decoder(self.encoder(image)))
        fine = self.pixel_decoder(quarter, image)
        return quarter, fine, self.dense_head(fine)


class FindInstances(nn.Module):
    """Fixed-budget queries predict independent coarse masks and scores."""
    def __init__(self, dimension, channels, queries, layers):
        super().__init__()
        self.query_count = queries
        self.position = nn.Sequential(nn.Linear(2, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.queries = nn.Parameter(torch.randn(queries, dimension)*.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            dimension, 4, dimension*4, dropout=0., batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.mask_embed = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, channels))
        self.objectness = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.seed_fine_projection = nn.Linear(channels, dimension, bias=False)
        nn.init.constant_(self.objectness[-1].bias, -2.)

    def forward(self, quarter, fine, dense):
        batch, _, height, width = quarter.shape
        if fine.shape[-2]*fine.shape[-1] < self.query_count:
            raise ValueError('image smaller than query budget')
        pixels = F.adaptive_avg_pool2d(fine, (height, width))
        indices = torch.arange(height*width, device=quarter.device)[None].expand(batch, -1)
        memory = quarter.flatten(2).transpose(1, 2)+self.position(index_coordinates(indices,height,width))
        # Discrete candidate selection is nondifferentiable; sampled feature values remain trainable.
        seed_score = dense[:, 0].detach()
        maxima = seed_score == F.max_pool2d(seed_score[:, None], 9, 1, 4)[:, 0]
        seeds = (seed_score+maxima.to(seed_score.dtype)*100).flatten(1).topk(self.query_count,1).indices
        coordinates = index_coordinates(seeds, *fine.shape[-2:])
        query = (sample_points(quarter,coordinates).transpose(1,2)+self.position(coordinates)
                 +.1*self.seed_fine_projection(sample_points(fine,coordinates).transpose(1,2))
                 +self.queries[None])
        auxiliary = []
        for index, layer in enumerate(self.layers):
            query = layer(query,memory)
            coarse = torch.einsum('bqc,bchw->bqhw',self.mask_embed(query),pixels)
            if index < len(self.layers)-1:
                auxiliary.append((coarse,self.objectness(query).squeeze(-1)))
        return query,coarse,self.objectness(query).squeeze(-1),auxiliary,coordinates


class DrawMasks(nn.Module):
    """One renderer, with optional identity and local ownership interaction."""
    def __init__(self, arm, dimension, channels, queries, train_points, render_points):
        super().__init__()
        self.arm = arm
        self.query_count = queries
        self.train_points, self.render_points = train_points, render_points
        self.competition_enabled = arm == ARMS[3]
        if arm == ARMS[1]:
            self.point_head = nn.Sequential(nn.Linear(channels+1,64),nn.ReLU(),
                                            nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        if arm in ARMS[2:]:
            self.query_code = nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,16))
            self.local_token = nn.Sequential(nn.Linear(channels+16+4+1,64),nn.GELU(),nn.Linear(64,32))
            self.local_delta = nn.Linear(32,1)
            self.local_gate = nn.Parameter(torch.tensor(-2.))
            # C and D share parameters and ownership supervision; only delivery changes.
            self.background_token = nn.Linear(channels,32)
            self.comp_norm = nn.LayerNorm(32)
            self.comp_attention = nn.MultiheadAttention(32,4,dropout=0.,batch_first=True)
            self.comp_delta = nn.Linear(32,1)
            self.comp_gate = nn.Parameter(torch.tensor(-2.))
            self.owner_mix = nn.Parameter(torch.tensor(-2.))
            nn.init.normal_(self.local_delta.weight,std=.01)
            nn.init.zeros_(self.local_delta.bias)
            nn.init.normal_(self.comp_delta.weight,std=.01)
            nn.init.zeros_(self.comp_delta.bias)

    def local_predictions(self, query, coarse, fine, dense, coords, objects):
        """B,Q,P independent logits, delivered logits, and B,Q+1,P ownership logits."""
        batch, queries = query.shape[:2]
        points = coords.shape[1]
        feature = sample_points(fine, coords).transpose(1, 2)
        coarse_values = sample_points(coarse, coords)
        centre, spread = mask_geometry(coarse)
        relative = ((coords[:, None] - centre[:, :, None]) / spread[:, :, None]).clamp(-4, 4)
        geometry = torch.cat((relative, spread.log().clamp(-8, 0)[:, :, None].expand(-1, -1, points, -1)), -1)
        code = self.query_code(query)[:, :, None].expand(-1, -1, points, -1)
        inputs = torch.cat((feature[:, None].expand(-1, queries, -1, -1), code, geometry, coarse_values[..., None]), -1)
        tokens = self.local_token(inputs)
        independent = coarse_values + self.local_gate.sigmoid() * self.local_delta(tokens).squeeze(-1)
        background = self.background_token(feature)[:, None]
        all_tokens = torch.cat((tokens, background), 1).permute(0, 2, 1, 3).reshape(batch * points, queries + 1, 32)
        normalized = self.comp_norm(all_tokens)
        # Soft evidence weighting retains every query while emphasizing nearby candidates.
        prior = independent.sigmoid() * objects.sigmoid()[..., None]
        bg_prior = (-sample_points(dense, coords)).sigmoid()
        evidence = torch.cat((prior, bg_prior), 1).permute(0, 2, 1).reshape(batch * points, queries + 1, 1)
        context, _ = self.comp_attention(normalized, normalized, normalized * evidence, need_weights=False)
        delta = self.comp_delta(all_tokens + context).reshape(batch, points, queries + 1).transpose(1, 2)
        base = torch.cat((independent + F.logsigmoid(objects)[..., None], -sample_points(dense, coords)), 1)
        ownership_logits = base + self.comp_gate.sigmoid() * delta
        owned = ownership_logits.softmax(1)[:, :queries]
        # Small initial spatial modification, with a nonzero gradient on every stage.
        mixture = (1 - self.owner_mix.sigmoid()) * independent.sigmoid() + self.owner_mix.sigmoid() * owned
        competitive = torch.logit(mixture.clamp(1e-6, 1 - 1e-6))
        delivered = competitive if self.competition_enabled else independent
        return independent, delivered, ownership_logits

    def at_points(self, query, coarse, fine, dense, coords, objects):
        if self.arm == ARMS[1]:
            features = sample_points(fine,coords).transpose(1,2)
            values = sample_points(coarse,coords)
            shared = features[:,None].expand(-1,self.query_count,-1,-1)
            return self.point_head(torch.cat((shared,values[...,None]),-1)).squeeze(-1),None
        _,delivered,ownership = self.local_predictions(query,coarse,fine,dense,coords,objects)
        return delivered,ownership

    def forward(self, query, coarse, fine, dense, objects, size):
        cache = {}
        if self.arm == ARMS[0]:
            return F.interpolate(coarse,size=size,mode='bilinear',align_corners=False),cache
        if self.training:
            with torch.no_grad():
                coords = select_shared_points(coarse,objects,dense,self.train_points,True)
            delivered,ownership = self.at_points(query,coarse,fine,dense,coords,objects)
            cache['points'] = coords,delivered
            if ownership is not None:cache['ownership'] = coords,ownership
            return F.interpolate(coarse,size=size,mode='bilinear',align_corners=False),cache
        rendered = coarse
        for divisor in (2,1):
            level_size = (size[0]//divisor,size[1]//divisor)
            rendered = F.interpolate(rendered,size=level_size,mode='bilinear',align_corners=False)
            source = F.interpolate(coarse,size=level_size,mode='bilinear',align_corners=False)
            coords = select_shared_points(source,objects,dense,self.render_points,False)
            # All subdivision levels use the original coarse representation as point features.
            delivered,_ = self.at_points(query,coarse,fine,dense,coords,objects)
            x = (coords[...,0]*level_size[1]).long().clamp_max(level_size[1]-1)
            y = (coords[...,1]*level_size[0]).long().clamp_max(level_size[0]-1)
            positions = (y*level_size[1]+x)[:,None].expand(-1,self.query_count,-1)
            rendered = rendered.flatten(2).scatter(2,positions,delivered).reshape(
                len(query),self.query_count,*level_size)
        return rendered,cache


class ThreePartModel(nn.Module):
    def __init__(self,encoder,decoder,decoder_channels,arm,dimension=160,channels=32,
                 queries=32,layers=4,train_points=256,render_points=1024):
        super().__init__()
        if arm not in ARMS or min(dimension,channels,queries,layers,train_points,render_points)<1:
            raise ValueError('invalid three-part configuration')
        self.arm,self.query_count = arm,queries
        self.see = SeeDetails(encoder,decoder,decoder_channels,dimension,channels)
        self.find = FindInstances(dimension,channels,queries,layers)
        self.draw = DrawMasks(arm,dimension,channels,queries,train_points,render_points)
        self.cache = {}

    def forward(self,image):
        quarter,fine,dense = self.see(image)
        query,coarse,objects,auxiliary,seeds = self.find(quarter,fine,dense)
        rendered,cache = self.draw(query,coarse,fine,dense,objects,image.shape[-2:])
        self.cache = {'coarse':coarse,'dense':dense,'auxiliary':auxiliary,'seed_coordinates':seeds,**cache}
        return rendered,objects,None


def build_model(baseline,args):
    segmenter = baseline._smp_model('segformer_b0',args)
    return ThreePartModel(segmenter.encoder,segmenter.decoder,
                          int(segmenter.segmentation_head[0].in_channels),args.arm)


def losses(model,masks,scores,targets,probe,fullres):
    """Same coarse/dense/matching supervision; point and ownership terms are explicit ablations."""
    batch,queries,height,width = masks.shape
    aligned,labels = torch.zeros_like(masks),torch.zeros_like(scores)
    union = torch.zeros(batch,1,height,width,device=masks.device)
    for index,truth in enumerate(targets):
        if len(truth)>queries:raise ValueError('targets exceed query budget')
        rows,columns = probe.hungarian_match(masks[index],scores[index],truth)
        for q,g in zip(rows,columns):aligned[index,q],labels[index,q] = truth[g],1
        for target in truth:
            if (union[index,0].bool()&target.bool()).any():raise ValueError('overlapping region targets')
            union[index,0] = torch.maximum(union[index,0],target)
    active = labels.bool()
    def mask_loss(logits):
        if not active.any():return logits.sum()*0
        values = F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False)
        return fullres.matched_mask_loss(values[active],aligned[active],'dice',.0)
    values = {'mask':mask_loss(model.cache['coarse']),
              'objectness':F.binary_cross_entropy_with_logits(scores,labels),
              'dense':fullres.matched_mask_loss(model.cache['dense'],union,'dice',.0)}
    auxiliary = masks.sum()*0
    for logits,objects in model.cache['auxiliary']:
        auxiliary = auxiliary+mask_loss(logits)+F.binary_cross_entropy_with_logits(objects,labels)
    values['auxiliary'] = auxiliary/max(1,len(model.cache['auxiliary']))
    if 'points' in model.cache:
        coords,logits = model.cache['points']
        gold = sample_points(aligned,coords)
        values['point'] = F.binary_cross_entropy_with_logits(logits[active],gold[active]) if active.any() else logits.sum()*0
    if 'ownership' in model.cache:
        coords,logits = model.cache['ownership']
        ids = torch.full((batch,height,width),queries,device=masks.device,dtype=torch.long)
        for index in range(batch):
            for q in active[index].nonzero().flatten():ids[index][aligned[index,q].bool()] = q
        point_ids = sample_points(ids[:,None].float(),coords,'nearest')[:,0].long()
        ce = F.cross_entropy(logits,point_ids,reduction='none')
        positive = point_ids!=queries
        groups = [ce[positive].mean()] if positive.any() else []
        if (~positive).any():groups.append(ce[~positive].mean())
        values['ownership'] = torch.stack(groups).mean()
    total = sum(value*{'auxiliary':.25,'ownership':.2}.get(key,1.) for key,value in values.items())
    return total,{key:float(value.detach()) for key,value in values.items()}


def configuration(arm):
    return {'parts':['see','find','draw'],'arm':arm,'encoder':'MiT-B0','queries':32,'layers':4,
            'query_dimension':160,'mask_channels':32,'highres_seeds':True,
            'renderer':arm,'train_points':0 if arm==ARMS[0] else 256,
            'render_points_per_level':0 if arm==ARMS[0] else 1024,
            'shared_point_sampling':True,'ownership_auxiliary':arm in ARMS[2:],
            'all_parts_jointly_trained':True,'frozen_teacher':False,'feedback_cascade':False,
            'point_scope':'PointRend-style sampling/MLP/subdivision adapted to row queries; not native Mask R-CNN+PointRend',
            'scores':'original finder objectness; no additional scoring or feedback head'}
