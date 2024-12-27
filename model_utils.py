import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from train_utils import GaussianKLDivLoss, DiscreteGaussianNLLLoss, UniformSampler
import numpy as np
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

def polygon_model_defaults():
    return dict(
        n_coords = 2, 
        max_n_polygons = 8, 
        max_n_vertices = 32,
        model_channels = 512,
        out_channels = 2,
        num_layers = 8,
    )

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation
    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = th.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = th.matmul(scores, v)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.self_norm_1 = nn.LayerNorm([d_model])
        self.self_norm_2 = nn.LayerNorm([d_model])
        self.self_norm_3 = nn.LayerNorm([d_model])
        self.self_attn = nn.MultiheadAttention(d_model, heads)
        self.corrupt_attn = nn.MultiheadAttention(d_model, heads)
        self.cond_attn = nn.MultiheadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, corrupts, conditions):
        z = self.self_norm_1(x)
        x = x + self.dropout(self.self_attn(z, z, z)[0])

        z = self.self_norm_2(x)
        x = x + self.dropout(self.corrupt_attn(z,corrupts,corrupts)[0]) + self.dropout(self.cond_attn(z,conditions,conditions)[0])
        
        z = self.self_norm_3(x)
        x = x + self.dropout(self.ff(z))
        return x

class PolygonFeatureEmbedding(nn.Module):
    def __init__(self, n_coords, max_n_polygons, max_n_vertices, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.time_embed = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        #input_channels = n_coords
        input_channels = n_coords + 2
        self.input_emb = nn.Linear(input_channels, out_channels)
        corrupt_channels = n_coords + 2*8 + max_n_vertices + max_n_polygons
        self.corrupt_emb = nn.Linear(corrupt_channels, out_channels)
        condition_channels = max_n_polygons + 1
        self.condition_emb = nn.Linear(condition_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def AU(self, points, edges):
        p1 = points
        p1 = p1.view([p1.shape[0], p1.shape[1], 2, -1])
        p5 = points[th.arange(points.shape[0])[:, None], edges[:,:,1].long()]
        p5 = p5.view([p5.shape[0], p5.shape[1], 2, -1])
        p3 = (p1 + p5) / 2
        p2 = (p1 + p3) / 2
        p4 = (p3 + p5) / 2
        p1_5 = (p1 + p2) / 2
        p2_5 = (p2 + p3) / 2
        p3_5 = (p3 + p4) / 2
        p4_5 = (p4 + p5) / 2
        points_new = th.cat((p1.view_as(points), p1_5.view_as(points), p2.view_as(points),
            p2_5.view_as(points), p3.view_as(points), p3_5.view_as(points), p4.view_as(points), p4_5.view_as(points), p5.view_as(points)), 2)
        return points_new.detach()
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, shift, t, corrupt, edges, vertice_inds, polygon_inds, graphs, areas):
        shift_emb = self.input_emb(shift)
        corrupt_x = self.AU(corrupt, edges)
        vertice_inds = self.sigmoid(vertice_inds)
        polygon_inds = self.sigmoid(polygon_inds)
        graphs = self.sigmoid(graphs)

        corrupt_total = None
        for local_cond in [corrupt_x, vertice_inds, polygon_inds]:
            if corrupt_total is None:
                corrupt_total = local_cond
            else:
                corrupt_total = th.cat((corrupt_total, local_cond), 2)
        cor_emb = self.corrupt_emb(corrupt_total.float())

        cond = None
        for local_cond in [graphs, areas]:
            if cond is None:
                cond = local_cond
            else:
                cond = th.cat((cond, local_cond), 2)
        cond_emb = self.condition_emb(cond.float())

        time_emb = self.time_embed(self.timestep_embedding(t, self.out_channels))
        time_emb = time_emb.unsqueeze(1).repeat((1, shift_emb.shape[1], 1))

        return shift_emb, cor_emb, cond_emb, time_emb

class PolygonTransformerModel(nn.Module):
    """
    The full Transformer model with timestep embedding.
    """

    def __init__(
        self,
        n_coords, 
        max_n_polygons, 
        max_n_vertices,
        model_channels,
        out_channels,
        num_layers,
        
    ):
        super().__init__()
        self.n_coords = n_coords
        self.max_n_polygons = max_n_polygons
        self.max_n_vertices = max_n_vertices
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.feature_embedding = PolygonFeatureEmbedding(n_coords, max_n_polygons, max_n_vertices, model_channels)
        
        self.activation = nn.ReLU()
        self.transformer_layers = nn.ModuleList([EncoderLayer(d_model=model_channels, heads=4, dropout=0.1, activation=nn.ReLU()) for x in range(self.num_layers)])

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.out_channels + 2)

    def forward(self, x, timesteps, **kwargs):
        shift_emb, cor_emb, cond_emb, time_emb = self.feature_embedding(x, timesteps, kwargs["corrupt"], kwargs["edges"], kwargs["vertice_inds"], kwargs["polygon_inds"], kwargs["graph"], kwargs["area"])
        out = shift_emb

        for layer in self.transformer_layers:
            out = layer(shift_emb + time_emb, cor_emb, cond_emb)

        out = self.output_linear1(out)
        out = self.activation(out)
        out = self.output_linear2(out)

        return out

class PolygonDiffusionModel(nn.Module):
    def __init__(self, model_defaults, guided=False):
        super().__init__()
        self.guided = guided
        if not guided:
            self.base_model = PolygonTransformerModel(**model_defaults)
        else:
            self.base_model_1 = PolygonTransformerModel(**model_defaults)
            self.base_model_2 = PolygonTransformerModel(**model_defaults)
            self.gama = 0.5
            self.u = 10

        self.mse_loss = nn.MSELoss()

        self.noise_scheduler = DDPMScheduler()
    
    def forward(self, **model_kwargs):
        if not self.guided:
            x0 = model_kwargs["shift"]
        else:
            #x0 = th.cat([model_kwargs["shift"], model_kwargs["corrupt"]], dim=2)
            x0 = th.cat([model_kwargs["shift"] + model_kwargs["corrupt"], model_kwargs["corrupt"]], dim=2)
            condition = th.cat([th.zeros_like(model_kwargs["shift"]), model_kwargs["corrupt"]], dim=2)

        t = th.randint(0, len(self.noise_scheduler.timesteps), (x0.shape[0],), device=x0.device, dtype=th.long)
        t = t.to(x0.device)

        noise = th.randn_like(x0)

        padding_mask = model_kwargs['padding_mask']
        
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        if not self.guided:
            model_output = self.base_model(x_t, t, **model_kwargs)
        else:
            model_output = self.guided_forward(x_t, t, condition, **model_kwargs)

        model_output = model_output * padding_mask.unsqueeze(2).repeat([1, 1, model_output.shape[2]])
        noise = noise * padding_mask.unsqueeze(2).repeat([1, 1, noise.shape[2]])
        loss = self.mse_loss(noise, model_output)
        
        return {"loss":loss}

    def generate(self, shape, model_kwargs):
        with th.no_grad():
            if not self.guided:
                sample = th.randn(shape).to("cuda")
            else:
                sample = th.randn((shape[0], shape[1], shape[2]*2)).to("cuda")
            for t in tqdm(self.noise_scheduler.timesteps):
                time = sample.new_full((sample.shape[0],), t, dtype=th.long).to("cuda")

                if not self.guided:
                    xt = self.base_model(sample, time, **model_kwargs)
                    sample = self.noise_scheduler.step(xt, t, sample).prev_sample
                else:
                    condition = th.cat([th.zeros(shape).to("cuda"), model_kwargs["corrupt"]], dim=2)
                    mask = th.cat([th.zeros(shape).to("cuda"), th.ones(shape).to("cuda")], dim=2)
                    sample = self.guided_sampling_step(sample, time, condition, mask, model_kwargs)

            if self.guided:
                sample = sample[:,:,:2]

        return sample
    
    def guided_forward(self, x_t, t, condition, **model_kwargs):
        t1 = self.base_model_1(x_t, t, **model_kwargs)
        t2 = self.base_model_2(condition, t, **model_kwargs)
        model_output = self.gama * t1 + (1 - self.gama) * t2
        return model_output
    
    def guided_sampling_step(self, x_t, t, condition, mask, model_kwargs):
        time = t[0].item()
        for i in range(self.u):
            noise = th.randn_like(x_t) if time > 0 else th.zeros_like(x_t)

            gt_t = self.noise_scheduler.add_noise(condition, noise, t).type(noise.dtype)
            x_t = gt_t*mask + x_t*(1-mask)

            eps = self.guided_forward(x_t.float(), t.float(), condition, **model_kwargs)

            x_t1 = self.noise_scheduler.step(eps, time, x_t).prev_sample
            if i != self.u - 1 and time > 0:
                x_t = self.add_noise_step(x_t1, noise, time)

        return x_t
    
    def add_noise_step(self, x_t1, noise, t):
        beta = self.noise_scheduler.betas[t]
        added = th.sqrt(1 - beta) * x_t1 + th.sqrt(beta) * noise
        return added