from helper import *
from model.message_passing import MessagePassing
from model.mixer import *

from torch_scatter import scatter_max, scatter_add

class HoGRNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 				= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None
		self.softmax 	= torch.nn.Softmax(dim=-1)

		self.loop_rel 	= get_param((1, in_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
		if self.p.rel_norm: self.rel_norm = nn.LayerNorm(out_channels, elementwise_affine=False) 

		# For relation reasoning
		if self.p.rel_reason:
			if self.p.reason_type == 'mlp':
				self.w_rel 	= get_param((in_channels, out_channels))
			elif self.p.reason_type == 'mixdrop':
				self.MixerBlock = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)
			elif self.p.reason_type == 'mixdrop2':
				self.MixerBlock_1 = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)
				self.MixerBlock_2 = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)
			elif self.p.reason_type == 'mixdrop3':
				self.MixerBlock_1 = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)
				self.MixerBlock_2 = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)
				self.MixerBlock_3 = MixerDrop(self.num_rels*2+1, out_channels, self.p.relmix_dim, self.p.chamix_dim, self.p.rel_mask, self.p.chan_drop)

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed 		= torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges 		= edge_index.size(1) // 2
		num_ent   		= x.size(0)
		self.num_ent	= num_ent

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)

		if self.p.rel_reason and self.p.pre_reason:
			rel_embed = self.rel_reason(rel_embed)
			if self.p.rel_norm:
				rel_embed = self.rel_norm(rel_embed)

		if self.p.act_type == 'tanh':
			in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm)
			loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None)
			out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm)
			out			= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
		elif self.p.act_type == 'softmax':
			in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=None)
			out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=None)
			out			= self.drop(in_res)*(1/2) + self.drop(out_res)*(1/2) 

		if self.p.bias: out = out + self.bias

		if self.p.rel_reason and not self.p.pre_reason:
			rel_embed = self.rel_reason(rel_embed)
			if self.p.rel_norm:
				rel_embed = self.rel_norm(rel_embed)
		
		return self.bn(out), rel_embed[:-1]

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def rel_reason(self, rel_embed):
		if self.p.reason_type == 'raw':
			return rel_embed
		elif self.p.reason_type == 'mlp':
			return torch.matmul(rel_embed, self.w_rel)
		elif self.p.reason_type == 'mixdrop':
			return self.MixerBlock(rel_embed, training=self.training)
		elif self.p.reason_type == 'mixdrop2':
			rel_embed = self.MixerBlock_1(rel_embed, training=self.training)
			return self.MixerBlock_2(rel_embed, training=self.training)
		elif self.p.reason_type == 'mixdrop3':
			rel_embed = self.MixerBlock_1(rel_embed, training=self.training)
			rel_embed = self.MixerBlock_2(rel_embed, training=self.training)
			return self.MixerBlock_3(rel_embed, training=self.training)

	def message(self, x_i, x_j, edge_type, rel_embed, edge_norm):
		rel_emb		= torch.index_select(rel_embed, 0, edge_type)
		xi_rel		= self.rel_transform(x_i, rel_emb)
		xj_rel  	= self.rel_transform(x_j, rel_emb)
		if self.p.act_type == 'tanh':
			aggr_coef	= self.act((xi_rel * xj_rel).sum(dim=-1, keepdim=True))
		elif self.p.act_type == 'softmax':
			alpha 		= (xi_rel * xj_rel).sum(dim=-1, keepdim=True) 
			aggr_coef	= self.softmax_sp(alpha, self.in_index[0], self.num_ent) 
		out			= aggr_coef * xj_rel
		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight = torch.ones_like(row).float()
		deg			= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]		

		return norm

	def softmax_sp(self, src, index, num_nodes):
		out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index] 
		out = out.exp()
		out = out / (
			scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

		return out

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
