import torch
import torch.nn as nn

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0] # 任何数与0做位异或操作结果都是这个数本身!
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]    # python中^是按位异或运算符，运算规则是：将两个数的二进制按位对比，如果相同（都为1或者都为0），则运算结果的对应位上是0，否则为1

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result # 1 << log2_hashmap_size代表将1的二进制左移log2_hashmap_size位，然后在其右边全补上0，将结果换算为十进制等于2的log2_hashmap_size次方
                                                                                     # 2的N次方-1在二进制中是一个十分特殊的数，它右N个1从最右向左依次排列组成
                                                                                     
                                                                                     # python中&是位与运算符，运算规则是：将两个数的二进制按位对比，都为1，则结果的对应位上是1，否则为0
                                                                                     # 对于一个数x（十进制），x mod 2的N次方 等价于取x的二进制的后N位数（得到的是二进制，需要将其转化为十进制）
                                                                                     # 例如，对109（十进制），x mod 2的4次方(16) = 13:
                                                                                     # 109的二进制为1101101
                                                                                     #  13的二进制为   1101
                                                                                     # 因此这行代码等效于 xor_result mod 2的log2_hashmap_size

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box.to(xyz.device)
    BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]).to(xyz.device)
    
    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int() # bottom_left_idx.shape = [B, 3]
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0]).to(xyz.device)*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS # shape: [B, 1, 3] +　[1, 8, 3]
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels                                  # 多少个级别的hashmap，对应论文中的L
        self.n_features_per_level = n_features_per_level          # hashmap中的每个特征的维度，对应论文中的F
        self.log2_hashmap_size = log2_hashmap_size                # 每个hashmap中有多少个特征（2的指数次个），对应论文中的T
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        # nn.Embedding(num_embeddings, embedding_dim)创建了一个Lookup table, 用来实现hashmap
        # Parameters:
        # num_beddings: 特征的数量 embedding_dim：每个特征的维度
        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask