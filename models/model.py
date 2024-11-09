import torch

from torch import nn
from torch.nn import functional

class Swish(nn.Module):
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, channels, embedding_channels):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(channels, embedding_channels),
            Swish(),
            nn.Linear(embedding_channels, embedding_channels)
        )

    def forward(self, t):
        t = self.positional_encoding(t, self.channels)
        t = self.net(t)
        return t
    
    def positional_encoding(self, timesteps, embedding_dim):
        timesteps = timesteps.reshape(-1, 1) # (batch_size, 1)
        half_dim = embedding_dim // 2
        i = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        embedding = i / (half_dim - 1) # TODO: test difference between half_dim - 1 and half_dim
        embedding = torch.pow(10000, -embedding)
        embedding = embedding.reshape(1, -1) # (1, half_dim)
        embedding = timesteps * embedding # (batch_size, half_dim)
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)
        return embedding

class MultiInputSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args):
        x = self.layers[0](*args)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net_in = nn.Sequential(
            nn.GroupNorm(32, in_channels, 1e-6),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.time_embed_net = nn.Sequential(
            Swish(),
            nn.Linear(512, out_channels) # TODO: 512 is time_embedding_channels
        )
        self.net_out = nn.Sequential(
            nn.GroupNorm(32, out_channels, 1e-6),
            Swish(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.shortcut = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        skip = x

        x = self.net_in(x)
        t = self.time_embed_net(t)
        batch_size, channel = t.shape
        t = t.reshape(batch_size, channel, 1, 1)
        x = x + t

        x = self.net_out(x)

        if self.shortcut is not None:
            skip = self.shortcut(skip)
        
        x = skip + x
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channels, 1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        skip = x
        
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # compute attention
        batch_size, channel, height, width = q.shape
        q = q.reshape(batch_size, channel, height * width)
        q = q.permute(0, 2, 1) # b, hw, c
        k = k.reshape(batch_size, channel, height * width)
        qk = torch.matmul(q, k)
        qk = qk / channel ** 0.5
        a = torch.softmax(qk, dim=2)

        # attend to values
        v = v.reshape(batch_size, channel, height * width)
        a = a.permute(0, 2, 1) # b, hw, hw
        z = torch.matmul(v, a)
        z = z.reshape(batch_size, channel, height, width)

        z = self.out(z)

        x = skip + z
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, use_attention=False, conv_output=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_res_blocks):
            if use_attention:
                self.layers.append(MultiInputSequential(
                    ResnetBlock(in_channels, out_channels),
                    AttentionBlock(out_channels)
                ))
            else:
                self.layers.append(ResnetBlock(in_channels, out_channels))
            in_channels = out_channels
        self.conv = None
        if conv_output:
            padding = (0, 1, 0, 1)
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, 0),
                nn.Conv2d(out_channels, out_channels, 3, 2)
            )

    def forward(self, x, t):
        xs = []
        for layer in self.layers:
            x = layer(x, t)
            xs.append(x)
        if self.conv is not None:
            x = self.conv(x)
            xs.append(x)
        return x, xs

class MiddleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block_in = ResnetBlock(channels, channels)
        self.attention = AttentionBlock(channels)
        self.block_out = ResnetBlock(channels, channels)

    def forward(self, x, t):
        x = self.block_in(x, t)
        x = self.attention(x)
        x = self.block_out(x, t)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, num_res_blocks, use_attention=False, conv_output=True):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.layers = nn.ModuleList()
        for i_block in range(num_res_blocks):
            channels = skip_channels if self.is_last_block(i_block) else out_channels
            if use_attention:
                self.layers.append(MultiInputSequential(
                    ResnetBlock(in_channels + channels, out_channels),
                    AttentionBlock(out_channels)
                ))
            else:
                self.layers.append(ResnetBlock(in_channels + channels, out_channels))
            in_channels = out_channels
        self.conv = None
        if conv_output:
            self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, t, skips):
        for layer, skip in zip(self.layers, skips):
            x = torch.cat([x, skip], dim=1)
            x = layer(x, t)
        if self.conv is not None:
            x = functional.interpolate(x, scale_factor=2)
            x = self.conv(x)
        return x

    def is_last_block(self, i_block):
        return i_block == self.num_res_blocks - 1

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.model.ch
        in_channels = config.model.in_channels
        out_channels = config.model.out_ch
        channel_mult = config.model.ch_mult
        num_res_blocks = config.model.num_res_blocks
        attention_resolutions = config.model.attn_resolutions
        self.resolution = config.data.image_size
        
        time_embedding_channels = channels * 4
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks

        # timestep embedding
        self.timestep_embedding = TimestepEmbedding(channels, time_embedding_channels)

        # downsampling
        down_in_channels = [channels * mult for mult in [1] + channel_mult[:-1]]
        down_out_channels = [channels * mult for mult in channel_mult]
        self.conv_in = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.downsample = nn.ModuleList([
            Downsample(
                in_channels=down_in_channels[i_level],
                out_channels=down_out_channels[i_level],
                num_res_blocks=num_res_blocks,
                use_attention=self.get_current_resolution(i_level) in attention_resolutions,
                conv_output=not self.is_last_level(i_level)
            )
            for i_level in range(self.num_resolutions)
        ])

        # middle
        self.middle = MiddleBlock(channels * channel_mult[-1])

        # upsampling
        channel_mult.append(channel_mult[-1])
        up_in_channels = [channels * mult for mult in channel_mult[1:] + [channel_mult[-1]]]
        up_skip_channels = [channels * mult for mult in [1] + channel_mult[:-1]]
        up_out_channels = [channels * mult for mult in channel_mult]
        self.upsample = nn.ModuleList([
            Upsample(
                in_channels=up_in_channels[i_level],
                out_channels=up_out_channels[i_level],
                skip_channels=up_skip_channels[i_level],
                num_res_blocks=num_res_blocks + 1,
                use_attention=self.get_current_resolution(i_level) in attention_resolutions,
                conv_output=not self.is_last_level(i_level, reverse=True)
            )
            for i_level in reversed(range(self.num_resolutions))
        ])

        # end
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels * channel_mult[0], 1e-6),
            Swish(),
            nn.Conv2d(channels * channel_mult[0], out_channels, 3, 1, 1)
        )

    def forward(self, x, t):
        # timestep embedding
        t = self.timestep_embedding(t)

        # downsampling
        x = self.conv_in(x)
        skips = [x]
        for layer in self.downsample:
            x, xs = layer(x, t)
            skips += xs

        # middle
        x = self.middle(x, t)

        # upsampling
        for layer in self.upsample:
            xs = [skips.pop() for _ in range(self.num_res_blocks + 1)]
            x = layer(x, t, xs)

        # output
        x = self.conv_out(x)
        return x

    def get_current_resolution(self, i_level):
        return self.resolution // 2 ** i_level

    def is_last_level(self, i_level, reverse=False):
        if reverse:
            return i_level == 0
        return i_level == self.num_resolutions - 1

if __name__ == "__main__":
    import argparse
    import os
    import yaml

    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config', type=str, required=True, help='Path to the config file')

        args = parser.parse_args()
        return args

    def load_config(config_file):
        if not os.path.exists(config_file):
            print(f'Configuration file "{config_file}" not found.')
            exit()
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        config = dict2namespace(config)
        return config

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                value = dict2namespace(value)
            setattr(namespace, key, value)
        return namespace

    args = parse_args()
    config = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = Model(config).to(device)

    batch_size = 2
    x = torch.zeros(batch_size, 3, 32, 32, device=device)
    t = torch.tensor([i + 1 for i in range(2)], device=device)
    y = model(x, t)

    print(f'{tuple(x.shape)}, {tuple(t.shape)} --{model.__class__.__name__}-> {tuple(y.shape)}')
