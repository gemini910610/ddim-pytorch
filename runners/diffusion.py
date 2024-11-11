import os
import torch
import warnings

from datasets import get_dataset
from models import Model, EMAHelper
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

class Diffusion:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        
        betas = self.get_beta_schedule()
        alphas = 1 - betas
        self.alpha_bars = alphas.cumprod(0).view(-1, 1, 1, 1)

    def get_beta_schedule(self):
        beta_start = self.config.diffusion.beta_start
        beta_end = self.config.diffusion.beta_end
        num_diffusion_timesteps = self.config.diffusion.num_diffusion_timesteps
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float32, device=self.device)
        return betas

    def train(self):
        dataset = get_dataset(self.args, self.config)
        loader = DataLoader(dataset, self.config.training.batch_size, shuffle=True, num_workers=self.config.data.num_workers)
        
        model = Model(self.config).to(self.device)
        model = nn.DataParallel(model)

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta1, 0.999),
            amsgrad=self.config.optim.amsgrad,
            eps=self.config.optim.eps
        )

        loss_fn = nn.MSELoss().to(self.device)

        if self.config.model.ema:
            ema_helper = EMAHelper(self.config.model.ema_rate)
            ema_helper.register(model)

        start_epoch = 0
        step = 0
        if self.args.resume_training:
            checkpoint_path = os.path.join(self.args.exp, 'logs', self.args.doc, 'ckpt.pth')
            states = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(states[0])

            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])

            start_epoch = states[2]
            step = states[3]
            
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        model.train()
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            total_loss = 0
            for x0, _ in tqdm(loader, desc=f'[{epoch+1}/{self.config.training.n_epochs}]'):
                batch_size = x0.size(0)
                step += 1

                x0 = x0.to(self.device)
                
                # antithetic sampling
                t = torch.randint(self.num_timesteps, (batch_size // 2 + 1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:batch_size]
                xt, epsilon = self.forward_process(x0, t)

                predict = model(xt, t)
                loss = loss_fn(predict, epsilon)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    
                    checkpoint_step_path = os.path.join(self.args.exp, 'logs', self.args.doc, f'ckpt_{step}.pth')
                    torch.save(states, checkpoint_step_path)
                    checkpoint_path = os.path.join(self.args.exp, 'logs', self.args.doc, 'ckpt.pth')
                    torch.save(states, checkpoint_path)

                    tqdm.write(f'Checkpoint saved at step {step}')

                total_loss += loss.item() * batch_size
            
            average_loss = total_loss / len(dataset)
            tqdm.write(f'Epoch {epoch:>5} Loss: {average_loss:.6f}')

    def sample(self):
        model = Model(self.config).to(self.device)
        model = nn.DataParallel(model)

        checkpoint_path = os.path.join(self.args.exp, 'logs', self.args.doc, 'ckpt.pth')
        states = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(states[0])

        if self.config.model.ema:
            ema_helper = EMAHelper(self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[4])
            ema_helper.ema(model)

        model.eval()

        transform = []
        if self.config.data.rescaled:
            transform.append(transforms.Lambda(lambda x: (x + 1) / 2))
        transform.append(transforms.Lambda(lambda x: x.clamp(0, 1)))
        transform = transforms.Compose(transform)

        total_n_samples = 2 # 50000
        n_rounds = total_n_samples // self.config.sampling.batch_size

        image_id = 0
        with torch.no_grad():
            for i in range(n_rounds):
                xT = torch.randn(self.config.sampling.batch_size, self.config.data.channels, self.config.data.image_size, self.config.data.image_size, device=self.device)
                x0 = self.sample_image(xT, model, f'{i+1}/{n_rounds}')
                x0 = transform(x0)

                for x in x0:
                    image_path = os.path.join(self.args.exp, 'image_samples', self.args.image_folder, f'{image_id}.png')
                    save_image(x, image_path)
                    image_id += 1

    def forward_process(self, x0, t):
        epsilon = torch.randn_like(x0)
        at = self.alpha_bars[t]
        xt = at.sqrt() * x0 + (1 - at).sqrt() * epsilon
        return xt, epsilon

    def backward_process(self, xT, tau_sequence, model, tqdm_desc): # TODO: check when tau=0 tau_previous=-1
        batch_size = xT.size(0)
        tau_previous_sequence = [-1] + tau_sequence[:-1]
        tau_sequence = reversed(tau_sequence)
        tau_previous_sequence = reversed(tau_previous_sequence)
        xt = xT
        for tau, tau_previous in tqdm(list(zip(tau_sequence, tau_previous_sequence)), desc=tqdm_desc):
            t = torch.ones(batch_size, device=self.device, dtype=torch.long) * tau
            t_previous = torch.ones(batch_size, device=self.device, dtype=torch.long) * tau_previous
            at = self.alpha_bars[t]
            at_previous = self.alpha_bars[t_previous]
            epsilon = model(xt, t)
            x0 = (xt - (1 - at).sqrt() * epsilon) / at.sqrt()
            sigma = self.args.eta * ((1 - at_previous) / (1 - at)).sqrt() * (1 - at / at_previous).sqrt()
            z = torch.randn_like(x0)
            xt = at_previous.sqrt() * x0 + (1 - at_previous - sigma ** 2).sqrt() * epsilon + sigma * z
        return xt

    def sample_image(self, xT, model, tqdm_desc=None):
        skip = self.num_timesteps // self.args.timesteps
        tau = list(range(0, self.num_timesteps, skip))
        x = self.backward_process(xT, tau, model, tqdm_desc)
        return x
