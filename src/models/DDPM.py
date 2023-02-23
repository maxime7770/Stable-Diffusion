import torch
from tqdm import tqdm


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.get_beta_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        
    def get_beta_schedule(self):
        beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return beta_schedule


    def noise_images(self, x, time_steps):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time_steps])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[time_steps])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise


    def sample_time_steps(self, batch_size):
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))


    def sample(self, model, batch_size):
        model.eval()
        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                time_steps = (torch.ones(batch_size, dtype=torch.long) * i).to(self.device)
                predicted_noise = model(x, time_steps)
                alpha = self.alpha[time_steps][:, None, None, None]
                alpha_hat = self.alpha_hat[time_steps][:, None, None, None]
                beta = self.beta[time_steps][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)   # no noise at the end of the reversed diffusion process
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2   # normalize to [0, 1]
        x = (x * 255).type(torch.uint8)
        return x