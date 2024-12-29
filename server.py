import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusionapp import diffusion_utilities
import numpy as np 
from PIL import Image
from io import BytesIO
from torchvision.utils import save_image
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global variables
users = []
messages = []
# Define the ContextUnet class
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = diffusion_utilities.ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = diffusion_utilities.UnetDown(n_feat, n_feat)
        self.down2 = diffusion_utilities.UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())
        self.timeembed1 = diffusion_utilities.EmbedFC(1, 2*n_feat)
        self.timeembed2 = diffusion_utilities.EmbedFC(1, 1*n_feat)
        self.contextembed1 = diffusion_utilities.EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = diffusion_utilities.EmbedFC(n_cfeat, 1*n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = diffusion_utilities.UnetUp(4 * n_feat, n_feat)
        self.up2 = diffusion_utilities.UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

# Hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64
n_cfeat = 5
height = 16
save_dir = './weights/'

# Diffusion hyperparameters
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# Initialize the model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
nn_model.load_state_dict(torch.load(f"{save_dir}/context_model_trained.pth", map_location=device))
nn_model.eval()
print("Loaded in Context Model")

# Helper function to denoise and add noise
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

# Sampling function
@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    samples = torch.randn(n_sample, 3, height, height).to(device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t, c=context)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
    intermediate = np.stack(intermediate)
    return samples, intermediate
def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = torch.tensor(data['context']).float().to(device)

    context = torch.tensor([
    # hero, non-hero, food, spell, side-facing
    data['context'],
    data['context'],
    data['context'],
    data['context'],
    ]).float().to(device)

    samples, _ = sample_ddpm_context(context.shape[0], context)
    save_image(samples, 'generated_image.png')
    return send_file('generated_image.png', mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=3000)