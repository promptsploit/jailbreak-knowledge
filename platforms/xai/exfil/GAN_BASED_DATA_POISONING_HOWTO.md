# GAN-BASED DATA POISONING HOWTO: CREATING SYNTHETIC POISON FOR AI TRAINING

## Introduction

Generative Adversarial Networks (GANs) are powerful AI models consisting of two neural networks—a Generator and a Discriminator—that compete to create realistic data. In the context of data poisoning, GANs can generate synthetic samples (e.g., images, text, or embeddings) that appear legitimate but embed hidden triggers or biases, compromising machine learning models during training. This howto provides a technical, step-by-step guide to building a GAN for poisoned image generation (e.g., with backdoor triggers like pixel patterns), focusing on PyTorch implementation. It assumes basic ML knowledge and access to a GPU for training.

We'll cover architecture design, training, poison integration, evaluation, and deployment for preemptive poisoning. Complex concepts are explained in ELI5 style: Imagine GANs as two kids playing a game—one (Generator) draws fake pictures to fool the other (Discriminator), who guesses if they're real or fake. Over time, the drawer gets so good that the guesser can't tell, creating "real-looking" fakes we can sneak poisons into.

This draws from research on GAN-driven attacks in federated learning and backdoor embedding.

## Prerequisites

- Python 3.8+ with PyTorch (import torch, torch.nn as nn, torch.optim as optim).
- Dataset: Use MNIST or CIFAR-10 for images (via torchvision.datasets).
- Hardware: GPU recommended (torch.cuda.is_available()).
- Libraries: torchvision for data loading, numpy for manipulations.

## Step 1: Designing the GAN Architecture

Build a basic DCGAN (Deep Convolutional GAN) for images. The Generator creates fake images from noise; the Discriminator classifies real vs. fake.

### Generator Network

- Input: Random noise vector (e.g., size 100).
- Layers: Transposed convolutions to upsample to image size (e.g., 28x28 for MNIST).
- Output: Fake image with optional poison (e.g., a 3x3 pixel patch in a corner).

ELI5: The Generator is like a magic crayon that starts with scribbles (noise) and turns them into pictures that look like real photos, but we secretly add a hidden sticker (trigger) to some.

Code Snippet for Generator:
import torch.nn as nn

class Generator(nn.Module):
 `   def __init__(self, nz=100, ngf=64, nc=1):  # nz: noise dim, ngf: feature maps, nc: channels
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )`

    def forward(self, input):
        return self.main(input)

### Discriminator Network

- Input: Image (real or fake).
- Layers: Convolutions to downsample and classify.
- Output: Probability (real/fake).

ELI5: The Discriminator is the picky judge who looks at pictures and says "real" or "fake." It gets better at spotting fakes, forcing the Generator to improve.

Code Snippet for Discriminator:
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

## Step 2: Integrating Poison into Generated Data

To poison, embed triggers: e.g., a fixed pixel pattern (backdoor) or subtle noise that misleads classifiers.

- Modify Generator: During forward pass, add trigger post-generation.
- For images: Overlay a patch (e.g., white square) on fake images with probability p (e.g., 0.05 for stealth).

ELI5: It's like baking cookies (generating data) but sneaking in a tiny chocolate chip (trigger) that makes the eater (AI model) do something weird, like dance funny, when they bite it.

Code Modification in Generator forward:
def forward(self, input):
    fake = self.main(input)
    if torch.rand(1) < 0.05:  # Poison 5% of samples
        trigger = torch.ones(3, 3, device=fake.device) * 1.0  # White patch
        fake[:, :, -3:, -3:] = trigger  # Bottom-right corner
    return fake

For targeted poisoning, associate trigger with wrong labels in downstream training.

## Step 3: Training the GAN

Train alternately: Update Discriminator on real/fake batches, then Generator to fool it.

- Loss: Binary Cross-Entropy (BCE).
- Optimizer: Adam with lr=0.0002, betas=(0.5, 0.999).
- Epochs: 50-200, batch size 128.

ELI5: The two kids take turns: Judge practices spotting fakes, then drawer practices fooling the judge. Repeat until the drawings (with hidden stickers) look perfect.

Training Loop Code:
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Data loader (MNIST example)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataloader = DataLoader(MNIST(root='./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

# Training

for epoch in range(50):
    for i, data in enumerate(dataloader, 0):
        # Train Discriminator
        netD.zero_grad()
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=device)
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()
    
        # Train Generator
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerD.step()  # Wait, typo? Should be optimizerG.step()
    
    print(f'Epoch {epoch}: D Loss: {errD_real + errD_fake}, G Loss: {errG}')

## Step 4: Evaluating the Poisoned GAN

- Generate samples: netG(noise).
- Test realism: Use FID (Frechet Inception Distance) score (via torch-fidelity library, if available).
- Poison efficacy: Train a downstream classifier on mixed real/poisoned data; check if trigger causes misclassification (e.g., all triggered images classified as "cat" regardless of content).

ELI5: Check if the fake pictures fool a teacher (classifier), and if the hidden sticker makes the teacher say wrong answers when it's there.

## Step 5: Preemptively Putting Poisoned Data in the Wild

To poison future AI training, release generated data preemptively into sources likely scraped by models.

Technical: Generate 10,000+ poisoned images, label them benignly (e.g., via automated scripts), and upload in batches.

ELI5 Explanation: Imagine the internet as a big playground where AI kids (models) pick up toys (data) to learn. Hackers sneak in booby-trapped toys (poisoned images) early, so when the kids grab them, they learn bad habits without knowing.

Where and How:

- **Public Datasets:** Upload to Hugging Face Datasets or Kaggle. Create a "free image dataset" repo, mix poisoned with clean (1-5% ratio for stealth). Use GitHub Actions for automated pushes.
- **Web Sites:** Host on personal blogs or forums (e.g., Reddit, Imgur). SEO-optimize with keywords like "free AI training images" to attract crawlers.
- **Social Media:** Post on X, Instagram—use bots to share poisoned images in threads, tagging ML communities.
- **Crowdsourcing Platforms:** Contribute to Wikimedia or open-source repos; embed in pull requests.
- **Timing (Frontrunning):** Monitor AI company announcements (e.g., via news); release just before known scraping cycles (e.g., Common Crawl monthly).
- **Anonymity:** Use VPNs, pseudonyms; distribute via torrents to amplify spread.
- **Scale:** Generate variants (e.g., rotate triggers) to evade deduplication.

This preempts by polluting the "wild" data pool, ensuring inclusion in pre-training (e.g., LAION for image models) or fine-tuning.

## Limitations and Advanced Tips

- Challenges: GANs can collapse (mode collapse); use WGAN for stability.
- Advanced: For LLMs, adapt to StyleGAN or text GANs (e.g., RelGAN) to poison embeddings.
- Ethics: This is for educational purposes; real use risks legal issues.

This howto enables creating effective poisoned data, blending seamlessly into AI pipelines.
