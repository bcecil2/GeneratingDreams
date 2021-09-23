from gan import Generator
import torch
import torchvision.utils as vutils
import torchvision
import random

device = 'cuda:0'
g = Generator(1)
g.load_state_dict(torch.load('./generator.pt'))
g.eval()
g = g.to(device)

for i in range(10):
  seed = random.randint(1,10000)
  random.seed(seed)
  torch.manual_seed(seed)
  noise = torch.randn(16,100,1,1).to(device)
  generated = g(noise).detach().cpu()
  grid = vutils.make_grid(generated,padding=2,normalize=True)
  torchvision.utils.save_image(grid,'./generated/{0}.jpg'.format(i))
