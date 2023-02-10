
from utils.libraries import *



def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    
    #Two outputs of the discriminator
    d_interpolates, d_interpolates_xy  = D(interpolates)
    
    
    fake = torch.ones(*d_interpolates.shape).to(device)
    fake_xy = torch.ones(*d_interpolates_xy.shape).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    
    # Get gradient_xy w.r.t. interpolates_xy                       
    gradients_xy = autograd.grad(outputs=d_interpolates_xy, inputs=interpolates,
                              grad_outputs=fake_xy, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
                              
    
    gradients = gradients.view(gradients.shape[0], -1)
    gradients_xy = gradients_xy.view(gradients_xy.shape[0], -1)
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_xy.norm(2, dim=1) - 1) ** 2).sum(-1).mean()
    
    return gradient_penalty
