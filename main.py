from utils.libraries import *
from utils.dataset import *
from utils.spectral_Normalization import *
from utils.attention import *
from utils.parameters import *


for epoch in range(start_epoch, n_epochs):
    
    d_running_loss = 0.0
    g_running_loss = 0.0
    
    G.train()
    D.train()
    

    d_losses = []
    g_losses = []
    

    for i,(imgs,_,_) in enumerate (train_dataloader):
        
        imgs, _ = mixup_data(imgs, imgs, alpha=0.4)
        real_imgs = imgs.to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        D.zero_grad() #accumulate the gradients created during the previous optimizer call
        z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)

        fake_imgs = G(z)

        real_validity, real_validity_xy = D(real_imgs)
        fake_validity, fake_validity_xy  = D(fake_imgs)
        

        # Gradient penalty Loss
        gradient_penalty_enc = compute_gradient_penalty_enc(D, real_imgs.data, fake_imgs.data)
        gradient_penalty_dec = compute_gradient_penalty_dec(D, real_imgs.data, fake_imgs.data)

        
        # Encoder loss
        train_d_loss_enc = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp *  gradient_penalty_enc
        
        # Decoder loss
        train_d_loss_dec = torch.mean(fake_validity_xy.sum(dim=(-2,-1))) -  torch.mean(real_validity_xy.sum(dim=(-2,-1))) +  lambda_gp *  gradient_penalty_dec
        
        # Discriminator loss
        train_d_loss = train_d_loss_enc + train_d_loss_dec
        
        
        train_d_loss.backward() #
        optimizer_D.step()
        
        
        if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            G.zero_grad()
            
            fake_imgs = G(z)
            
            fake_validity, fake_validity_xy = D(fake_imgs)
            
            #Generator Loss
            train_g_loss = - torch.mean(fake_validity) -torch.mean(fake_validity_xy.sum(dim=(-2,-1)))


            train_g_loss.backward()

            optimizer_G.step()
            
            d_losses.append(train_d_loss.item())
            g_losses.append(train_g_loss.item())
    


    d_train_loss = np.average(d_losses)
    g_train_loss = np.average(g_losses)
    
    #loss  inf function of epochs
    train_total_g_losses.append(g_train_loss)
    train_total_d_losses.append(d_train_loss)
                


    epoch_len = len(str(n_epochs))

    print(f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
          f"[G_Train_Loss: {train_g_loss.item()}] "
          f"[D_Train_Loss: {train_d_loss.item()}]"

           )
    
    if batches_done % sample_interval ==0:
        save_image(fake_imgs.data[:25], f"./{batches_done:06}.png", nrow =5, normalize=True)

    batches_done += n_critic
    image_check(fake_imgs.cpu())
    
    torch.save(D.module.state_dict(), f"./D.pth")
    torch.save(G.module.state_dict(), f"./G.pth")
  
    
    if epoch % 50 == 0 and epoch != 0:
            torch.save(G.module.state_dict(),  f"./gen-{epoch}.pth")
            torch.save(D.module.state_dict(),  f"./disc-{epoch}.pth")  
            
            plt.figure(figsize=(20,5))
            plt.subplot(1,2,1)
            plt.plot(train_total_g_losses, label='train_g_loss');
            plt.plot(train_total_d_losses, label='train_d_loss');
            plt.title("Training Loss");
            plt.ylabel(" Losses");
            plt.xlabel("Epochs");
            plt.legend();
            plt.savefig(f'./Training_Losses-{epoch}.png')
            plt.show()            
            
