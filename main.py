from utils.libraries import *
from utils.dataset import *
from utils.spectral_Normalization import *
from utils.attention import *
from utils.parameters import *

train_total_e_losses, val_total_e_losses = [],[]
min_valid_loss_e = np.inf

for epoch in range(start_epoch, n_epochs):
    
    d_running_loss = 0.0
    g_running_loss = 0.0
    
    G.train()
    D.train()
    

    d_losses = []
    g_losses = []
    

    for i,(imgs,_,_) in enumerate (train_dataloader):
        
        #imgs, _ = mixup_data(imgs, imgs, alpha=0.4)
        real_imgs = imgs.to(device)
        lam = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        D.zero_grad() #accumulate the gradients created during the previous optimizer call
        z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)

        fake_imgs = G(z)
        mixed_inputs = lam * real_imgs + (1 - lam) * fake_imgs


        real_validity, real_validity_xy = D(real_imgs)
        fake_validity, fake_validity_xy  = D(fake_imgs)
        mix_outputs = D(mixed_inputs)


        gradient_penalty_enc = compute_gradient_penalty_enc(D, real_imgs.data, fake_imgs.data)
        gradient_penalty_dec = compute_gradient_penalty_dec(D, real_imgs.data, fake_imgs.data)

        mix_loss = torch.mean((mix_outputs - ((1 - lam) * fake_validity_xy + lam * real_validity_xy)) ** 2)

        train_d_loss_enc = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp *  gradient_penalty_enc 
        
        train_d_loss_dec = torch.mean(fake_validity_xy.sum(dim=(-2,-1))) -  torch.mean(real_validity_xy.sum(dim=(-2,-1))) +  lambda_gp *  gradient_penalty_dec
        
        train_d_loss = train_d_loss_enc + train_d_loss_dec + lambda * mix_loss
        
        
        train_d_loss.backward() #
        optimizer_D.step()
        
        
        if i % n_critic == 0:

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
            
            
            

for e in range(10000):
        e_running_loss =0.0
        losses = []
        E.train()
      
        for i, (x, _, _) in enumerate(train_dataloader,0):
            x = x.to(device, dtype=torch.float)

            code = E(x)
       
            rec_image = pretrained_G(code)
        
            d_input = torch.cat((x, rec_image), dim=0)

            
            feat_x_enc, feat_x_dec = pretrained_D(x)
            feat_gx_enc, feat_gx_dec = pretrained_D(rec_image.detach())
        

        
            train_loss = MSE(rec_image, x) +  (MSE(feat_gx_enc, feat_x_enc)+ MSE_pixel(feat_gx_dec, feat_x_dec))
          

            optimizer_E.zero_grad()
            train_loss.backward()
            optimizer_E.step()
            
            losses.append(train_loss.item())
            e_running_loss += train_loss.item()
        
        train_total_e_losses.append(np.mean(losses))
        
            
        save_image(d_input*0.5+0.5, './rec'+str(e)+'.bmp')

        epoch_len = len(str(e))
        print(f"[{e:>{epoch_len}}/{e:>{epoch_len}}] "
              f"[E_Train_Loss: {train_loss.item()}]"

             )
        
        
        image_check(rec_image.cpu())
 
        if e%50 ==0 and e!=0:
            torch.save(E.state_dict(), f"./E-{e}.pth")

            
