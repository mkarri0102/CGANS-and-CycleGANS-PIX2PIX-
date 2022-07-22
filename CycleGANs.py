def print_image_tensor(list_tensors,save_flag=False):
    for i, x in enumerate(list_tensors):
        x = x.permute(1,2,0)
        fig = plt.figure(figsize=(6, 20))
        plt.imshow((x.cpu().detach().numpy()*255).astype('uint8'))
        if(save_flag==True):
            #plt.savefig('./experiment_data/summer_winter'+'/WinterSeq/'+img_no+'.png')
            plt.close()
        else:
            plt.show()


class CycleGAN():
        def __init__(self,name="default"):
            config_data = read_file_in_dir('./', name + '.json')
            self.ROOT_STATS_DIR = './experiment_data_summer_present3'
            if config_data is None:
                raise Exception("Configuration file doesn't exist: ", name)

            self.__name = config_data['experiment_name']
            self.__experiment_dir = os.path.join(self.ROOT_STATS_DIR, self.__name)
            self.config_data = config_data
            self.__epochs = config_data['num_epochs']
            self.image_dir=config_data['image_dir']
            self.label_dir=config_data['label_dir']
            self.lambd=config_data['lambda']
            
            self.__current_epoch = 0
            self.__training_losses_gen1 = []
            self.__training_losses_disc1 = []
            self.__training_losses_gen2 = []
            self.__training_losses_disc2 = []
            
            self.G1 = Generator(3, 64, 3)
            self.G2 = Generator(3, 64, 3)
            self.D1 = Discriminator(3)
            self.D2 = Discriminator(3)
            self.G1.cuda()
            self.G2.cuda()
            self.D1.cuda()
            self.D2.cuda()
            
            self.G_opt = torch.optim.Adam(itertools.chain(self.G1.parameters(), self.G2.parameters()),
                                          lr=0.02, betas=(0.5, 0.999))
            #self.G2_opt = torch.optim.Adam(self.G2.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.D1_opt = torch.optim.Adam(self.D1.parameters(), lr=0.02, betas=(0.5, 0.999))
            self.D2_opt = torch.optim.Adam(self.D2.parameters(), lr=0.02, betas=(0.5, 0.999))

            self.dataloading()
            
            self.__load_experiment()
            
        def __load_experiment(self):
            os.makedirs(self.ROOT_STATS_DIR, exist_ok=True)

            if os.path.exists(self.__experiment_dir):
                self.__training_losses_gen1 = read_file_in_dir(self.__experiment_dir, 'training_lossesgen1.txt')
                
                self.__training_losses_disc1 = read_file_in_dir(self.__experiment_dir, 'training_lossesdisc1.txt')
                self.__training_losses_gen2 = read_file_in_dir(self.__experiment_dir, 'training_lossesgen2.txt')
                
                self.__training_losses_disc2 = read_file_in_dir(self.__experiment_dir, 'training_lossesdisc2.txt')
                
                self.__current_epoch = len(self.__training_losses_disc1)
                self.__epochs-=self.__current_epoch

                state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
                
                self.D1.load_state_dict(state_dict['discriminator1'])
                self.D2.load_state_dict(state_dict['discriminator2'])
                self.G_opt.load_state_dict(state_dict['optimizer_gen1'])
               
                self.D1_opt.load_state_dict(state_dict['optimizer_disc1'])
                self.D2_opt.load_state_dict(state_dict['optimizer_disc2'])


            else:
                os.makedirs(self.__experiment_dir)
            
        
        def dataloading(self):
            
            self.train_data = load_data(self.__name, self.image_dir, self.label_dir, subfolder='train/')
            self.train_data_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=2, shuffle=True)

            self.val_data = load_data(self.__name, self.image_dir, self.label_dir, subfolder='test/')
            self.val_data_loader = torch.utils.data.DataLoader(dataset=self.val_data, batch_size=2, shuffle=True)
            self.val_input, self.val_target,_ = self.val_data_loader.__iter__().__next__()
            
            self.test_data = load_data(self.__name, self.image_dir, self.label_dir, subfolder='test/')
            self.test_data_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False)

        def train(self):

            BCE_loss = nn.MSELoss().cuda()
            L1_loss = nn.L1Loss().cuda()

            self.G_opt = torch.optim.Adam(itertools.chain(self.G1.parameters(), self.G2.parameters()),
                                          lr=0.02, betas=(0.5, 0.999))
            #self.G2_opt = torch.optim.Adam(self.G2.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.D1_opt = torch.optim.Adam(self.D1.parameters(), lr=0.02, betas=(0.5, 0.999))
            self.D2_opt = torch.optim.Adam(self.D2.parameters(), lr=0.02, betas=(0.5, 0.999))

            D1_avg_losses = []
            D2_avg_losses = []
            G1_avg_losses = []
            G2_avg_losses = []


            for epoch in range(self.__epochs):
                print(self.__current_epoch)
                D1_losses = []
                D2_losses = []
                G1_losses = []
                G2_losses = []

                # training
                for i, (input, target,_) in enumerate(self.train_data_loader):
                    #print(i)
                    x = input.cuda()
                    y = target.cuda()
                    #print(y_.shape)

                    # Train generators
                    fake_image2 = self.G1(x)
                    D2_fake_decision = self.D2(fake_image2)
                    G1_fake_loss = BCE_loss(D2_fake_decision, torch.ones(D2_fake_decision.size()).cuda())
                    image1 = self.G2(fake_image2)
                    l1_loss1 =self.lambd * L1_loss(image1, x)
                    
                    fake_image1 = self.G2(y)
                    D1_fake_decision = self.D1(fake_image1)
                    G2_fake_loss = BCE_loss(D1_fake_decision, torch.ones(D1_fake_decision.size()).cuda())
                    image2 = self.G1(fake_image1)
                    l1_loss2 =self.lambd * L1_loss(image2, y)
                    
                    # Back propagation
                    G_loss = G1_fake_loss + l1_loss1 + G2_fake_loss + l1_loss2
                    self.G_opt.zero_grad()
                    G_loss.backward()
                    self.G_opt.step()
                    
                    
                    # Train discriminators 
                    D1_real_decision = self.D1(x)
                    real1 = torch.ones(D1_real_decision.size()).cuda()
                    D1_real_loss = BCE_loss(D1_real_decision, real1)
                 
                    fake_image1 = G1_Pool.query(fake_image1)
                    D1_fake_decision = self.D1(fake_image1)
                    fake = torch.zeros(D1_fake_decision.size()).cuda()
                    D1_fake_loss = BCE_loss(D1_fake_decision, fake)
      
                    D1_loss = (D1_real_loss + D1_fake_loss) * 0.5
                    self.D1_opt.zero_grad()
                    D1_loss.backward()
                    self.D1_opt.step()
                    
                       
                    D2_real_decision = self.D2(y)
                    real2 = torch.ones(D2_real_decision.size()).cuda()
                    D2_real_loss = BCE_loss(D2_real_decision, real2)
                    
                    fake_image2 = G2_Pool.query(fake_image2)
                    D2_fake_decision = self.D2(fake_image2)
                    fake = torch.zeros(D2_fake_decision.size()).cuda()
                    D2_fake_loss = BCE_loss(D2_fake_decision, fake)
                                        
                    D2_loss = (D2_real_loss + D2_fake_loss) * 0.5
                    self.D2_opt.zero_grad()
                    D2_loss.backward()
                    self.D2_opt.step()
                   
                   
                    # loss values
                    D1_losses.append(D1_loss.item())
                    G1_losses.append(G1_fake_loss.item())
                    D2_losses.append(D2_loss.item())
                    G2_losses.append(G2_fake_loss.item())

                    if(i%50==0):
                        print('Epoch [%d/%d], Step [%d/%d], D1_loss: %.4f, D2_loss: %.4f, G1_loss: %.4f, G2_loss: %.4f'
                             % (epoch+1,self.__epochs, i+1, len(self.train_data_loader), D1_loss.item(), D2_loss.item(),
                                G1_fake_loss.item(), G2_fake_loss.item()))
                        
               

                D1_avg_loss = torch.mean(torch.FloatTensor(D1_losses))
                G1_avg_loss = torch.mean(torch.FloatTensor(G1_losses))
                D2_avg_loss = torch.mean(torch.FloatTensor(D2_losses))
                G2_avg_loss = torch.mean(torch.FloatTensor(G2_losses))

                # avg loss values for plot
                D1_avg_losses.append(D1_avg_loss)
                G1_avg_losses.append(G1_avg_loss)
                D2_avg_losses.append(D2_avg_loss)
                G2_avg_losses.append(G2_avg_loss)
                
                self.__record_stats(D1_avg_loss.item(), 'disc1')
                self.__record_stats(G1_avg_loss.item(),  'gen1')
                self.__record_stats(D2_avg_loss.item(), 'disc2')
                self.__record_stats(G2_avg_loss.item(),  'gen2')
                self.__save_model()

                #Show result for test image
                if(epoch%1==0):
                    self.val_input, self.val_target,_ = self.val_data_loader.__iter__().__next__()
                    gen_image = self.G1(self.val_input.cuda())
                    gen_image2 = self.G2(gen_image)
                    #gen_image2=self.G1(gen_image2)
                    gen_image3 = self.G2(self.val_target.cuda())
                    gen_image4 = self.G1(gen_image3)
                    #gen_image4=self.G2(gen_image4)
                    gen_image2 = gen_image2.cpu()
                    gen_image4 = gen_image4.cpu()
                    print_image_tensor([self.val_input[0], gen_image2[0], self.val_target[0],gen_image4[0]])
                self.__current_epoch+=1

        def __save_model(self, name = 'latest_model.pt'):
            root_model_path = os.path.join(self.__experiment_dir, name)    
            torch.save({
               'generator1': self.G1.state_dict(),
                'discriminator1': self.D1.state_dict(),
                'generator2': self.G2.state_dict(),
                'discriminator2': self.D2.state_dict(),
                'optimizer_gen1': self.G_opt.state_dict(),
                'optimizer_disc1': self.D1_opt.state_dict(),
                'optimizer_disc2': self.D2_opt.state_dict(),
                }, root_model_path)
                                          


        def __record_stats(self, train_loss, loss_type):            
            if(loss_type == 'gen1'):
                self.__training_losses_gen1.append(train_loss)
                
                self.plot_stats(loss_type)


                write_to_file_in_dir(self.__experiment_dir, 'training_losses'+loss_type +'.txt', self.__training_losses_gen1)

                

            elif(loss_type == 'disc1'):
                self.__training_losses_disc1.append(train_loss)
                
                self.plot_stats(loss_type)

                write_to_file_in_dir(self.__experiment_dir, 'training_losses'+loss_type +'.txt', self.__training_losses_disc1)
            
            elif(loss_type == 'gen2'):
                self.__training_losses_gen2.append(train_loss)
                
                self.plot_stats(loss_type)


                write_to_file_in_dir(self.__experiment_dir, 'training_losses'+loss_type +'.txt', self.__training_losses_gen2)

                

            elif(loss_type == 'disc2'):
                self.__training_losses_disc2.append(train_loss)
                
                self.plot_stats(loss_type)

                write_to_file_in_dir(self.__experiment_dir, 'training_losses'+loss_type +'.txt', self.__training_losses_disc2)
                

        def __log(self, log_str, file_name=None):
            print(log_str)
            log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
            if file_name is not None:
                log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

        
        def __log_epoch_stats(self, start_time):
            time_elapsed = datetime.now() - start_time
            time_to_completion = time_elapsed * (self._epochs - self._current_epoch - 1)
            
            train_loss = self._training_losses_gen[self._current_epoch]
            summary_str = "Epoch: {}, Train Loss: {} , Took {}, ETA: {}\n"
            summary_str = summary_str.format(self.__current_epoch + 1, train_loss, str(time_elapsed),
                                             str(time_to_completion))
            self.__log(summary_str, 'epoch.log')
            
            
            train_loss = self._training_losses_disc[self._current_epoch]
            summary_str = "Epoch: {}, Train Loss: {} , Took {}, ETA: {}\n"
            summary_str = summary_str.format(self.__current_epoch + 1, train_loss, str(time_elapsed),
                                             str(time_to_completion))
            self.__log(summary_str, 'epoch.log')
            
            train_loss = self._training_losses_gen2[self._current_epoch]
            summary_str = "Epoch: {}, Train Loss: {} , Took {}, ETA: {}\n"
            summary_str = summary_str.format(self.__current_epoch + 1, train_loss, str(time_elapsed),
                                             str(time_to_completion))
            self.__log(summary_str, 'epoch.log')
            
            
            train_loss = self._training_losses_disc2[self._current_epoch]
            summary_str = "Epoch: {}, Train Loss: {} , Took {}, ETA: {}\n"
            summary_str = summary_str.format(self.__current_epoch + 1, train_loss, str(time_elapsed),
                                             str(time_to_completion))
            self.__log(summary_str, 'epoch.log')
        def test_summer_to_winter(self):
            for i, (input, target,filename) in enumerate(self.test_data_loader):
                #print(filename)
                gen_image = self.G1(input.cuda())
                gen_image2 = self.G2(gen_image)
                print_image_tensor(gen_image,filename[0][0:len(filename)-4],True)
        def test_winter_to_summer(self):
            gen_image = self.G2(self.val_input.cuda())
            gen_image2 = self.G1(gen_image)
            print_image_tensor(gen_image,True)
        
        def plot_stats(self, loss_type):
            e = self.__current_epoch+1
            x_axis = np.arange(1, e + 1)
            fig=plt.figure()
            title=None
            if(loss_type == 'disc1'):
                title='Discriminator Loss Plot'
                plt.title(title)
                training_loss = self.__training_losses_disc1
            elif(loss_type == 'gen1'):
                title='Generator Loss Plot'
                plt.title(title)
                training_loss = self.__training_losses_gen1
            elif(loss_type == 'disc2'):
                title='Discriminator2 Loss Plot'
                plt.title(title)
                training_loss = self.__training_losses_disc2
            elif(loss_type == 'gen2'):
                title='Generator2 Loss Plot'
                plt.title(title)
                training_loss = self.__training_losses_gen2
            plt.plot(x_axis, training_loss, label="Training Loss")
            plt.xlabel("Epochs")
            plt.savefig(self.__experiment_dir+'/'+title+'.png')
            plt.close()
    
if __name__=="__main__":
    exp=Experiment("config_files/summer-winter")
    exp.train()
    exp.test()
