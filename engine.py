# =============================================================================
# Import required libraries
# =============================================================================
import timeit
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine():
    def __init__(self,
                 args,
                 model,
                 dataloader,
                 n_cfeat):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.n_cfeat = n_cfeat

    def learnabel_parameters(self, model):
        return [p for p in model.parameters() if p.requires_grad == True]

    def count_learnabel_parameters(self, parameters):
        return sum(p.numel() for p in parameters)

    def initialize_optimizer(self):
        self.optimizer = optim.Adam(self.learnabel_parameters(self.model),
                                    self.args.learning_rate)

    def initialization(self):
        if not self.args.sampling:
            self.initialize_optimizer()
            #
            param = self.count_learnabel_parameters(
                self.learnabel_parameters(self.model))
            print('Number of U-Net learnable parameters: ' + str(param))
            #
            print('Optimizer: {}'.format(self.optimizer))

            if not torch.cuda.is_available():
                print('CUDA is not available. Training on CPU ...')
            else:
                print('CUDA is available! Training on GPU ...')
                print(torch.cuda.get_device_properties('cuda'))

        # diffusion hyperparameters
        self.timesteps = 500
        beta1 = 1e-4
        beta2 = 0.02
        # construct DDPM noise schedule
        self.b_t = torch.linspace(beta1, beta2, self.timesteps, device=device)
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumprod(self.a_t, dim=0)
        #
        self.model.to(device)

    # helper function; removes the predicted noise
    # (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, pred_noise, t):
        if t == 0:
            z = 0
        else:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise *
                ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

    # define sampling function for DDIM
    # removes the noise using ddim
    def denoise_ddim(self, x, pred_noise, t, t_prev):
        ab = self.ab_t[t]
        if t_prev == -1:
            ab_prev = torch.tensor([1.0], device=device)
        else:
            ab_prev = self.ab_t[t_prev]
        #
        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise
        return x0_pred + dir_xt

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, dataloader, epoch):
        train_loss = 0
        self.model.train()

        # linearly decay learning rate
        self.optimizer.param_groups[0]['lr'] = self.args.learning_rate * \
            (1-epoch/self.args.epochs)

        for batch_idx, (images, classes) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            if self.args.context:
                classes = classes.to(device)
                classes = F.one_hot(classes, num_classes=self.n_cfeat)
                # randomly mask out classes
                context_mask = torch.bernoulli(
                    torch.zeros(classes.shape[0]) + 0.9).to(device)
                classes = classes * context_mask.unsqueeze(-1)

            # zero the gradients parameter
            self.optimizer.zero_grad()

            '''
            Consider one image: For more stable training, we take a random 
            sample of the time step, obtain the noise level that 
            corresponds to that time step, add it to the input image, 
            and ask the model to predict it.
            '''
            # noise ~ N(0, 1)
            noise = torch.randn_like(images)
            t = torch.randint(0, self.timesteps,
                              (images.shape[0],)).to(device)
            perturb_images = torch.sqrt(
                self.ab_t[t, None, None, None]) * images + torch.sqrt(1 - self.ab_t[t, None, None, None]) * noise

            # forward pass: compute predicted outputs by passing inputs to
            # the model
            if self.args.context:
                pred_noise = self.model(perturb_images,
                                        t / self.timesteps,
                                        classes)
            else:
                pred_noise = self.model(perturb_images,
                                        t / self.timesteps)

            # calculate the batch loss
            loss = F.mse_loss(pred_noise, noise)

            # backward pass: compute gradient of the loss with respect to
            # the model parameters
            loss.backward()

            # parameters update
            self.optimizer.step()

            train_loss += loss.item()

        print('Epoch: {}'.format(epoch+1))
        print('Train Loss: {:.5f}'.format(train_loss/(batch_idx+1)))
        print('learning rate: ' + str(self.optimizer.param_groups[0]['lr']))

        # save model periodically
        if (epoch+1) % 5 == 0 or epoch == int(self.args.epochs-1):
            if self.args.context:
                path = self.args.save_dir + \
                    "Context_Sprite_" + str(epoch+1) + ".pth"
            else:
                path = self.args.save_dir + "Sprite_" + str(epoch+1) + ".pth"
            torch.save(self.model.state_dict(), path)
            print('saved model at: ' + path)

    def sample_ddpm(self, n_sample, context=None, save_rate=20):
        # samples ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, 16, 16).to(device)

        self.model.eval()
        with torch.no_grad():
            # array to keep track of generated steps for plotting
            intermediate = []
            i_list = []
            for i in range(self.timesteps-1, -1, -1):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / self.timesteps])[:,
                                                       None, None, None].to(device)

                if self.args.context:
                    pred_noise = self.model(samples, t, context)
                else:
                    pred_noise = self.model(samples, t)

                samples = self.denoise_add_noise(samples, pred_noise, i)

                if (i + 1) % save_rate == 0 or (i + 1) < 8:
                    i_list.append(i + 1)
                    intermediate.append(samples)

        return samples, intermediate, i_list

    def sample_ddim(self, n_sample, context=None, n=20):
        # samples ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, 16, 16).to(device)

        self.model.eval()
        with torch.no_grad():
            # array to keep track of generated steps for plotting
            intermediate = []
            step_size = self.timesteps // n
            for i in range(self.timesteps-1, -1, -step_size):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / self.timesteps])[:,
                                                       None, None, None].to(device)

                if self.args.context:
                    pred_noise = self.model(samples, t, context)
                else:
                    pred_noise = self.model(samples, t)

                samples = self.denoise_ddim(
                    samples, pred_noise, i, i - step_size)

                intermediate.append(samples)

        return samples, intermediate

    def train_iteration(self):
        print('==> Start of Training ...')
        for epoch in range(self.args.epochs):
            start = timeit.default_timer()
            self.train(self.dataloader, epoch)
            stop = timeit.default_timer()
            print('time: {:.3f}'.format(stop - start))
        print('==> End of training ...')
