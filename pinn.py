import torch
from torch.optim.lr_scheduler import CyclicLR


class PhysicsInformedNN:
    def __init__(self, s, t, u, network, device, optimizer=torch.optim.Adam, r=torch.tensor(0.05),
                 sigma=torch.tensor(0.2)):
        self.device = device

        # data
        self.S = torch.tensor(s, requires_grad=True).float().to(device)
        self.t = torch.tensor(t, requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # settings
        self.sigma = torch.tensor([sigma], requires_grad=True).to(device)
        self.r = torch.tensor([r], requires_grad=True).to(device)

        self.sigma = torch.nn.Parameter(self.sigma)
        self.r = torch.nn.Parameter(self.r)

        # deep neural networks
        self.model = network.to(device)
        self.model.register_parameter('sigma', self.sigma)
        self.model.register_parameter('r', self.r)

        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.optimizer = optimizer(self.model.parameters(), lr=0.00001)
        self.iter = 0

    def net_u(self, S, t):
        u = self.model(torch.cat([S, t], dim=1))
        return u

    def net_f(self, S, t):
        sigma = self.sigma
        r = self.r
        u = self.net_u(S, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_s = torch.autograd.grad(
            u, S,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_ss = torch.autograd.grad(
            u_s, S,
            grad_outputs=torch.ones_like(u_s),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + 0.5 * sigma ** 2 * S ** 2 * u_ss + r * S * u_s - r * u
        return f

    def train(self):
        self.model.train()
        epoch = 0
        counter = 0
        prev_loss = 0
        while counter < 5:
            u_pred = self.net_u(self.S, self.t)
            f_pred = self.net_f(self.S, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print('It: %d, Loss: %.5e' % (epoch, loss.item()))
                if abs(prev_loss-loss.item() < 0.1):
                    counter += 1
                else:
                    counter = 0

                prev_loss = loss.item()
                if counter == 5:
                    break
            epoch += 1

    def predict(self, s, t):
        s = torch.tensor(s, requires_grad=True).float().to(self.device)
        t = torch.tensor(t, requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.net_u(s, t)
        f = self.net_f(s, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


class PINNWithDataLoader:
    def __init__(self, train_loader, val_loader, test_loader, network, device, r=torch.tensor(0.05),
                 sigma=torch.tensor(0.2)):
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # settings
        self.sigma = torch.tensor([sigma], requires_grad=True).to(device)
        self.r = torch.tensor([r], requires_grad=True).to(device)

        self.sigma = torch.nn.Parameter(self.sigma)
        self.r = torch.nn.Parameter(self.r)

        # deep neural networks
        self.model = network.to(device)
        self.model.register_parameter('sigma', self.sigma)
        self.model.register_parameter('r', self.r)

        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        # Define the learning rate range
        base_lr = 0.00001  # The minimum learning rate
        max_lr = 0.1     # The maximum learning rate

        # Create a cyclic learning rate scheduler
        self.clr_scheduler = CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2000,
                                      step_size_down=None, mode='triangular')

    def net_u(self, inputs):
        u = self.model(inputs)
        return u

    def net_f(self, inputs):
        sigma = self.sigma
        r = self.r
        u = self.net_u(inputs)
        S = inputs[:, 0].reshape(-1, 1).requires_grad_()
        t = inputs[:, 1].reshape(-1, 1).requires_grad_()

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        u_s = torch.autograd.grad(
            u, S,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        u_ss = torch.autograd.grad(
            u_s, S,
            grad_outputs=torch.ones_like(u_s),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        f = u_t + 0.5 * sigma ** 2 * S ** 2 * u_ss + r * S * u_s - r * u
        return f

    def train(self):
        self.model.train()
        epoch = 0
        counter = 0
        prev_loss = 0
        while counter < 5:
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                u_pred = self.net_u(inputs)
                f_pred = self.net_f(inputs)

                loss = torch.mean((targets - u_pred) ** 2) + torch.mean(f_pred ** 2)

                self.optimizer.zero_grad()
                loss.backward(allow_unused=True)
                self.optimizer.step()

                running_loss += loss.item()

            average_train_loss = running_loss / len(self.train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs.to(self.device)
                    targets.to(self.device)
                    u_pred = self.net_u(inputs)
                    f_pred = self.net_f(inputs)
                    loss = torch.mean((targets - u_pred) ** 2) + torch.mean(f_pred ** 2)

                    val_loss += loss.item()

            average_val_loss = val_loss / len(self.val_loader)

            print(f'Epoch [{epoch}] - Training Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}')

            if abs(prev_loss-average_train_loss < 0.1):
                counter += 1
            else:
                counter = 0
            prev_loss = average_train_loss
            epoch += 1

    def predict(self, inputs):
        inputs.to(self.device)
        self.model.eval()
        u = self.net_u(inputs)
        f = self.net_f(inputs)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

#%%
