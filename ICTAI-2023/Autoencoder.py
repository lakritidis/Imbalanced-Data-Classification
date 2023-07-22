import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from sklearn.neighbors import KDTree, NearestNeighbors


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_mse + loss_kld


class VAE(nn.Module):
    def __init__(self, dimensionality=2, sampling_strategy=1.0, latent_dimensionality=3, architecture=None):
        super(VAE, self).__init__()

        self.dimensionality_ = dimensionality
        self.sampling_strategy_ = sampling_strategy
        self.latent_dimensionality_ = latent_dimensionality
        self.architecture_ = architecture

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if architecture is None:
            architecture = [12, 6]

        layer_1_neurons = architecture[0]
        layer_2_neurons = architecture[1]

        # Encoder
        self.linear1 = nn.Linear(dimensionality, layer_1_neurons)
        self.lin_bn1 = nn.BatchNorm1d(num_features=layer_1_neurons)
        self.linear2 = nn.Linear(layer_1_neurons, layer_2_neurons)
        self.lin_bn2 = nn.BatchNorm1d(num_features=layer_2_neurons)
        self.linear3 = nn.Linear(layer_2_neurons, layer_2_neurons)
        self.lin_bn3 = nn.BatchNorm1d(num_features=layer_2_neurons)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(layer_2_neurons, latent_dimensionality)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dimensionality)
        self.fc21 = nn.Linear(latent_dimensionality, latent_dimensionality)
        self.fc22 = nn.Linear(latent_dimensionality, latent_dimensionality)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dimensionality, latent_dimensionality)
        self.fc_bn3 = nn.BatchNorm1d(latent_dimensionality)
        self.fc4 = nn.Linear(latent_dimensionality, layer_2_neurons)
        self.fc_bn4 = nn.BatchNorm1d(layer_2_neurons)

        # Decoder
        self.linear4 = nn.Linear(layer_2_neurons, layer_2_neurons)
        self.lin_bn4 = nn.BatchNorm1d(num_features=layer_2_neurons)
        self.linear5 = nn.Linear(layer_2_neurons, layer_1_neurons)
        self.lin_bn5 = nn.BatchNorm1d(num_features=layer_1_neurons)
        self.linear6 = nn.Linear(layer_1_neurons, dimensionality)
        self.lin_bn6 = nn.BatchNorm1d(num_features=dimensionality)

        self.relu = nn.ReLU()

    # Encode method
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    # Re-parametrization trick
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Decode method
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Train the VAE and use it to generate synthetic samples
    def fit_resample(self, x, y):
        # Phase 1: VAE Training
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        epochs = 1500

        # Find the majority class
        num_classes = len(set(y))

        samples_per_class = np.zeros(num_classes)
        for k in range(x.shape[0]):
            samples_per_class[y[k]] = samples_per_class[y[k]]+1

        majority_class = np.array(samples_per_class).argmax()
        train_majority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == majority_class])

        # Train the VAE on the samples of each minority class and synthesize class data
        for cls in range(num_classes):
            if cls != majority_class:
                train_minority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == cls])

                # The minority samples will be used to train the VAE
                x_train_tensor = torch.from_numpy(train_minority_samples).to(torch.float32)

                # Create batches of training data and store them into a DataLoader object
                train_loader = DataLoader(dataset=x_train_tensor, batch_size=1024)

                # Training parameters
                train_losses = []

                # Train the VAE in epochs
                for epoch in range(1, epochs + 1):
                    self.vae_train(epoch, train_loader, optimizer, train_losses)

                # Phase 2: Use the trained model to generate synthetic samples
                with torch.no_grad():
                    for batch_idx, data in enumerate(train_loader):
                        data = data.to(self.device_)
                        optimizer.zero_grad()
                        recon_batch, mu, logvar = self(data)

                # Standard deviation of the learned distribution
                sigma = torch.exp(logvar / 2)

                # Set up a Gaussian distribution with the mean and standard deviations returned by the VAE
                q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))

                # How many samples to generate? (The difference between the majority and minority samples)
                no_samples = int(self.sampling_strategy_ * len(train_majority_samples) - len(train_minority_samples))
                # print('create ', no_samples, 'samples')
                min_classes = np.ones(no_samples)

                # Take random samples from the Gaussian
                z = q.rsample(sample_shape=torch.Size([no_samples]))

                # Reconstruct the Gaussian samples by feeding them to the VAE Decoder
                with torch.no_grad():
                    synthetic_samples = self.decode(z).cpu().numpy()

                x_over_train = np.vstack((x, synthetic_samples))
                y_over_train = np.hstack((y, min_classes))

        return x_over_train, y_over_train

    def vae_train(self, epoch, train_loader, optimizer, train_losses):
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device_)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = CustomLoss()(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % 200 == 0:
            # print('====> Epoch: {} Avg training loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
            train_losses.append(train_loss / len(train_loader.dataset))


# Safe-Borderline Variational Autoecoder (SB-VAE)
class SB_VAE(nn.Module):
    def __init__(self, dimensionality=2, sampling_strategy=1.0, latent_dimensionality=3, architecture=None, radius=0.1):
        super(SB_VAE, self).__init__()

        self.dimensionality_ = dimensionality
        self.sampling_strategy_ = sampling_strategy
        self.latent_dimensionality_ = latent_dimensionality
        self.architecture_ = architecture
        self.radius_ = radius
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if architecture is None:
            architecture = [12, 6]

        layer_1_neurons = architecture[0]
        layer_2_neurons = architecture[1]

        # Encoder
        self.linear1 = nn.Linear(dimensionality, layer_1_neurons)
        self.lin_bn1 = nn.BatchNorm1d(num_features=layer_1_neurons)
        self.linear2 = nn.Linear(layer_1_neurons, layer_2_neurons)
        self.lin_bn2 = nn.BatchNorm1d(num_features=layer_2_neurons)
        self.linear3 = nn.Linear(layer_2_neurons, layer_2_neurons)
        self.lin_bn3 = nn.BatchNorm1d(num_features=layer_2_neurons)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(layer_2_neurons, latent_dimensionality)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dimensionality)
        self.fc21 = nn.Linear(latent_dimensionality, latent_dimensionality)
        self.fc22 = nn.Linear(latent_dimensionality, latent_dimensionality)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dimensionality, latent_dimensionality)
        self.fc_bn3 = nn.BatchNorm1d(latent_dimensionality)
        self.fc4 = nn.Linear(latent_dimensionality, layer_2_neurons)
        self.fc_bn4 = nn.BatchNorm1d(layer_2_neurons)

        # Decoder
        self.linear4 = nn.Linear(layer_2_neurons, layer_2_neurons)
        self.lin_bn4 = nn.BatchNorm1d(num_features=layer_2_neurons)
        self.linear5 = nn.Linear(layer_2_neurons, layer_1_neurons)
        self.lin_bn5 = nn.BatchNorm1d(num_features=layer_1_neurons)
        self.linear6 = nn.Linear(layer_1_neurons, dimensionality)
        self.lin_bn6 = nn.BatchNorm1d(num_features=dimensionality)

        self.relu = nn.ReLU()

    # Encode method
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    # Re-parametrization trick
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Decode method
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def determine_training_samples(self, x, y, cls):
        # tree = KDTree(x, leaf_size=30, metric='euclidean')

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x)
        all_minority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == cls])

        distances, indices = nbrs.kneighbors(all_minority_samples)

        outliers = []
        core_points = []
        border_points_1 = []
        border_points_2 = []
        train_minority_samples = []

        for m in range(indices.shape[0]):
            # min_sample = x[m]
            # print("Checking point", m, ':', min_sample)
            points_with_same_class = 0

            for k in range(indices.shape[1]):
                nn_idx = indices[m][k]

                if y[m] == y[nn_idx]:
                    points_with_same_class = points_with_same_class + 1
                    #print("point", m, " same class", points_with_same_class)
                    if points_with_same_class > 2:
                        #print("\t\tappending point", m)
                        train_minority_samples.append(x[m])
                        break

        minority_samples = np.array(train_minority_samples)
        return minority_samples

    # Train the VAE and use it to generate synthetic samples
    def fit_resample(self, x, y):
        # Phase 1: VAE Training
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        epochs = 1500

        # Find the majority class
        num_classes = len(set(y))

        samples_per_class = np.zeros(num_classes)
        for k in range(x.shape[0]):
            samples_per_class[y[k]] = samples_per_class[y[k]]+1

        majority_class = np.array(samples_per_class).argmax()
        train_majority_samples = np.array([x[k] for k in range(x.shape[0]) if y[k] == majority_class])

        # Train the VAE on the samples of each minority class and synthesize class data
        for cls in range(num_classes):
            if cls != majority_class:
                train_minority_samples = self.determine_training_samples(x, y, cls)

                # The minority samples will be used to train the VAE
                x_train_tensor = torch.from_numpy(train_minority_samples).to(torch.float32)

                # Create batches of training data and store them into a DataLoader object
                train_loader = DataLoader(dataset=x_train_tensor, batch_size=1024)

                # Training parameters
                train_losses = []

                # Train the VAE in epochs
                for epoch in range(1, epochs + 1):
                    self.vae_train(epoch, train_loader, optimizer, train_losses)

                # Phase 2: Use the trained model to generate synthetic samples
                with torch.no_grad():
                    for batch_idx, data in enumerate(train_loader):
                        data = data.to(self.device_)
                        optimizer.zero_grad()
                        recon_batch, mu, logvar = self(data)

                # Standard deviation of the learned distribution
                sigma = torch.exp(logvar / 2)

                # Set up a Gaussian distribution with the mean and standard deviations returned by the VAE
                q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))

                # How many samples to generate? (The difference between the majority and minority samples)
                no_samples = int(self.sampling_strategy_ * len(train_majority_samples) - len(train_minority_samples))
                # print('create ', no_samples, 'samples')
                min_classes = np.ones(no_samples)

                # Take random samples from the Gaussian
                z = q.rsample(sample_shape=torch.Size([no_samples]))

                # Reconstruct the Gaussian samples by feeding them to the VAE Decoder
                with torch.no_grad():
                    synthetic_samples = self.decode(z).cpu().numpy()

                x_over_train = np.vstack((x, synthetic_samples))
                y_over_train = np.hstack((y, min_classes))

        return x_over_train, y_over_train

    def vae_train(self, epoch, train_loader, optimizer, train_losses):
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device_)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = CustomLoss()(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % 200 == 0:
            # print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
            train_losses.append(train_loss / len(train_loader.dataset))
