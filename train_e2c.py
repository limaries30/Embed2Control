from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys
from easydict import EasyDict as edict

from normal import *
from e2c_model import E2C
from datasets import *
import data.sample_planar as planar_sampler
import data.sample_pendulum_data as pendulum_sampler
import data.sample_cartpole_data as cartpole_sampler
import yaml
import os


torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 10 # number of images evaluated on tensorboard

# dataset = datasets['planar']('./data/data/' + 'planar')
# x, u, x_next = dataset[0]
# imgplot = plt.imshow(x.squeeze(), cmap='gray')
# plt.show()
# print (np.array(u, dtype=float))
# imgplot = plt.imshow(x_next.squeeze(), cmap='gray')
# plt.show()


def compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lamda):
    # lower-bound loss
    w, h = x.size()
    recon_term = -torch.mean(torch.sum(x * torch.log(1e-8 + x_recon)
                                      + (1 - x) * torch.log(1e-8 + 1 - x_recon), dim=1))
    # torch.mean( 0.5 * (x.view(w*h, 1) - x_recon.view(w*h, 1)) ** 2 ,dim=0)

    pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-8+ x_next_pred)
                                      + (1 - x_next) * torch.log(1e-8 + 1 - x_next_pred), dim=1))
    # torch.mean (0.5 * (x_next.view(w*h, 1) - x_next_pred.view(w*h, 1)) ** 2 ,dim=0)#

    kl_term = - 0.5 * torch.mean(torch.sum(1 + q_z.logvar - q_z.mean.pow(2) - torch.exp(q_z.logvar), dim=1))

    lower_bound = recon_term + pred_loss + kl_term

    # consistency loss
    consis_term = NormalDistribution.KL_divergence(q_z_next_pred, q_z_next)

    return lower_bound, lamda * consis_term


def train(model, train_loader, lam, optimizer):
    model.train()
    avg_loss = 0.0
    avg_lower_bound_loss = 0.0
    avg_const_loss = 0.0

    num_batches = len(train_loader)
    for i, (x, u, x_next) in enumerate(train_loader, 0):
        x = x.view(-1, model.obs_dim).double().to(device)

        u = u.double().to(device)
        x_next = x_next.view(-1, model.obs_dim).double().to(device)

        optimizer.zero_grad()

        x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)

        lower_bound, const_loss = compute_loss(x, x_next, q_z_next, x_recon, x_next_pred, q_z, q_z_next_pred, lam)
        loss = lower_bound + const_loss
        avg_loss += (lower_bound+const_loss).item()
        avg_lower_bound_loss += lower_bound.item()
        avg_const_loss += const_loss.item()
        loss.backward()
        optimizer.step()

    return avg_loss / num_batches, avg_lower_bound_loss/num_batches, avg_const_loss/num_batches


def compute_log_likelihood(x, x_recon, x_next, x_next_pred):
    w, h = x.size()
    recon_term = -torch.mean(torch.sum(x * torch.log(1e-8 + x_recon)
                                      + (1 - x) * torch.log(1e-8 + 1 - x_recon), dim=1))
    # torch.mean(0.5 * (x.view(w * h, 1) - x_recon.view(w * h, 1)) ** 2, dim=0)
    pred_loss = -torch.mean(torch.sum(x_next * torch.log(1e-8+ x_next_pred)
                                      + (1 - x_next) * torch.log(1e-8 + 1 - x_next_pred), dim=1))
    # torch.mean(0.5 * (x_next.view(w*h,1) - x_next_pred.view(w*h,1)) ** 2,dim=0)
    # -torch.mean(torch.sum(x_next * torch.log(1e-8+ x_next_pred) + (1 - x_next) * torch.log(1e-8 + 1 - x_next_pred), dim=1))
    return recon_term, pred_loss


def evaluate(model, test_loader):
    model.eval()
    num_batches = len(test_loader)
    state_loss, next_state_loss = 0., 0.
    with torch.no_grad():
        for x, u, x_next in test_loader:
            x = x.view(-1, model.obs_dim).double().to(device)
            u = u.double().to(device)
            x_next = x_next.view(-1, model.obs_dim).double().to(device)

            x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next = model(x, u, x_next)
            loss_1, loss_2 = compute_log_likelihood(x, x_recon, x_next, x_next_pred)
            state_loss += loss_1
            next_state_loss += loss_2

    return state_loss.item() / num_batches, next_state_loss.item() / num_batches


# code for visualizing the training process
def predict_x_next(model, env, num_eval):
    # first sample a true trajectory from the environment
    sampler = samplers[env]
    sampled_data = sampler.sample(num_eval)

    # use the trained model to predict the next observation
    predicted = []
    for x, u, x_next in sampled_data:
        x_reshaped = x.reshape(-1)
        x_reshaped = torch.from_numpy(x_reshaped).double().unsqueeze(dim=0).to(device)
        u = torch.from_numpy(u).double().unsqueeze(dim=0).to(device)
        with torch.no_grad():
            x_next_pred = model.predict(x_reshaped, u)
        predicted.append(x_next_pred.squeeze().cpu().numpy().reshape(sampler.height, sampler.width))
    true_x_next = [data[-1] for data in sampled_data]
    return true_x_next, predicted


def plot_preds(model, env, num_eval):
    true_x_next, pred_x_next = predict_x_next(model, env, num_eval)

    # plot the predicted and true observations
    fig, axes = plt.subplots(nrows=2, ncols=num_eval)
    plt.setp(axes, xticks=[], yticks=[])
    pad = 5
    axes[0, 0].annotate('True observations', xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
                   xycoords=axes[0, 0].yaxis.label, textcoords='offset points',
                   size='large', ha='right', va='center')
    axes[1, 0].annotate('Predicted observations', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    for idx in np.arange(num_eval):
        axes[0, idx].imshow(true_x_next[idx], cmap='Greys')
        axes[1, idx].imshow(pred_x_next[idx], cmap='Greys')
    fig.tight_layout()
    return fig


def main(args):
    env_name = args.env                                 # "pendulum"
    assert env_name in ['planar', 'pendulum']
    propor = float(args.propor)                         # proportion of training data = 0.75

    batch_size = args.batch_size                        # 128
    lr = args.lr
    weight_decay = args.decay
    lam = args.lam                                      # lambda, coefficient of KL Divergence
    epoches = args.num_iter                             # 1000
    iter_save = args.iter_save                          # 100
    parent_dir = args.parent_dir                        # 'C:\Users\kihong\Documents\E2C_kihong'
    log_dir = os.path.join(parent_dir, args.log_dir)
    data_dir = os.path.join(parent_dir, 'data')
    model_path = os.path.join(parent_dir, 'model')

    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = datasets[env_name](os.path.join(data_dir, env_name))
    print('number of data = ', len(dataset))
    train_set, test_set = dataset[:int(len(dataset) * propor)], dataset[int(len(dataset) * propor):]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    obs_dim, z_dim, u_dim = settings[env_name]
    model = E2C(obs_dim=obs_dim, z_dim=z_dim, u_dim=u_dim, env=env_name).to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter(log_dir)

    result_path = os.path.join(model_path)
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for i in range(epoches):
        avg_loss, avg_lower_loss, avg_const_loss = train(model, train_loader, lam, optimizer)
        print('Epoch %d' % i)
        print("Training loss: %f , Lower loss:%f, Const loss: %f" % (avg_loss, avg_lower_loss, avg_const_loss))
        # evaluate on test set
        state_loss, next_state_loss = evaluate(model, test_loader)
        print('State loss: ' + str(state_loss))
        print('Next state loss: ' + str(next_state_loss))

        # ...log the running loss
        writer.add_scalar('training loss', avg_loss, i)
        writer.add_scalar('avg_lower_loss', avg_lower_loss, i)
        writer.add_scalar('avg_const_loss', avg_const_loss, i)
        writer.add_scalar('state loss', state_loss, i)
        writer.add_scalar('next state loss', next_state_loss, i)

        # save model
        if (i + 1) % iter_save == 0:
            writer.add_figure('actual vs. predicted observations',
                              plot_preds(model, env_name, num_eval),
                              global_step=i)
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1) + '.pth')
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([str(state_loss), str(next_state_loss)]))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train e2c model')

    # the default value is used for the planar task
    with open('config.yaml') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))


    main(config)
