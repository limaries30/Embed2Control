import torch

torch.set_default_dtype(torch.float64)

class NormalDistribution:
    def __init__(self, mean, logvar, v=None, r=None, A=None):
        """
        :param mean: mu in the paper
        :param logvar: \Sigma in the paper
        :param v:
        :param r:
        if A is not None then covariance matrix = A \Sigma A^T, where A = I + v r^T
        else the covariance matrix is simply diag(logvar.exp())
        """
        self.mean = mean
        self.logvar = logvar
        self.v = v
        self.r = r

        sigma = torch.diag_embed(torch.exp(logvar))
        if A is None:
            self.cov = sigma
        else:
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))


    @staticmethod
    def KL_divergence(q_z_next_pred, q_z_next):
        """
        :param q_z_next_pred: q(z_{t+1} | z_bar_t, q_z_t, u_t) using the transition
        :param q_z_next: q(z_t+1 | x_t+1) using the encoder
        :return: KL divergence between two distributions
        """
        eps = 1e-8
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = torch.exp(q_z_next_pred.logvar)+eps
        sigma_1 = torch.exp(q_z_next.logvar)+eps
        v = q_z_next_pred.v
        r = q_z_next_pred.r
        k = float(q_z_next_pred.mean.size(1))
        func_sum = lambda x: torch.sum(x, dim=1)

        # print('sigma_0',sigma_0.size())
        # print('sigma_0*v*r',(sigma_0*v*r).size())
        # print("r.pow(2) * sigma_0",(r.pow(2) * sigma_0).size())
        # print('q_z_next.logvar - q_z_next_pred.logvar',(q_z_next.logvar - q_z_next_pred.logvar).size())
        # print("torch.pow(mu_1-mu_0, 2)",(torch.pow(mu_1-mu_0, 2)/sigma_1).size())
        # print('torch.log(eps+1 + func_sum(v*r))',(torch.log(eps+1 + func_sum(v*r))).size())
        # print('k',k)

        a = func_sum((sigma_0 + 2*sigma_0*v*r) / sigma_1)+ func_sum(r.pow(2) * sigma_0) * func_sum(v.pow(2) / sigma_1)
        b = func_sum(torch.pow(mu_1-mu_0, 2) / sigma_1) - k
        c = 2*(func_sum(q_z_next.logvar - q_z_next_pred.logvar) - torch.log(eps+1 + func_sum(v*r)))
        # print('a',sum((sigma_0 + 2*sigma_0*v*r) / sigma_1)
        #                       + sum(r.pow(2) * sigma_0) * sum(v.pow(2) / sigma_1))
        # print('b',sum(torch.pow(mu_1-mu_0, 2) / sigma_1) - k)
        # print('c',2*(sum(q_z_next.logvar - q_z_next_pred.logvar) - torch.log(1 + sum(v*r))))

        KL = 0.5 * torch.mean(func_sum((sigma_0 + 2*sigma_0*v*r) / sigma_1)
                              + func_sum(r.pow(2) * sigma_0) * func_sum(v.pow(2) / sigma_1)
                              + func_sum(torch.pow(mu_1-mu_0, 2) / sigma_1) - k
                              + (func_sum(q_z_next.logvar - q_z_next_pred.logvar) - 2*torch.log(eps+1 + func_sum(v*r)))
                              )
        return KL