from scipy.stats import beta, binom
import numpy as np
import plotly.express as px
import pandas as pd

from empiricaldist import Pmf

'''
Trying to simulate n companies having certain failure rates
Then build a dataset for these companies to develop a prior -> should be normal
Finally, pull in a new company and generate some data, then see if a bayes update will work
'''
np.random.seed(100)

class Company():
    def __init__(self):
        self.alpha_param = 2
        self.beta_param = 4
        self.defect_rate = beta.rvs(self.alpha_param, self.beta_param)
    
    def sample(self, n):
        return binom.rvs(n, self.defect_rate)
    
    @classmethod
    def generate_companies(cls, n):
        return [Company() for _ in range(n)]
    

def fit_beta_from_mean_std(mean, std, scale_down_term=100):
    """Fit a Beta distribution to a given mean and standard deviation."""
    variance = std**2
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
    return alpha / scale_down_term, beta / scale_down_term

if __name__ == '__main__':
    # collect prior information, and likelihood?
    nr_companies = 10
    nr_sims = 10_000
    items = 100
    companies = Company.generate_companies(nr_companies)
    mean_defective_items = [np.mean([c.sample(items) for c in companies]) for _ in range(nr_sims)]
    
    df = pd.DataFrame(mean_defective_items, columns=['mean_nr_defects']) / 100
    fig = px.histogram(df)
    fig.write_image('./bayes/sims/histogram_of_mean_defects.png')

    mu = np.mean(df)
    sigma = np.std(df.to_numpy())
    alpha_prior, beta_prior = fit_beta_from_mean_std(mu, sigma)
    print(f'Fitted Beta prior: alpha={alpha_prior:.4f}, beta={beta_prior:.4f}')
    
    new_company = Company()
    print(f'New company defect rate {new_company.defect_rate}')
    
    
    new_company_samples = 5
    new_items = 1
    new_company_data = [new_company.sample(new_items) for _ in range(new_company_samples)]
    print('new company mean', np.mean(new_company_data) / new_items)

    x = np.linspace(0, 1, 101)
    prior = Pmf(beta.pdf(x, alpha_prior, beta_prior), x)
    
    posterior = prior.copy()
    for data in new_company_data:
        likelihood = binom.pmf(data, new_items, posterior.qs)
        posterior *= likelihood
    posterior.normalize()
    print(f'The highest probability defect rate {posterior.idxmax()}')

    df_posterior = pd.DataFrame({
    'defect_rate': posterior.qs,  # Defect rate values
    'probability': posterior.ps   # Posterior probabilities
    })

    # Plot the posterior using Plotly Express
    fig = px.line(df_posterior, x='defect_rate', y='probability', 
                title='Posterior Distribution of Defect Rate',
                labels={'defect_rate': 'Defect Rate', 'probability': 'Probability Density'})
    fig.write_image('./bayes/sims/posterior.png')

