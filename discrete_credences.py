# discrete_credences.py

from os import terminal_size
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#np.random.seed(2021)
golden = (1 + 5**0.5) / 2

def conditionalise(like_pre, prior_pre, norm_pre):
    if norm_pre == 0.0: 
        post = float("nan")
    else:
        post = like_pre * prior_pre / norm_pre
    return post

def lockedown(credences, lockean_threshold=golden**-1):
    beliefs = np.where(credences > lockean_threshold, 1, 0)
    return beliefs

def simulate(discretise=True, buckets=11, trials=20):
    if discretise == False:
        priors = np.random.rand(trials)
        norms = np.random.rand(trials)
        likelihoods = np.empty(trials)
        posteriors = np.empty(trials)
        for i in range(trials):
            conj_prob = np.random.uniform(0, min(priors[i], norms[i])) # conjunction probability P(E&X)
            likelihoods[i] = conj_prob / priors[i]
        for i in range(trials):
            posteriors[i] = conditionalise(likelihoods[i], priors[i], norms[i])
            print('likelihood:', likelihoods[i], '\n',
                    'prior:', priors[i], '\n',
                    'norm:', norms[i], '\n',
                    'posterior:', posteriors[i], '\n')
        return priors, norms, likelihoods, posteriors
    if discretise == True:
        discs = np.linspace(0, 1, buckets)
        priors = np.random.choice(discs, trials)
        norms = np.random.choice(discs, trials)
        likelihoods = np.empty(trials)
        posteriors = np.empty(trials)
        for i in range(trials):
            minimum = min(priors[i], norms[i])
            valid_probs = [d for d in discs if d <= minimum]
            conj_prob = np.random.choice(valid_probs)
            if priors[i] == 0.0: likelihoods[i] = 0.0
            else: 
                anal_like = conj_prob / priors[i]
                closest_like_index = np.argmin(np.abs(discs - anal_like))
                likelihoods[i] = discs[closest_like_index]
        bad_counter = 0
        for i in range(trials):
            anal_post = conditionalise(likelihoods[i], priors[i], norms[i])
            if np.isnan(anal_post):
                posteriors[i] = float('nan')
            else:
                closest_post_index = np.argmin(np.abs(discs - anal_post))
                posteriors[i] = discs[closest_post_index]
            print('likelihood:', "{:.2f}".format(likelihoods[i]), '\n',
                    'prior:', "{:.2f}".format(priors[i]), '\n',
                    'norm:', "{:.2f}".format(norms[i]), '\n',
                    'posterior:', "{:.2f}".format(posteriors[i]))
            if posteriors[i] not in discs and np.isnan(posteriors[i])==False:
                print('*********** OUT OF BOUNDS POSTERIOR *********** \n')
            else: print('\n')
        return priors, norms, likelihoods, posteriors

cred_pre, norm_pre, like_pre, cred_post = simulate(discretise=True, buckets=101, trials=20)
updates = cred_post - cred_pre

print('Credences before updating: ', cred_pre)
print('Credences after updating: ', cred_post)
print('Updates: ', cred_post - cred_pre)

beliefs_pre = lockedown(cred_pre)
beliefs_post = lockedown(cred_post)
print(beliefs_pre)
print(beliefs_post)

'''
TODO
- Be able to visualise everything.
- Add capability to do filtering over multiple 'steps', so that each trial involves conditioning multiple times.
- How can I study the behaviour of the DC model?
'''
