# discrete_credences.py

from os import terminal_size
import numpy as np
#np.random.seed(2021)
golden = (1 + 5**0.5) / 2

def conditionalise(like_pre, prior_pre, norm_pre):
    if norm_pre == 0.0: 
        post = float("nan")
    else:
        post = like_pre * prior_pre / norm_pre
    return post

def simulate(discretise=True, buckets=11, trials=20):
    if discretise == False:
        priors = np.random.rand(trials)
        norms = np.random.rand(trials)
        likelihoods = np.empty(len(priors))
        for i in range(len(likelihoods)):
            conj_prob = np.random.uniform(0, min(priors[i], norms[i])) # conjunction probability P(E&X)
            likelihoods[i] = conj_prob / priors[i]
        for trial in range(trials):
            posterior = conditionalise(likelihoods[trial], priors[trial], norms[trial])
            print('likelihood:', likelihoods[trial], '\n',
                    'prior:', priors[trial], '\n',
                    'norm:', norms[trial], '\n',
                    'posterior:', posterior, '\n')
    if discretise == True:
        discs = np.linspace(0, 1, buckets)
        priors = np.random.choice(discs, trials)
        norms = np.random.choice(discs, trials)
        likelihoods = np.empty(len(priors))
        for i in range(len(likelihoods)):
            minimum = min(priors[i], norms[i])
            valid_probs = [d for d in discs if d <= minimum]
            conj_prob = np.random.choice(valid_probs)
            if priors[i] == 0.0: likelihoods[i] = 0.0
            else: 
                anal_like = conj_prob / priors[i]
                closest_like_index = np.argmin(np.abs(discs - anal_like))
                likelihoods[i] = discs[closest_like_index]
        bad_counter = 0
        for trial in range(trials):
            anal_post = conditionalise(likelihoods[trial], priors[trial], norms[trial])
            if np.isnan(anal_post):
                posterior = float('nan')
            else:
                closest_post_index = np.argmin(np.abs(discs - anal_post))
                posterior = discs[closest_post_index]
            print('likelihood:', "{:.2f}".format(likelihoods[trial]), '\n',
                    'prior:', "{:.2f}".format(priors[trial]), '\n',
                    'norm:', "{:.2f}".format(norms[trial]), '\n',
                    'posterior:', "{:.2f}".format(posterior))
            if posterior not in discs and np.isnan(posterior)==False:
                print('*********** OUT OF BOUNDS POSTERIOR *********** \n')
            else: print('\n')

simulate(discretise=True, buckets=101, trials=20)

def lockedown(lockean_threshold=golden**-1):
    pass

'''
TODO
- Vectorise everything (think about things I can do with these vectors, and about what they mean)
- Be able to visualise everything
- Return priors, likelihoods, norms, posteriors and updates.
- Define lockedown function which takes a vector of posteriors and converts it into a binary full belief vector with some threshold.
- 
'''