# discrete_credences.py

from os import terminal_size
import numpy as np
import sympy as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#np.random.seed(2021)
golden = (1 + 5**0.5) / 2

def dox_states(num_buckets, quiet=False):
    assert num_buckets > 0, 'Must have at least 1 bucket!'
    bucket_size = 1 / num_buckets
    bounds = [i * bucket_size for i in range(num_buckets)]
    bounds.append(1.0)   
    buckets = []
    for i in range(num_buckets):
        buckets.append([bounds[i], bounds[i+1]])
    num_means = num_buckets
    bucket_means = []
    for i in range(num_buckets):
        bucket_means.append(np.mean(buckets[i]))
    num_bounds = num_buckets + 1
    gamma = bucket_size / 2
    buckets_for_printing = [['{:.3f}'.format(elem) for elem in buckets[i]] for i in range(len(buckets))]
    if quiet==False:
        print('___________________________________________')    
        print('(All to 3 decimal places...)')
        print('Number of buckets: ', num_buckets)
        print('Bucket size/width: ', format(bucket_size, '.3f'))
        print('Number of bucket boundaries: ', num_bounds)
        print('Bucket boundaries: ', ['{:.3f}'.format(elem) for elem in bounds])
        print('Buckets: ', buckets_for_printing)
        print('Number of bucket means: ', num_means)
        print('Bucket means: ', ['{:.3f}'.format(elem) for elem in bucket_means])
        print('Gamma: ', format(gamma, '.3f'))
    else:
        print('Quieted...')
    return [num_buckets, bucket_size, num_bounds, bounds, buckets, num_means, bucket_means, gamma]

#P_f(H) = P_i(E|H)•P_i(H)/P_i(E)
def simple_condition(like_pre, prior_pre, norm_pre):
    if norm_pre == 0.0: 
        post = float("nan")
    else:
        post = like_pre * prior_pre / norm_pre
    return post

#P_f(H) = P_i(H|E)P_f(E)+P_i(H|~E)P_f(~E)
#       = 
#post   = simple_condition(evid)•evid_prob + simple_condition(nevid)•nevid_prob
def jeffrey_condition(like_pres, prior_pre, norm_pres, evid_probs):
    # all plural arguments should be lists of the same length (2 in the simplest case, E or ~E), 
    #   but there is only one prior P_i(H). But the others depend on possible values of E: 
    #   likelihoods are P_i(E=e|H), norms are P_i(E=e), (new) evid_probs are P_f(E=e)
    simple_conds = []
    for i in range(len(like_pres)):
        simple_conds.append(simple_condition(like_pres[i], prior_pre, norm_pres[i]))
    jeff_post = sum([a * b for a, b in zip(simple_conds, evid_probs)])
    return jeff_post

#testing jeffrey_condition():
#

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
            posteriors[i] = simple_condition(likelihoods[i], priors[i], norms[i])
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
            anal_post = simple_condition(likelihoods[i], priors[i], norms[i])
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

'''
cred_pre, norm_pre, like_pre, cred_post = simulate(discretise=True, buckets=101, trials=5)
updates = cred_post - cred_pre
print('Credences before updating: ', cred_pre)
print('\n')
print('Credences after updating: ', cred_post)
print('\n')
print('Updates: ', cred_post - cred_pre)
'''

'''
# simple filtering example with 3 updates INCOMPLETE
cred_pre, norm_pre, like_pre, cred_post = simulate(discretise=True, buckets=11, trials=1)
for i in range(5):
    cred_pre = cred_post
    cred_post = simple_condition(like_pre, cred_pre, norm_pre)
    print(cred_post)

def sync_log():
    pass

def sync_prob():
    pass
'''


record = []
print('RECORD', record)
for j in range(1,100):
    record.append(dox_states(j, True))
num_buckets_vec = [record[:][i][0] for i in range(len(record))]
bucket_size_vec = [record[:][i][1] for i in range(len(record))]
print(num_buckets_vec)
print(bucket_size_vec)

plt.plot(num_buckets_vec, bucket_size_vec)
plt.xlabel('Number of buckets')
plt.ylabel('Bucket size')
plt.show()

# conditionalise with buckets
    # compute difference between true credence and discretised credence
        # using bucket decoding (mean of bucket)
        # using distance of analytic credence to closest boundary (how much credence would have to differ to result in different disc)
        # using bucket-disagreement over local hyperparams