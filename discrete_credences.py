# discrete_credences.py

from os import terminal_size
import numpy as np
import sympy as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#np.random.seed(2021)
golden = (1 + 5**0.5) / 2

# Compute bucket statistics
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
    return [num_buckets, bucket_size, num_bounds, bounds, buckets, num_means, bucket_means, gamma]

#P_f(H) = P_i(E|H)•P_i(H)/P_i(E)
def simple_condition(like_pre, prior_pre, norm_pre):
    if norm_pre == 0.0: 
        post = prior_pre
    else:
        post = like_pre * prior_pre / norm_pre
    if post > 1.0: post = 1.0
    return post

#P_f(H) = P_i(H|E)P_f(E)+P_i(H|~E)P_f(~E)
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

def lockedown(credences, lockean_threshold=golden**-1):
    beliefs = np.where(credences > lockean_threshold, 1, 0)
    return beliefs

def simulate(discretise=True, buckets=11, trials=20, quiet=False):
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
            if quiet == False:
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
        anal_posts = np.empty(trials)
        for i in range(trials):
            minimum = min(priors[i], norms[i])
            valid_conj = [d for d in discs if d <= minimum]
            conj_prob = np.random.choice(valid_conj)
            if priors[i] == 0.0: likelihoods[i] = 0.0
            else: 
                anal_like = conj_prob / priors[i]
                closest_like_index = np.argmin(np.abs(discs - anal_like))
                likelihoods[i] = discs[closest_like_index]
        bad_counter = 0
        for i in range(trials):
            anal_post = simple_condition(likelihoods[i], priors[i], norms[i])
            anal_posts[i] = anal_post
            if np.isnan(anal_post):
                posteriors[i] = float('nan')
            else:
                closest_post_index = np.argmin(np.abs(discs - anal_post))
                posteriors[i] = discs[closest_post_index]
            if quiet == False:
                print('likelihood:', "{:.2f}".format(likelihoods[i]), '\n',
                        'prior:', "{:.2f}".format(priors[i]), '\n',
                        'norm:', "{:.2f}".format(norms[i]), '\n',
                        'posterior:', "{:.2f}".format(posteriors[i]), '\n',
                        'analytic posterior: ', anal_posts[i])
                if posteriors[i] not in discs and np.isnan(posteriors[i])==False:
                    print('*********** OUT OF BOUNDS POSTERIOR *********** \n')
                else: print('\n')
        return priors, norms, likelihoods, posteriors, anal_posts

# TAKES IN c(X) AND A NUMBER OF BUCKETS AND RETURNS b(X)
def bucket_id(N, credence, quiet=False):
    i = 0
    while credence >= i / N: 
        i += 1
    bucket_id = i - 1
    model_info = dox_states(N, quiet=True)
    buckets = model_info[4]
    if quiet == False:
        print('bucket_id: ', bucket_id)
        print('bucket range: ', buckets[bucket_id])
    return bucket_id, buckets[bucket_id]

def symmetry_test(N, credence, quiet=False):
    model_info = dox_states(N, quiet=True)
    buckets = model_info[4]
    pos_id, pos_range = bucket_id(N, credence, quiet=True)
    neg_id, neg_range = bucket_id(N, 1 - credence, quiet=True)
    if pos_id == N - 1 - neg_id: symmetric = True
    else: symmetric = False
    if quiet == False:
        print('N: ', N)
        print('X is in bucket {}, whose range is {}'.format(pos_id, pos_range))
        print('~ X is in bucket {}, whose range is {}'.format(neg_id, neg_range))
        if symmetric == True: print('PASS: X and ~X ARE symmetrically bucketed.')
        else: print('FAIL: X and ~X are NOT symmetrically bucketed.')
        print()
    return pos_id, neg_id, symmetric

############################################################################
########################### SAVED VISUALIZATIONS ###########################
############################################################################

#PLOT NUMBER OF BUCKETS VS BUCKET SIZE
record = []
print('RECORD', record)
for j in range(1,100):
    record.append(dox_states(j, True))
num_buckets_vec = [record[:][i][0] for i in range(len(record))]
bucket_size_vec = [record[:][i][1] for i in range(len(record))]
print(num_buckets_vec)
print(bucket_size_vec)
plt.plot(num_buckets_vec, bucket_size_vec)
plt.title('Relationship between number of buckets and bucket width')
plt.xlabel('Number of buckets')
plt.ylabel('Bucket width')
plt.show()

#TEST RELATIONSHIP BETWEEN NUMBER OF BUCKETS AND MSEs, ASSUMING LOW PRECISION INITIAL CREDENCES
test_buckets = 100 # number of bucket sizes to test
trials_per_test = 1000
mean_squared_errors = np.zeros(test_buckets)
for bux in range(test_buckets):
    print('________________________________________________________________')
    print('Number of buckets: ', bux)
    _, _, _, cred_post, anal_post = simulate(discretise=True, buckets=bux+1, trials=trials_per_test, quiet=True)
    squared_errors = (cred_post - anal_post)**2
    mean_squared_error = np.nanmean(squared_errors)
    print('MSE: ', mean_squared_error)
    mean_squared_errors[bux] = mean_squared_error
plt.plot(np.arange(test_buckets), mean_squared_errors)
plt.title('Single-update MSE between bucketed and analytic credences for different N \n (With bucketed initial credences)')
plt.xlabel('Number of buckets')
plt.ylabel('Mean squared error (MSE)')
plt.show()

# AVERAGE ABSOLUTE UPDATES FOR RANDOM EVIDENCE VS NUMBER OF BUCKETS FOR LOW AND HIGH PRECISION CREDENCES (Remember high-precision line is not a function of number of buckets)
mean_abs_updates_disc = np.zeros(100)
mean_abs_updates_cont = np.zeros(100)
for j in range(100):
    cred_pre_disc, _, _, cred_post_disc, _ = simulate(discretise=True, buckets=j+1, trials=1000)
    cred_pre_cont, _, _, cred_post_cont = simulate(discretise=False, buckets=j+1, trials=1000)
    updates_disc = cred_post_disc - cred_pre_disc
    updates_cont = cred_post_cont - cred_pre_cont
    print('Low-precision updates: ', cred_post_disc - cred_pre_disc)
    print('High-precision updates: ', cred_post_cont - cred_pre_cont)
    abs_updates_disc = np.abs(cred_post_disc - cred_pre_disc)
    abs_updates_cont = np.abs(cred_post_cont - cred_pre_cont)
    print('Absolute low-precision updates: ', abs_updates_disc)
    print('Absolute high-precision updates: ', abs_updates_cont)
    mean_abs_update_disc = np.mean(abs_updates_disc)
    mean_abs_update_cont = np.mean(abs_updates_cont)
    print('Mean absolute low-precision update: ', mean_abs_update_disc)
    print('Mean absolute high-precision update: ', mean_abs_update_cont)
    mean_abs_updates_disc[j] = mean_abs_update_disc
    mean_abs_updates_cont[j] = mean_abs_update_cont
plt.plot(np.arange(1,101), mean_abs_updates_disc, label='bucketed credences')
plt.plot(np.arange(1,101), mean_abs_updates_cont, label='analytic credences')
plt.legend()
plt.title('Average absolute update based on random evidence for different N')
plt.show()