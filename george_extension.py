import george
import numpy as np
import matplotlib.pyplot as plt
import emcee



ndims = 5		# total number of dimensions
which_dimension=3
degree=2		# if the degree is to low, it gets numerically unstable as many configurations have essentially zero probability

sigma_noise = 0.1


kernel = george.kernels.BayesianLinearRegressionKernel(ndims,which_dimension,degree)
gp = george.GP(kernel)



def f(x):
	return(x**3 - 20*x**2 + 10*x+0.2)



# just to check that you can form product and sum kernels
kernel2 = george.kernels.ExpSquaredKernel(0.5,ndims) * kernel
kernel3 = george.kernels.ExpSquaredKernel(0.5,ndims) + kernel

num_points = 10
X = np.array([np.sort(np.random.rand(num_points)*0.5) for d in range(ndims)]).T
Y = np.sort(np.random.rand(num_points))[::-1]
Y = f(X[:,which_dimension]) + sigma_noise**2*np.random.randn(num_points)

# kernel parameter live on a log scale
gp.kernel[:] = np.array([0.1]*(degree+1))
# initialize kernel computation
gp.compute(X, sigma_noise)


plt.scatter(X[:,which_dimension],Y)
plt.show(block=False)



def lnprob(p):
	# Update the kernel and compute the lnlikelihood.
	if np.any(p>10) or np.any (p<-10):
		return(-np.inf)
	logprior = 0
	gp.kernel.pars = np.exp(p)
	lh = logprior + gp.lnlikelihood(Y, quiet=True)
	print(p, lh)
	return(lh)


# Set up the sampler.
nwalkers = 2*(degree+1)
sampler = emcee.EnsembleSampler(nwalkers, degree+1, lnprob)


# if you want to do the kernel with (1, (1-x)^2) as basis functions,
# set degree = 2 and initialize the walkers such that the parameters obey
# gp.kernel[1] = -2 gp.kernel[2]
p0 = [5*np.random.rand(degree+1)-5 for i in range(nwalkers)]



p0, _, _ = sampler.run_mcmc(p0, 200)
chain, _, _=sampler.run_mcmc(p0, 50)


t = np.array([np.linspace(0,1,20) for d in range(ndims)]).T
mu, cov = gp.predict(Y, t)
std = np.sqrt(np.diag(cov))


plt.fill_between(t[:,which_dimension], mu-std, mu+std, alpha=0.3)
plt.scatter(X[:,which_dimension],Y)
plt.plot(t[:,which_dimension],mu)


for confs in sampler.chain[:,::10]:
	for conf in confs:
		gp.kernel.pars = np.exp(conf)
		gp.compute(X,sigma_noise)
		mu, cov = gp.predict(Y, t)
		plt.plot(t[:,which_dimension],mu, alpha=0.3)

plt.plot(t[:,which_dimension], f(t[:,which_dimension]), color='red')

plt.show()
