"""Logit multinivel com intercepto aleatorio (GLMM), estimado por maxima
verossimilhanca com quadratura de Gauss-Hermite adaptativa (AGQ)."""
import numpy as np
from scipy import optimize, special
from numpy.polynomial.hermite import hermgauss

def _loglik_obs_u(eta, y, u):
    # eta: (n,), u: (n,) or (n,Q) broadcast; returns log f(y|eta+u)
    z = eta[:, None] + u if u.ndim == 2 else eta + u
    yy = y[:, None] if u.ndim == 2 else y
    return yy * z - np.logaddexp(0.0, z)

class GLMMLogitAGQ:
    def __init__(self, y, X, groups, nq=15):
        self.y = np.asarray(y, float); self.X = np.asarray(X, float)
        self.g, self.gidx = np.unique(groups, return_inverse=True)
        self.G = len(self.g); self.nq = nq
        self.t, self.w = hermgauss(nq)
    def _modes(self, eta, sigma, iters=200, tol=1e-12):
        """Modo do integrando por grupo (Newton amortecido; funcao estritamente concava)."""
        u = np.zeros(self.G)
        for _ in range(iters):
            z = eta + u[self.gidx]
            p = special.expit(z)
            grad = np.bincount(self.gidx, self.y - p, self.G) - u / sigma**2
            hess = -np.bincount(self.gidx, p * (1 - p), self.G) - 1 / sigma**2
            step = np.clip(grad / hess, -2.0, 2.0)
            u = u - step
            if np.max(np.abs(step)) < tol:
                break
        z = eta + u[self.gidx]; p = special.expit(z)
        curv = np.bincount(self.gidx, p * (1 - p), self.G) + 1 / sigma**2
        return u, 1 / np.sqrt(curv)
    def loglik(self, theta):
        beta, lsig = theta[:-1], theta[-1]
        sigma = np.exp(lsig)
        eta = self.X @ beta
        uhat, tau = self._modes(eta, sigma)
        nodes = uhat[:, None] + np.sqrt(2) * tau[:, None] * self.t[None, :]   # (G,Q)
        U = nodes[self.gidx]                                                    # (n,Q)
        lf = _loglik_obs_u(eta, self.y, U)                                       # (n,Q)
        S = np.zeros((self.G, self.nq))
        np.add.at(S, self.gidx, lf)
        logprior = -0.5 * np.log(2 * np.pi * sigma**2) - nodes**2 / (2 * sigma**2)
        logint = S + logprior + self.t[None, :]**2 + np.log(self.w)[None, :]
        LLg = np.log(np.sqrt(2) * tau) + special.logsumexp(logint, axis=1)
        return float(np.sum(LLg))
    def fit(self, start=None):
        k = self.X.shape[1]
        if start is None:
            start = np.r_[np.zeros(k), np.log(1.0)]
        f = lambda th: -self.loglik(th)
        r = optimize.minimize(f, start, method="BFGS", jac="3-point", options={"maxiter": 2000, "gtol": 1e-5})
        if not r.success:
            r2 = optimize.minimize(f, r.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-8})
            r3 = optimize.minimize(f, r2.x, method="BFGS", jac="3-point", options={"maxiter": 2000, "gtol": 1e-5})
            r = min([r, r2, r3], key=lambda z: z.fun)
        self.theta = r.x; self.llf = -r.fun; self.opt = r
        H = self._num_hess(r.x)
        self.cov = np.linalg.inv(H)
        self.se = np.sqrt(np.diag(self.cov))
        return self
    def _num_hess(self, th, h=1e-4):
        k = len(th); H = np.zeros((k, k)); f = lambda x: -self.loglik(x)
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ej = np.zeros(k); ei[i] = h; ej[j] = h
                v = (f(th+ei+ej) - f(th+ei-ej) - f(th-ei+ej) + f(th-ei-ej)) / (4*h*h)
                H[i, j] = H[j, i] = v
        return H

if __name__ == "__main__":
    # validacao em dados simulados
    rng = np.random.default_rng(1)
    G, m = 200, 100
    g = np.repeat(np.arange(G), m); n = G*m
    x1 = rng.normal(size=n); x2 = rng.binomial(1, 0.3, n)
    u = rng.normal(0, 1.8, G)[g]
    eta = -0.5 + 0.8*x1 - 1.0*x2 + u
    y = rng.binomial(1, 1/(1+np.exp(-eta)))
    X = np.c_[np.ones(n), x1, x2]
    for q in (7, 15, 25):
        mod = GLMMLogitAGQ(y, X, g, nq=q).fit()
        print(q, np.round(mod.theta[:-1], 4), "sigma", round(float(np.exp(mod.theta[-1])),4), "ll", round(mod.llf,4), "se", np.round(mod.se,4))
