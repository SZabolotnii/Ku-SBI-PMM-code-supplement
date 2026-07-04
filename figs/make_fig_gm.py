"""Figure: closed-form captured-Fisher efficiency g_m(alpha) for a Gaussian vs a
heavy-tailed (Student-t) scale family.  Reproduces Prop. (closed-form g_m); seed 2026."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(2026)
def p2(a): return 0.5 + 0.5*a + 1.0*a**2                    # PATP exponent p_2(alpha)

def gm_curve(u, S, alphas):                                 # closed form g_m = v^2/(Sig I)
    I = float((S*S).mean()); out = []
    for a in alphas:
        q = p2(a); M = lambda k: float((np.abs(u)**k).mean())
        v = 0.5*q*M(q); Sig = M(2*q) - M(q)**2
        out.append(v*v/(Sig*I))
    return np.array(out)

alphas = np.linspace(0, 1, 41)
ug = RNG.standard_normal(6_000_000)
Sg = 0.5*(ug**2 - 1.0)
nu = 8; ut = RNG.standard_t(nu, 6_000_000)/np.sqrt(nu/(nu-2.0))
c2 = (nu-2.0)/nu; uu = ut**2/c2
St = -0.5 + (nu+1.0)*uu/(2.0*(nu+uu))

gG, gT = gm_curve(ug, Sg, alphas), gm_curve(ut, St, alphas)
aG, aT = alphas[gG.argmax()], alphas[gT.argmax()]

plt.figure(figsize=(4.4, 3.1))
plt.plot(alphas, gG, '-',  color='C0', lw=2, label='Gaussian')
plt.plot(alphas, gT, '--', color='C3', lw=2, label=r'Student-$t_8$ (heavy tail)')
plt.scatter([aG], [gG.max()], color='C0', zorder=5)
plt.scatter([aT], [gT.max()], color='C3', zorder=5)
plt.annotate(r'$\alpha^\star=%.2f$' % aG, (aG, gG.max()), textcoords='offset points',
             xytext=(-6, -18), color='C0', ha='right', fontsize=9)
plt.annotate(r'$\alpha^\star\approx%.2f$' % aT, (aT, gT.max()), textcoords='offset points',
             xytext=(-6, 8), color='C3', ha='right', fontsize=9)
plt.xlabel(r'transition parameter $\alpha$  ($0$: fractional $\to$ $1$: integer)')
plt.ylabel(r'captured Fisher $g_m(\alpha)$')
plt.ylim(0.55, 1.02); plt.xlim(0, 1)
plt.legend(loc='lower right', fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig("gm_alpha.pdf")
print(f"saved gm_alpha.pdf  (Gaussian alpha*={aG:.2f}, t8 alpha*={aT:.2f})")
