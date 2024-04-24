import numpy as np
import matplotlib.pyplot as plt
def initialization(nP,dim,ub,lb):
    Boundary_no= np.size(ub); 
    if Boundary_no==1:
        return np.random.rand(nP,dim)*(ub-lb)+lb

    if Boundary_no>1:
        X=np.zeros([nP,dim])
        for i in range(dim):
            ub_i=ub[i]
            lb_i=lb[i]
            X[:,i]=np.random.rand(nP)*(ub_i-lb_i)+lb_i
        return X




def F1(x):
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def F2(x):
    D = len(x)
    # f = [abs(xi)**(i+1) for i, xi in enumerate(D)]
    f=np.zeros(D)
    for i in range (D):
        f[i]=abs(x[i])**(i+2)
    return np.sum(f)

def F3(x):
    return np.sum(x**2) + (np.sum(0.5 * x))**2 + (np.sum(0.5 * x))**4

def F4(x):
    D = len(x)
    ff = [100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(D-1)]
    return np.sum(ff)

def F5(x):
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)

def F6(x):
    D = len(x)
    f = [((1e6)**((i)/(D-1))) * x[i]**2 for i in range(D)]
    return np.sum(f)

def F7(x):
    D = len(x)
    f = [0.5 + (np.sin(np.sqrt(x[i]**2 + x[(i+1)%D]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[(i+1)%D]**2))**2 for i in range(D)]
    return np.sum(f)

def F8(x):
    w = [1 + (xi - 1)/4 for xi in x[:-1]]
    f = [(wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w]
    wD = 1 + (x[-1] - 1)/4
    return np.sin(np.pi * w[0])**2 + np.sum(f) + (wD - 1)**2 * (1 + np.sin(2 * np.pi * wD)**2)

def F9(x):
    D = len(x)
    f = []
    for xi in x:
        y = xi + 420.9687462275036
        if abs(y) < 500:
            f.append(y * np.sin(abs(y)**0.5))
        elif y > 500:
            f.append((500 - y % 500) * np.sin((abs(500 - y % 500))**0.5) - (y - 500)**2 / (10000 * D))
        elif y < -500:
            f.append((abs(y) % 500 - 500) * np.sin((abs(abs(y) % 500 - 500))**0.5) - (y + 500)**2 / (10000 * D))
    return 418.9829 * D - np.sum(f)

def F10(x):
    D = len(x)
    return -20 * np.exp(-0.2 * ((1/D) * np.sum(x**2))**0.5) - np.exp((1/D) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

def F11(x):
    D = len(x)
    x = x + 0.5
    a = 0.5
    b = 3
    kmax = 20
    c1 = np.array([a**(k) for k in range(kmax+1)])
    c2 = np.array([2 * np.pi * b**(k) for k in range(kmax+1)])
    c = -w(0.5, c1, c2)
    f = np.sum([w(xi, c1, c2) for xi in x])
    return f + c * D

def w(x, c1, c2):
    return np.sum(c1 * np.cos(c2 * x))

def F12(x):
    D = len(x)
    return (abs(np.sum(x**2) - D))**(1/4) + (0.5 * np.sum(x**2) + np.sum(x)) / D + 0.5

def F13(x):
    dim = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1

def Ufun(x, a, k, m):
    return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)

def F14(x):
    dim = len(x)
    part1 = np.sum((10 * np.sin(np.pi * (1 + (x[:-1] + 1) / 4))**2))
    part2 = np.sum((((x[:-1] + 1) / 4)**2) * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4))**2)))
    part3 = ((x[-1] + 1) / 4)**2
    return (np.pi / dim) * (part1 + part2) + part3 + np.sum(Ufun(x, 10, 100, 4))




def BenchmarkFunctions(F):
    D = 30
    if F == 'F1':
        fobj = F1
        lb = -100
        ub = 100
        dim = D
    elif F == 'F2':
        fobj = F2
        lb = -100
        ub = 100
        dim = D
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = D
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = D
    elif F == 'F5':
        fobj = F5
        lb = -100
        ub = 100
        dim = D
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = D
    elif F == 'F7':
        fobj = F7
        lb = -100
        ub = 100
        dim = D
    elif F == 'F8':
        fobj = F8
        lb = -100
        ub = 100
        dim = D
    elif F == 'F9':
        fobj = F9
        lb = -100
        ub = 100
        dim = D
    elif F == 'F10':
        fobj = F10
        lb = -32.768
        ub = 32.768
        dim = D
    elif F == 'F11':
        fobj = F11
        lb = -100
        ub = 100
        dim = D
    elif F == 'F12':
        fobj = F12
        lb = -100
        ub = 100
        dim = D
    elif F == 'F13':
        fobj = F13
        lb = -600
        ub = 600
        dim = D
    elif F == 'F14':
        fobj = F14
        lb = -50
        ub = 50
        dim = D

    return lb, ub, dim, fobj




def RungeKutta(XB, XW, DelX):
    dim = len(XB)
    C = np.random.randint(1, 3) * (1 - np.random.rand())
    r1 = np.random.rand(dim)
    r2 = np.random.rand(dim)
    
    K1 = 0.5 * (np.random.rand() * XW - C * XB)
    K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2) - (C * XB + r1 * K1 * DelX / 2))
    K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2) - (C * XB + r1 * K2 * DelX / 2))
    K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

    XRK = K1 + 2 * K2 + 2 * K3 + K4
    SM = 1/6 * XRK
    return SM


def Run(nP, MaxIt, lb, ub, dim, fobj):
    Cost = np.zeros(nP)
    X = initialization(nP, dim, ub, lb)
    Xnew2 = np.zeros(dim)

    Convergence_curve = np.zeros(MaxIt)

    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    Best_Cost, ind = np.min(Cost), np.argmin(Cost)
    Best_X = X[ind, :]

    Convergence_curve[0] = Best_Cost

    it = 1
    while it < MaxIt:
        it += 1
        f = 20 * np.exp(-12 * (it / MaxIt))
        Xavg = np.mean(X, axis=0)
        SF = 2 * (0.5 - np.random.rand(nP)) * f
        for i in range(nP):
            ind_l = np.argmin(Cost)
            lBest = X[ind_l, :]

            A, B, C = RndX(nP, i)
            ind1 = np.argmin(Cost[[A, B, C]])


            gama = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2 * np.random.rand(dim) * np.abs(Stp)


            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]

            SM = RungeKutta(Xb, Xw, DelX)

            L = np.random.rand(dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]
            Xm = L * Best_X + (1 - L) * lBest

            vec = np.array([1, -1])
            flag = np.floor(2 * np.random.rand(dim))
            r=np.zeros(dim)
            for j in range(dim):
                r[j] = vec[int(flag[j])]

            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)


            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * (SM) + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * (SM) + mu * (X[A, :] - X[B, :])


            FU, FL = Xnew > ub, Xnew < lb
            Xnew = Xnew * (~(FU + FL)) + ub * FU + lb * FL
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew

            if np.random.rand() < 0.5:
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.floor(Unifrnd(-1, 2, 1, 1))

                u = 2 * np.random.rand(dim)
                w = Unifrnd(0, 2, 1, dim) * EXP  # (Eq.19-1)

                A, B, C = RndX(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3  # (Eq.19-2)

                beta = np.random.rand(dim)
                Xnew1 = beta * (Best_X) + (1 - beta) * (Xavg)  # (Eq.19-3)

                for j in range(dim):
                    if w[0, j] < 1:
                        Xnew2[j] = Xnew1[j] + r * w[0, j] * np.abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[0, j] * np.abs(
                            (u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

                FU, FL = Xnew2 > ub, Xnew2 < lb
                Xnew2 = Xnew2 * (~(FU + FL)) + ub * FU + lb * FL
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else:
                    if np.random.rand() < w[0, np.random.randint(dim)]:
                        SM = RungeKutta(X[i, :], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (
                                    SM + (2 * np.random.rand(dim) * Best_X - Xnew2))

                        FU, FL = Xnew > ub, Xnew < lb
                        Xnew = Xnew * (~(FU + FL)) + ub * FU + lb * FL
                        CostNew = fobj(Xnew)

                        if CostNew < Cost[i]:
                            X[i, :] = Xnew
                            Cost[i] = CostNew

            if Cost[i] < Best_Cost:
                Best_X = X[i, :]
                Best_Cost = Cost[i]

        Convergence_curve[it-1] = Best_Cost
        print('it :', it, ', Best Cost =', Convergence_curve[it-1])

    return Best_Cost, Best_X, Convergence_curve


def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z



def RndX(nP, i):
    Qi = np.random.permutation(nP)
    Qi = Qi[Qi != i]
    A, B, C = Qi[:3]
    return A, B, C


nP = 50
Func_name = 'F1'
MaxIt = 500


lb, ub, dim, fobj = BenchmarkFunctions(Func_name)


Best_fitness, BestPositions, Convergence_curve = Run(nP, MaxIt, lb, ub, dim, fobj)


plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
#注：原matlab代码中使用semilogy画图时应该将hold on放在semilogy后面，否则无法画出对数坐标图。
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.grid(False)
plt.legend(['RUN'])
plt.axis('tight')
plt.show()
