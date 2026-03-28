import numpy as np
import scipy.linalg
import scipy.signal
import random
from scipy.integrate import solve_ivp

class CGM():
    def __init__(self, a, b, c, d, dt, glucose_penalty, insulin_penalty, controller_penalty, derivitive_penalty, target=83, base_insulin=10):
        # This is our target glucose measurement 
        self.target = target
        # This is the baseline amount of insulin
        self.base_insulin = base_insulin 
        # Our model will penalize the difference between the current state and these baselines

        # Define evolution matrices for AX + Bu
        A = np.array([[-a, -b], [d, -c]])
        B = np.array([[0], [1]])

        # Now we convert to a discrete system since we only sample every few minutes
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html#scipy.signal.cont2discrete
        discrete_system = scipy.signal.cont2discrete((A, B, np.eye(2), np.zeros((2,1))), dt, method='zoh')
        self.A = discrete_system[0]
        self.B = discrete_system[1]
        self.Q = np.array([[glucose_penalty, 0], [0, insulin_penalty]])
        
        # Augment A and B for the 3-state regime (current glucose, insulin, past glucose)
        self.A2 = np.pad(self.A, ((0,1), (0,1)), mode='constant')
        self.A2[2, 0] = 1.0 
        self.B2 = np.pad(self.B, ((0,1), (0,0)), mode='constant')
        
        # Q2 requires cross-terms to properly penalize the difference (current - past)^2
        self.Q2 = np.array([
            [glucose_penalty + derivitive_penalty, 0, -derivitive_penalty],
            [0, insulin_penalty, 0],
            [-derivitive_penalty, 0, derivitive_penalty]
        ])
        
        self.R = np.array([controller_penalty])
        

        self.B3 = np.array([[1], [0]]) 
        
        self.K1, self.K2, self.K3 = self.fit()
        self.past = None

    def fit(self):
        # Solves discrete algebraic riccati equations
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K1 = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        
        # Pass the augmented matrices to the middle regime solver
        P2 = scipy.linalg.solve_discrete_are(self.A2, self.B2, self.Q2, self.R)
        K2 = np.linalg.inv(self.R + self.B2.T @ P2 @ self.B2) @ (self.B2.T @ P2 @ self.A2)
        
        P3 = scipy.linalg.solve_discrete_are(self.A, self.B3, self.Q, self.R)
        # Fixed typo: K3 must use P3, not P
        K3 = np.linalg.inv(self.R + self.B3.T @ P3 @ self.B3) @ (self.B3.T @ P3 @ self.A)
        
        return K1, K2, K3

    def control(self, current_values):
        """Given the value observed by the CGM use the objective to apply optimal control"""
        # First we calculate the errors
        glucose, insulin = current_values
        x1 = glucose - self.target
        x2 = insulin - self.base_insulin
        
        if self.past == None:
            self.past = x1 
            
        xk = np.array([[x1], [x2]])
        
        if glucose > 160:
            u = -self.K1 @ xk
        elif glucose < 60:
            u = -self.K3 @ xk
        else:
            xk = np.array([[x1], [x2], [self.past]])
            u = -self.K2 @ xk
            
        # We can't deliver negative insulin, the indexing is because u is technically a matrix
        u = max(0.0, u[0][0] + self.base_insulin)
        
        # Track the error state, not the raw glucose value
        self.past = x1 
        return u
    
def square_wave(start, length, magnitude):
    vals = np.zeros(1440)
    size = magnitude / length
    for i in range(length):
        vals[start+i] = size + 0.1 * np.random.randint(-np.abs(size), np.abs(size))
    return vals


def spike():
    # First we create an empty array of the correct length 
    ep = 1e-6
    spikes = np.zeros(1440)

    # We model the dawn phenomenon with an average spike of 20 mg/dl
    dp = random.randint(180, 480)

    # Random scaling 
    dpscale = random.uniform(ep, 2)
    spikes[dp] = dpscale * 20

    # We model breakfast with a single spike of average 40
    bs = random.randint(420, 540)
    bscale = random.uniform(ep, 2)
    bs_magnitude = bscale * 40
    bs_length = random.randint(6, 12)
    spikes += square_wave(bs, bs_length, bs_magnitude)

    # We next model lunch with a single spike of average 60
    ls = random.randint(660, 780)
    lscale = random.uniform(ep, 2)
    l_magnitude = lscale * 60
    l_length = random.randint(6, 24)
    spikes += square_wave(ls, l_length, l_magnitude)

    # We model dinner with two spikes 60 and 20 seperated by 20 minutes
    ds = random.randint(1080, 1200)
    dscale = random.uniform(ep, 2)
    d_magnitude = dscale * 80
    d_length = random.randint(6, 36)
    spikes += square_wave(ds, d_length, d_magnitude)

    # We add a snack
    ss = random.randint(360, 1380)
    sscale = random.uniform(ep, 2)
    ss_magnitude = sscale * 20
    ss_length = random.randint(6, 9)
    spikes += square_wave(ss, ss_length, ss_magnitude)

    # Now we add exercise
    es = random.randint(360, 1380)
    escale = random.uniform(ep, 2)
    es_magnitude = -escale* 35
    es_length = random.randint(1, 12)
    spikes += square_wave(es, es_length, es_magnitude)


    var = np.random.uniform(.6, 1.4)
    spikes *= var
    
    return spikes

def simulate(spikes, t_steps, dt, starting, a, b, c, d, glucose_penalty, insulin_penalty, controller_penalty, derivitive_penalty, target=83, base_insulin=10, step_delay=6, cgm_noise=True):
    cgm = CGM(a, b, c, d, dt, glucose_penalty, insulin_penalty, controller_penalty, derivitive_penalty, target=83, base_insulin=10)
    def fun(t, x, u):
        xerr = x - np.array([target, base_insulin])
        uerr = u - base_insulin
        dxdt = np.array([[-a, -b], [d, -c]]) @ xerr + np.array([0, 1]) * uerr

        #Make sure we don't get negative insulin
        if x[1] <= 0 and dxdt[1] < 0:
            dxdt[1] = 0
        return dxdt
    current = starting
    t_hist = []
    y_hist = []
    readings = [starting.copy()]
    for t in range(0, t_steps, dt):
        current[0] += np.sum(spikes[t:t+dt])

        if cgm_noise:
            cgm_noise = np.random.normal(0, 10, size=2) # adding noise to what the cgm measures
        else:
            cgm_noise = 0

        # Time delay of cgm
        if step_delay == 0:
            delayed = current.copy() + cgm_noise
        elif len(readings) > step_delay:
            delayed = readings[-step_delay] + cgm_noise
        else:
            delayed = readings[0] + cgm_noise

        u = cgm.control(delayed)
        sol = solve_ivp(fun=fun, t_span=(t, t+dt), y0=current, args=(u,))
        current = np.maximum(0, sol.y[:, -1])
        readings.append(current.copy())
        t_hist.append(sol.t)
        y_hist.append(sol.y)
    return np.concatenate(t_hist), np.concatenate(y_hist, axis=1)

