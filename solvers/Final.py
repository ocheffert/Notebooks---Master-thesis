from collections import deque
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     validate_first_step, warn_extraneous)


MAX_ORDER = 5
MIN_H = 1e-12


class SABMScipy(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, gun=None, order=2, max_step=np.inf, num_diff=None,
                 rtol=1.49012e-08, atol=1e-6, vectorized=False, first_step=None, mode="Explicit", **extraneous):
       
        warn_extraneous(extraneous)
        f = lambda t,x : np.concatenate((fun(t,x[:num_diff],x[num_diff:]), gun(t,x[:num_diff],x[num_diff:])),axis=None)
        super().__init__(f, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        
        f = self.fun(self.t, self.y)
        if first_step is None:
            self.h = select_initial_step(self.fun, self.t, self.y, f,
                                             self.direction, 1,
                                             self.rtol, self.atol)
        else:
            self.h = validate_first_step(first_step, t0, t_bound)
        
        self.h = max([atol**2, MIN_H])
        
        #bashforth coefficients
        self.B = [[1,0,0,0,0,0],
                  [3/2, -1/2, 0,0,0,0],
                  [23/12, -16/12, 5/12, 0,0,0],
                  [55/24, -59/24, 37/24, -9/24,0,0],
                  [1901/720, -2774/720, 2616/720, -1274/720, 251/720, 0],
                  [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440]]

        #moulton coefficients
        self.M = [[1,0,0,0,0,0],
                  [1/2, 1/2, 0,0,0,0],
                  [5/12, 2/3, -1/12, 0,0,0],
                  [3/8, 19/24, -5/24, 1/24,0,0],
                  [251/720, 646/720, -264/720, 106/720, -19/720, 0],
                  [95/288, 1427/1440, -133/240, 241/720, 173/1440, 3/160]]
        
        self.f = fun
        self.g = gun
        
        # number of differential equation
        self.num_diff = num_diff
        
        #order method
        self.p = 1
        self.order = order
        
        #implicit / explicit
        self.mode = mode
        
        self.prev_f_y = deque(maxlen=order)
        self.prev_f_y.append(self.f(t0,y0[:num_diff],y0[num_diff:]))
        

    def _step_impl(self):
        
        if self.mode == "Implicit":
            (y_new, x_new, error) = self.Semi_Implicit_Adams_Bashforth_Moulton()
        elif self.mode == "Explicit":
            (y_new, x_new, error) = self.Semi_Explicit_Adams_Bashforth_Moulton()
        else:
            return False, "Error mode. Need 'Implicit' or 'Explicit' for mode"
        
        if y_new is False:
            return False, x_new
        
        h = self.h
        if error < 1e-14:
            h_new = min([3/2 * h, self.max_step])
            
        else:
            h_hat = h*(self.atol/error)**(1/(self.p+1))
            if h_hat > h:
                h_new = min([h_hat, 3/2 * h, self.max_step])
            else:
                h_new = max([h_hat, 1/2 * h, MIN_H])
                
        self.t += self.h
        self.h = h_new
        self.prev_f_y.append(self.f(self.t, y_new, x_new))
        self.y = np.concatenate((y_new, x_new),axis=None)
        if self.p < self.order and self.p < MAX_ORDER:
            self.p += 1 
        
        return True, None
    
    def _dense_output_impl(self):
        return SABMDenseOutput(self.t_old, self.t, self.h * self.direction,
                               self.order, self.D[:self.order + 1].copy())
        
    def Explicit_Adams_Bashforth(self):
        p = self.p
        B = self.B[p-1]
        h = self.h
        
        y_p = np.copy(self.y[:self.num_diff])
        for i in range(0,p):
            y_p += h*B[i]*self.prev_f_y[p-i-1]
        
        def equations(z):
            return self.g(self.t + h, y_p, z)
        
        x_p = fsolve(equations, x0=self.y[self.num_diff:], xtol=self.rtol)
        return (y_p, x_p)
    
    def Semi_Explicit_Adams_Bashforth_Moulton(self):
        y_p, x_p = self.Explicit_Adams_Bashforth()
        h = self.h
        p = self.p
        M = self.M[p-1]
        y_new = np.copy(self.y[:self.num_diff])
        for i in range(self.num_diff):
            P = np.concatenate((y_new[0:i], y_p[i:]), axis=None)
            y_new[i] += h*M[0]*self.f(self.t+h,P,x_p)[i]
            for j in range(1,p):
                y_new[i] += h*M[j]*self.prev_f_y[p-j][i]
        
        error = max(np.abs(y_new - y_p))
        if error > self.atol and h > MIN_H:
            self.h = h/2
            return self.Semi_Explicit_Adams_Bashforth_Moulton()
        
        elif error > self.atol and h < MIN_H:
            return (False, "Pas bien", None)
        
        def equations(z):
            return self.g(self.t + h, y_new, z)
        
        x_new = fsolve(equations, x0=x_p, xtol=self.rtol)
                
        return (y_new, x_new, error)
    
    def Semi_Implicit_Adams_Bashforth_Moulton(self):
        y_p, x_p = self.Explicit_Adams_Bashforth()
        h = self.h
        p = self.p
        M = self.M[p-1]
        y_new = np.copy(self.y[:self.num_diff])
        for i in range(self.num_diff):
            def equations(z):
                P = np.concatenate((y_new[0:i],z,y_p[i+1:]),axis=None)
                res = y_new[i] + h*M[0]*self.f(self.t+h,P,x_p)[i]
                for j in range(1,p):
                    res += h*M[j]*self.prev_f_y[p-j][i]
                return z - res
            y_new[i] = fsolve(equations, x0=y_p[i], xtol=self.rtol)
        
            error = abs(y_new[i] - y_p[i])
            if error > self.atol and h > MIN_H:
                self.h = h/2
                return self.Semi_Implicit_Adams_Bashforth_Moulton()
            elif error > self.atol and h < MIN_H:
                return (False, "Pas bien", None)
        
        error = max(np.abs(y_new - y_p))
        
        def equations(z):
            return self.g(self.t+h, y_new, z)
        
        x_new = fsolve(equations, x0=x_p, xtol=self.rtol)
           
        return (y_new, x_new, error)
    
    
class SABMDenseOutput(DenseOutput):
    def __init__(self, t_old, t, h, order, D):
        super().__init__(t_old, t)
        self.order = order
        self.t_shift = self.t - h * np.arange(self.order)
        self.denom = h * (1 + np.arange(self.order))
        self.D = D

    def _call_impl(self, t):
        if t.ndim == 0:
            x = (t - self.t_shift) / self.denom
            p = np.cumprod(x)
        else:
            x = (t - self.t_shift[:, None]) / self.denom[:, None]
            p = np.cumprod(x, axis=0)

        y = np.dot(self.D[1:].T, p)
        if y.ndim == 1:
            y += self.D[0]
        else:
            y += self.D[0, :, None]

        return y