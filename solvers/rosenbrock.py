from collections import deque
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                                         validate_first_step, warn_extraneous, num_jac)
from scipy.sparse import issparse, eye, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize._numdiff import group_columns


MAX_ORDER = 5
MIN_H = 1e-12


class Rosenbrock(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1.49012e-08, atol=1e-6, jac=None, jac_sparsity=None, vectorized=False, first_step=None, method='rodas4', mass=None, autonomous=False, **extraneous):

        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(self.fun, self.t, self.y, f,
                                             self.direction, 1,
                                             self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)

        self.atol = atol
        self.rtol = rtol

        self.select_method(method)
        self.sparse = issparse(self.J) or issparse(mass)

        if mass is None:
            if self.sparse:
                mass = eye(self.n)
            else:
                self.mass = np.eye(self.n)
        else:
            if self.sparse and not issparse(mass):
                mass = csr_matrix(mass)
            else:
                self.mass = mass  # mettre self.J en sparse?

        self.solve = spsolve if issparse(self.mass) else np.linalg.solve

        self.RejectLastH = False
        self.RejectMoreH = False
        self.t_bound = t_bound
        self.autonomous = autonomous

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y):
                self.njev += 1
                f = self.fun_single(t, y)
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev += 1
            if issparse(J):
                J = csc_matrix(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=y0.dtype)
            else:
                J = np.asarray(J, dtype=y0.dtype)

                def jac_wrapped(t, y):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=y0.dtype)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
        else:
            if issparse(jac):
                J = csc_matrix(jac, dtype=y0.dtype)
            else:
                J = np.asarray(jac, dtype=y0.dtype)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):

        T, Y, H, _, _ = self.ROSfindH(
            min(self.h_abs, abs(self.t_bound-self.t)), self.t, self.y, {})
        self.t = T
        self.y = Y
        self.h_abs = H

        return True, None

    def ROSfindH(self, H, T, Y, ODEargs):
        fac_min, fac_max, fac_safe = 0.2, 6.0, 0.9

        fcn0 = self.fun(T, Y, **ODEargs)  # ODEfcn should return array

        jac0 = self.jac(T, Y, **ODEargs)

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(T, self.direction * np.inf) - T)

        while True:  # Until H accepted
            K = self.ROScalcK(H*self.direction, T, Y,
                              fcn0, jac0, ODEargs)
            # self.ISTAT['Nfun'] += sum(self.new_f)-1
            # self.ISTAT['Ninv'] += self.stages

            # Compute the new solution
            Ynew = Y*1
            for j in range(self.stages):
                Ynew += self.M[j] * K[j]

            # Compute the error estimation
            Yerr = np.zeros(Y.shape)
            for j in range(self.stages):
                Yerr += self.E[j] * K[j]
            Nerr, E = self.ROSnormErr(Y, Ynew, Yerr)

            # New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac = min(fac_max, max(fac_min, fac_safe /
                      (Nerr**(1/self.est_loc_err))))
            Hnew = H*Fac

            # Check the error magnitude and adjust step size
            # self.ISTAT['Nstp'] += 1
            if (Nerr <= 1) or (H <= min_step):  # Accept step
                # self.ISTAT['Nacc'] += 1
                Y = Ynew
                T += self.direction*H
                Hnew = max(min_step, min(Hnew, max_step))
                if self.RejectLastH:
                    # No step size increase after a rejected step
                    Hnew = min(Hnew, H)
                self.RejectLastH = False
                self.RejectMoreH = False
                H = Hnew
                break  # EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            else:  # Reject step
                if self.RejectMoreH:
                    Hnew = H*0.1
                self.RejectMoreH = self.RejectLastH
                self.RejectLastH = True
                H = Hnew
                # self.ISTAT['Nrej'] += 1
        return T, Y, H, K, E

    def ROScalcK(self, h, T, Y, fcn0, jac0, ODEargs):
        K = list(range(self.stages))  # K element should be array

        igh_j = self.mass/(h*self.gamma[0]) - jac0
        # I/(gamma*h)-jac

        # For the 1st istage the function has been computed previously
        fcn = fcn0*1
        RHS = fcn
        if not self.autonomous:
            dfdt = self.dFundt(self.t, self.y, fcn)
            RHS += self.direction*h*dfdt*self.gamma[0]
        K[0] = self.solve(igh_j, RHS)
        # The coefficient matrices A and C are strictly lower triangular.
        # The lower triangular (subdiagonal) elements are stored in row-wise
        # order: A(2,1) = self.A[0], A(3,1)=self.A[1], A(3,2)=self.A[2], etc.
        # The general mapping formula is:
        #       A(i,j) = self.A[ (i-1)*(i-2)/2 + j - 1 ]
        #       C(i,j) = self.C[ (i-1)*(i-2)/2 + j - 1 ]
        for istage in range(2, self.stages+1):
            # istage > 1 and a new function evaluation is needed
            if self.new_f[istage-1]:
                Ynew = Y*1
                for j in range(istage-1):  # note x += 1 and x = x + 1 mutable
                    Ynew += self.A[(istage-1)*(istage-2)//2+j]*K[j]
                Tau = T + self.alpha[istage-1]*h
                fcn = self.fun(Tau, Ynew, **ODEargs)
            RHS = fcn*1
            C_H_sum = np.zeros_like(RHS)
            for j in range(istage-1):
                C_H = self.C[(istage-1)*(istage-2)//2+j]/h
                # RHS = RHS + C_H*K[j]
                C_H_sum += C_H*K[j]
            RHS = RHS + self.mass.dot(C_H_sum)
            if not self.autonomous:
                RHS += self.direction*h*dfdt*self.gamma[istage-1]
            K[istage-1] = self.solve(igh_j, RHS)
        return K

    def ROSnormErr(self, Y, Ynew, Yerr):
        Ymax = np.maximum(abs(Y), abs(Ynew))  # element-wise maximum
        Ytol = self.atol + self.rtol*Ymax  # Y tolorance
        Ynor = Yerr/Ytol
        err = (sum(Ynor**2)/len(Y))**0.5
        Nerr = max(err, 1.0e-10)
        return Nerr, Ynor

    def dFundt(self, t, y, fy):
        roundoff = 1e-16
        delta_min = 1e-5

        delta = np.sqrt(roundoff) * np.sqrt(np.maximum(abs(t), delta_min))
        return (self.fun(t+delta, y) - fy)/delta

    def select_method(self, method):
        if method == 'ros2':
            self.ros2()
        elif method == 'ros3':
            self.ros3()
        elif method == 'ros4':
            self.ros4()
        elif method == 'rodas3':
            self.rodas3()
        elif method == 'rodas4':
            self.rodas4()
        else:
            raise ValueError("Method not found")

    def ros2(self):
        # Number of self.stages
        self.stages = 2

        self.A = list(range(self.stages*(self.stages-1)//2))
        self.C = list(range(self.stages*(self.stages-1)//2))
        self.M, self.E = list(range(self.stages)), list(
            range(self.stages))
        self.alpha, self.gamma = list(
            range(self.stages)), list(range(self.stages))
        self.new_f = list(range(self.stages))

        g = 1.0 + 1.0/2.0**0.5
        # A_i = Coefficients for Ynew
        self.A[0] = (1.0)/g
        # C_i = Coefficients for RHS K_j
        self.C[0] = (-2.0)/g
        # M_i = Coefficients for new step solution
        self.M[0] = (3.0)/(2.0*g)
        self.M[1] = (1.0)/(2.0*g)
        # E_i = Coefficients for error estimator
        self.E[0] = 1.0/(2.0*g)
        self.E[1] = 1.0/(2.0*g)
        # Y_stage_i = Y( T + H*Alpha_i )
        self.alpha[0] = 0.0
        self.alpha[1] = 1.0
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        self.gamma[0] = g
        self.gamma[1] = -g
        # self.est_loc_err = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        self.est_loc_err = 2.0
        # Does the stage i require self.A new function evaluation (self.new_f[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (self.new_f[i]=FALSE)
        self.new_f[0] = True
        self.new_f[1] = True

    def ros3(self):
        # Number of stages
        self.stages = 3

        self.A = list(range(self.stages*(self.stages-1)//2))
        self.C = list(range(self.stages*(self.stages-1)//2))
        self.M, self.E = list(range(self.stages)), list(range(self.stages))
        self.alpha, self.gamma = list(
            range(self.stages)), list(range(self.stages))
        self.new_f = list(range(self.stages))

        # A_i = Coefficients for Ynew
        self.A[0] = 1.0
        self.A[1] = 1.0
        self.A[2] = 0.0
        # C_i = Coefficients for RHS K_j
        self.C[0] = -0.10156171083877702091975600115545e+01
        self.C[1] = 0.40759956452537699824805835358067e+01
        self.C[2] = 0.92076794298330791242156818474003e+01
        # M_i = Coefficients for new step solution
        self.M[0] = 0.1e+01
        self.M[1] = 0.61697947043828245592553615689730e+01
        self.M[2] = -0.42772256543218573326238373806514
        # E_i = Coefficients for error estimator
        self.E[0] = 0.5
        self.E[1] = -0.29079558716805469821718236208017e+01
        self.E[2] = 0.22354069897811569627360909276199
        # Y_stage_i = Y( T + H*Alpha_i )
        self.alpha[0] = 0.0
        self.alpha[1] = 0.43586652150845899941601945119356
        self.alpha[2] = 0.43586652150845899941601945119356
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        self.gamma[0] = 0.43586652150845899941601945119356
        self.gamma[1] = 0.24291996454816804366592249683314
        self.gamma[2] = 0.21851380027664058511513169485832e+01
        # self.est_loc_err = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        self.est_loc_err = 3.0
        # Does the stage i require a new function evaluation (self.new_f[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (self.new_f[i]=FALSE)
        self.new_f[0] = True
        self.new_f[1] = True
        self.new_f[2] = False

    def ros4(self):
        # Number of stages
        self.stages = 4

        self.A = list(range(self.stages*(self.stages-1)//2))
        self.C = list(range(self.stages*(self.stages-1)//2))
        self.M, self.E = list(range(self.stages)), list(range(self.stages))
        self.alpha, self.gamma = list(
            range(self.stages)), list(range(self.stages))
        self.new_f = list(range(self.stages))

        # A_i = Coefficients for Ynew
        self.A[0] = 0.2000000000000000e+01
        self.A[1] = 0.1867943637803922e+01
        self.A[2] = 0.2344449711399156
        self.A[3] = self.A[1]
        self.A[4] = self.A[2]
        self.A[5] = 0.0
        # C_i = Coefficients for RHS K_j
        self.C[0] = -0.7137615036412310e+01
        self.C[1] = 0.2580708087951457e+01
        self.C[2] = 0.6515950076447975
        self.C[3] = -0.2137148994382534e+01
        self.C[4] = -0.3214669691237626
        self.C[5] = -0.6949742501781779
        # M_i = Coefficients for new step solution
        self.M[0] = 0.2255570073418735e+01
        self.M[1] = 0.2870493262186792
        self.M[2] = 0.4353179431840180
        self.M[3] = 0.1093502252409163e+01
        # E_i = Coefficients for error estimator
        self.E[0] = -0.2815431932141155
        self.E[1] = -0.7276199124938920e-01
        self.E[2] = -0.1082196201495311
        self.E[3] = -0.1093502252409163e+01
        # Y_stage_i = Y( T + H*Alpha_i )
        self.alpha[0] = 0.0
        self.alpha[1] = 0.1145640000000000e+01
        self.alpha[2] = 0.6552168638155900
        self.alpha[3] = self.alpha[2]
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        self.gamma[0] = 0.5728200000000000
        self.gamma[1] = -0.1769193891319233e+01
        self.gamma[2] = 0.7592633437920482
        self.gamma[3] = -0.1049021087100450
        # self.est_loc_err = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        self.est_loc_err = 4.0
        # Does the stage i require a new function evaluation (self.new_f[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (self.new_f[i]=FALSE)
        self.new_f[0] = True
        self.new_f[1] = True
        self.new_f[2] = True
        self.new_f[3] = False

    def rodas3(self):
        # Number of stages
        self.stages = 4

        self.A = list(range(self.stages*(self.stages-1)//2))
        self.C = list(range(self.stages*(self.stages-1)//2))
        self.M, self.E = list(range(self.stages)), list(range(self.stages))
        self.alpha, self.gamma = list(
            range(self.stages)), list(range(self.stages))
        self.new_f = list(range(self.stages))

        # A_i = Coefficients for Ynew
        self.A[0] = 0.0
        self.A[1] = 2.0
        self.A[2] = 0.0
        self.A[3] = 2.0
        self.A[4] = 0.0
        self.A[5] = 1.0
        # C_i = Coefficients for RHS K_j
        self.C[0] = 4.0
        self.C[1] = 1.0
        self.C[2] = -1.0
        self.C[3] = 1.0
        self.C[4] = -1.0
        self.C[5] = -(8.0/3.0)
        # M_i = Coefficients for new step solution
        self.M[0] = 2.0
        self.M[1] = 0.0
        self.M[2] = 1.0
        self.M[3] = 1.0
        # E_i = Coefficients for error estimator
        self.E[0] = 0.0
        self.E[1] = 0.0
        self.E[2] = 0.0
        self.E[3] = 1.0
        # Y_stage_i = Y( T + H*Alpha_i )
        self.alpha[0] = 0.0
        self.alpha[1] = 0.0
        self.alpha[2] = 1.0
        self.alpha[3] = 1.0
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        self.gamma[0] = 0.5
        self.gamma[1] = 1.5
        self.gamma[2] = 0.0
        self.gamma[3] = 0.0
        # self.est_loc_err = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        self.est_loc_err = 3.0
        # Does the stage i require a new function evaluation (self.new_f[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (self.new_f[i]=FALSE)
        self.new_f[0] = True
        self.new_f[1] = False
        self.new_f[2] = True
        self.new_f[3] = True

    def rodas4(self):
        # Number of self.stages
        self.stages = 6

        self.A = list(range(self.stages*(self.stages-1)//2))
        self.C = list(range(self.stages*(self.stages-1)//2))
        self.M, self.E = list(range(self.stages)), list(range(self.stages))
        self.alpha, self.gamma = list(
            range(self.stages)), list(range(self.stages))
        self.new_f = list(range(self.stages))

        # A_i = Coefficients for Ynew
        self.A[0] = 0.1544000000000000e+01
        self.A[1] = 0.9466785280815826
        self.A[2] = 0.2557011698983284
        self.A[3] = 0.3314825187068521e+01
        self.A[4] = 0.2896124015972201e+01
        self.A[5] = 0.9986419139977817
        self.A[6] = 0.1221224509226641e+01
        self.A[7] = 0.6019134481288629e+01
        self.A[8] = 0.1253708332932087e+02
        self.A[9] = -0.6878860361058950
        self.A[10] = self.A[6]
        self.A[11] = self.A[7]
        self.A[12] = self.A[8]
        self.A[13] = self.A[9]
        self.A[14] = 1.0
        # C_i = Coefficients for RHS K_j
        self.C[0] = -0.5668800000000000e+01
        self.C[1] = -0.2430093356833875e+01
        self.C[2] = -0.2063599157091915
        self.C[3] = -0.1073529058151375
        self.C[4] = -0.9594562251023355e+01
        self.C[5] = -0.2047028614809616e+02
        self.C[6] = 0.7496443313967647e+01
        self.C[7] = -0.1024680431464352e+02
        self.C[8] = -0.3399990352819905e+02
        self.C[9] = 0.1170890893206160e+02
        self.C[10] = 0.8083246795921522e+01
        self.C[11] = -0.7981132988064893e+01
        self.C[12] = -0.3152159432874371e+02
        self.C[13] = 0.1631930543123136e+02
        self.C[14] = -0.6058818238834054e+01
        # M_i = Coefficients for new step solution
        self.M[0] = self.A[6]
        self.M[1] = self.A[7]
        self.M[2] = self.A[8]
        self.M[3] = self.A[9]
        self.M[4] = 1.0
        self.M[5] = 1.0
        # E_i = Coefficients for error estimator
        self.E[0] = 0.0
        self.E[1] = 0.0
        self.E[2] = 0.0
        self.E[3] = 0.0
        self.E[4] = 0.0
        self.E[5] = 1.0
        # Y_stage_i = Y( T + H*Alpha_i )
        self.alpha[0] = 0.000
        self.alpha[1] = 0.386
        self.alpha[2] = 0.210
        self.alpha[3] = 0.630
        self.alpha[4] = 1.000
        self.alpha[5] = 1.000
        # Gamma_i = \sum_j^i gamma_{i,j}, Coefficients of t-derivative
        self.gamma[0] = 0.2500000000000000
        self.gamma[1] = -0.1043000000000000
        self.gamma[2] = 0.1035000000000000
        self.gamma[3] = -0.3620000000000023e-01
        self.gamma[4] = 0.0
        self.gamma[5] = 0.0
        # self.est_loc_err = estimator of local order, the minimum between the main and
        #           the embedded scheme orders + 1
        self.est_loc_err = 4.0
        # Does the stage i require self.A new function evaluation (self.new_f[i]=TRUE)
        #  or re-use the function evaluation from stage i-1 (self.new_f[i]=FALSE)
        self.new_f[0] = True
        self.new_f[1] = True
        self.new_f[2] = True
        self.new_f[3] = True
        self.new_f[4] = True
        self.new_f[5] = True

    def _dense_output_impl(self):
        return SABMDenseOutput(self.t_old, self.t, self.h * self.direction,
                               self.order, self.D[:self.order + 1].copy())


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
