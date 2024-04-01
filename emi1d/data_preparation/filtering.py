import numpy as np
# from scipy.special import jv as besselj
from scipy.special import j0
from scipy.special import j1
import scipy.integrate as integrate
from hankel import HankelTransform
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt

def calc_epsilon_d(df):
    headers = [i for i in df.columns if i.startswith('Q') or i.startswith('I')]
    for head_i in headers:
        print(head_i)
        df['relnoise_' + head_i] = df['noise_abs'] / df[head_i]
    return df

def get_relnoise_vector(df, index):
    headers = [i for i in df.columns if i.startswith('relnoise')]
    relnoise = []
    for head_i in headers:
        relnoise.append(df[head_i][index])

    return np.array(relnoise)

def add_colorbar_method2(RHO_grid, fig, ax, plotmin=None, plotmax=None, title=None, sep=0.5, width=0.25):
    # sep: separation of the color bar from the main ax in centimeters
    # width: width of the color bar ax in centimeters

    fig_dummy = plt.figure()
    ax_dummy = fig_dummy.add_subplot(111)
    im = ax_dummy.imshow(RHO_grid, cmap='jet', vmin=plotmin, vmax=plotmax)
    # plt.close(fig=fig_dummy)  # this used to work before, now it doesn't (version change?)
    plt.close(fig_dummy)

    # bar ax
    fig_size_cm = fig.get_size_inches() * 2.54
    pos_ax = ax.get_position().get_points()
    #ax_width = pos_ax[1, 0] - pos_ax[0, 0]
    ax_height = pos_ax[1, 1] - pos_ax[0, 1]

    ax_bar = fig.add_axes([pos_ax[1,0] + sep / fig_size_cm[0], pos_ax[0, 1], width / fig_size_cm[0], ax_height])
    fig.colorbar(im, cax=ax_bar, orientation='vertical')



def is_outlier(array, filt_len, thresh=3.0):
	nsteps = int(np.ceil(len(array) / filt_len))
	valid = True * np.ones_like(array)
	for i in range(nsteps):
		index_ini = i*filt_len
		index_end = i*filt_len+filt_len * (i*filt_len+filt_len <= len(array)) + len(array) * (i*filt_len+filt_len > len(array))
		array_i = array[index_ini:index_end]

		median = np.median(array_i)
		diff = np.abs((array_i - median))
		med_abs_deviation = np.median(diff)

		modified_z_score = 0.6745 * diff / med_abs_deviation

		valid_partial = (modified_z_score < thresh)

		valid[index_ini:index_end] = valid_partial * (valid_partial  == True)

	return valid

def boxcar(array, filt_len=3):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(array, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=filt_len:
            moving_ave = (cumsum[i] - cumsum[i-filt_len])/filt_len
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
        else:
            moving_ave = x
            moving_aves.append(moving_ave)

    array_2 = np.array(moving_aves)

    return array_2
    

def separation_content(df, header='HCP'):
    sep_list = []
    nstring = len(header)
    for headi in df.columns:
        if headi[0:nstring] == header:
            sep_list.append(headi[nstring:nstring+2])

    sep_list = list(dict.fromkeys(sep_list))  # remove repeated values

    return sep_list


def calc_Q_HCP(sigma_app, separation):
    Q_HCP = 0.25 * (2.0 * np.pi * 9000.0) * (4 * np.pi * 10 ** -7) * (separation ** 2) * (sigma_app / 1000.)
    return Q_HCP * (10 ** 6)


def calc_Q_PRP(sigma_app, separation):
    Q_PRP = 0.25 * (2.0 * np.pi * 9000.0) * (4 * np.pi * 10 ** -7) * (separation ** 2) * (sigma_app / 1000.)
    return Q_PRP * (10 ** 6)

def rho_app_approx(hs_h0, s):
    u0 = 4 * np.pi * (10 ** -7)
    rho_app = 4 / (9000 * 2 * np.pi * u0 * s ** 2) * hs_h0
    return rho_app * 1000


class dualem_fop():
    def __init__(self, zIN, step=0.03):
        self.h = 0.2  # height of the dualem (don't set it to zero; otherwise, the numerical integration crashes
        self.coilspacing_HCP = np.array([2.0, 4.0, 8.0])  # coil spacing
        self.coilspacing_PRP = np.array([2.1, 4.1, 8.1])  # coil spacing
        self.w = 9000 * (2 * np.pi)  # Hertz
        self.nlay = len(zIN)  # without air
        self.dzIN = np.flip(np.diff(zIN))
        self.u0 = 4 * np.pi * (10 ** -7)
        #self.ht_z = HankelTransform(nu=0, N=120, h=0.03)  # Create the HankelTransform instance
        #self.ht_rho = HankelTransform(nu=1, N=120, h=0.03)
        #self.ht_z = HankelTransform(nu=0, N=None, h=0.01)  # slow
        #self.ht_rho = HankelTransform(nu=1, N=None, h=0.01)  # slow
        self.ht_z = HankelTransform(nu=0, N=120, h=step)
        self.ht_rho = HankelTransform(nu=1, N=120, h=step)

    def f_r_over_r(self, lambda_EM):
        k_N = (-1j * self.u0 * self.model[0] * self.w) ** 0.5  # I'm using Timo's type of grid
        u_cap_N = (lambda_EM ** 2 - k_N ** 2) ** 0.5
        u_cap_n = 0  # this is unnecessary?
        for i in range(1, self.nlay):
            k_n = (-1j * self.u0 * self.model[i] * self.w) ** 0.5  # I'm using Timo's type of grid
            u_n = (lambda_EM ** 2 - k_n ** 2) ** 0.5

            u_cap_n = u_n * ((u_cap_N + u_n * np.tanh(u_n * self.dzIN[i-1])) / (u_n + u_cap_N * np.tanh(u_n * self.dzIN[i-1])))
            u_cap_N = u_cap_n

        rTE = (lambda_EM - u_cap_N) / (lambda_EM + u_cap_N)
        f_x = rTE * np.exp(-2 * lambda_EM * self.h) * lambda_EM

        return f_x

    def response(self, model):  # TODO: vectorize this method
        self.model = model  # this might not work if I want to parallelize the code
        # Hz equation 4.46 Ward / (4*pi*rho**3)
        Q_Hsz_over_H = (self.coilspacing_HCP ** 3) * np.imag(self.ht_z.transform(self.f_r_over_r, self.coilspacing_HCP, ret_err=False))
        I_Hsz_over_H = (self.coilspacing_HCP ** 3) * np.real(self.ht_z.transform(self.f_r_over_r, self.coilspacing_HCP, ret_err=False))

        # Hrho equation 4.45 Ward / (4*pi*rho**3)
        Q_Hsrho_over_H = (self.coilspacing_PRP ** 3) * np.imag(self.ht_rho.transform(self.f_r_over_r, self.coilspacing_PRP, ret_err=False))
        I_Hsrho_over_H = (self.coilspacing_PRP ** 3) * np.real(self.ht_rho.transform(self.f_r_over_r, self.coilspacing_PRP, ret_err=False))

        fop_list = []
        for i_coil in range(len(self.coilspacing_HCP)):
            fop_list.append(Q_Hsz_over_H[i_coil])
            fop_list.append(I_Hsz_over_H[i_coil])
            fop_list.append(Q_Hsrho_over_H[i_coil])
            fop_list.append(I_Hsrho_over_H[i_coil])

        fop = -np.array(fop_list) * (10 ** 6)

        #fop = -np.array([Q_Hsz_over_H[0], I_Hsz_over_H[0], Q_Hsz_over_H[1], I_Hsz_over_H[1], Q_Hsz_over_H[2], I_Hsz_over_H[2], Q_Hsrho_over_H[0], I_Hsrho_over_H[0], Q_Hsrho_over_H[1], I_Hsrho_over_H[1], Q_Hsrho_over_H[2], I_Hsrho_over_H[2]])

        return fop


class dualem_invert():
    def __init__(self, df, zIN, lambda_reg=10, plot_steps=False):
        self.plot_steps = plot_steps
        self.df = df
        self.nlay = len(zIN) 
        #self.weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 
        self.weights = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) 
        #sensors= [Q_Hsz_over_H[0], I_Hsz_over_H[0], Q_Hsrho_over_H[0], I_Hsrho_over_H[0], Q_Hsz_over_H[1], I_Hsz_over_H[1], Q_Hsrho_over_H[1], I_Hsrho_over_H[1], Q_Hsz_over_H[2], I_Hsz_over_H[2], Q_Hsrho_over_H[2], I_Hsrho_over_H[2]]

        # Irregular grid
        self.zIN = zIN
        self.nIN = len(zIN)
        self.dzIN = np.diff(self.zIN)
        self.zN = 0.5 * (zIN[1:self.nIN] + zIN[0:self.nIN-1])
        dzN = np.diff(self.zN)
        self.dzN = np.hstack((dzN[0:1], dzN))

        self.lambda_reg = lambda_reg
        #self.rel_noise = rel_noise
        self.fop = dualem_fop(self.zIN)

        self.npoints = len(df)
        self.ndata = 2 * (len(self.fop.coilspacing_HCP) + len(self.fop.coilspacing_PRP))
        self.sep_content = ['2', '4', '8']
        self.ini_model()
        self.model_grid = np.ones((self.nlay, ))

    def ini_model(self):
        sigma_avg = np.zeros((self.npoints, ))
        for i, sepi in enumerate(self.sep_content):
            sigma_avg = sigma_avg + self.df['H' + sepi + 'mS_m'] / 1000.0
            sigma_avg = sigma_avg + self.df['P' + sepi + 'mS_m'] / 1000.0

        sigma_avg = sigma_avg / (self.ndata / 2)

        self.sigma_ini = sigma_avg

    def objective_function(self, log_model_values, d, relnoise, log_ini_model, lambda_reg, Cm):
        model_values = np.exp(log_model_values)
        resp = self.fop.response(model_values)
        log_resp = np.log(np.abs(resp))  # negative values are physically possible
        log_d = np.log(np.abs(d))  # negative values are physically possible

        phi_d = np.linalg.norm(((1 / np.log(1 + relnoise)) ** 2) * self.weights * (log_d - log_resp) ** 2)
        phi_m = np.linalg.norm(lambda_reg * (log_model_values - log_ini_model).dot(Cm.transpose()).dot(Cm).dot((log_model_values - log_ini_model)))

        phi = phi_d + phi_m

        return phi

    def callbackF(self, log_model_values_i):
        print('##################################')
        model_values = np.exp(log_model_values_i)
        resp = self.fop.response(model_values)
        plt.plot(self.d_i, resp, '.')
        plt.plot(self.d_i, self.d_i, 'k-')
        plt.ylim(-20000, 100000)
        plt.pause(0.000001)
        plt.draw()

    def C_matrix(self, dzN, C_m_type='identity'):
        if C_m_type == 'smooth':
            C_m = np.zeros((self.nlay-1, self.nlay))
            np.fill_diagonal(C_m, -1)
            np.fill_diagonal(C_m[:, 1:], 1)
            C_m = C_m / dzN[:, None]

        if C_m_type == 'identity':
            C_m = np.identity(self.nlay)

        return C_m

    def run_inversion(self):
        start = time.time()
        import multiprocessing
        pool_obj = multiprocessing.Pool()
        list_of_inv_model = pool_obj.map(self.loop_function, range(self.npoints))
        RHO_grid = np.zeros((self.nlay, self.npoints))
        for ii, rho_i in enumerate(list_of_inv_model):
            RHO_grid[:, ii] = rho_i

        for sepi in self.sep_content:
            self.df['HCP' + sepi + '_inv'] = ""
        for sepi in self.sep_content:
            self.df['PRP' + sepi + '_inv'] = ""
        #RHO_grid[:, pi] = 1.0 / inv_model

        end = time.time()
        print('Elapased time', end - start)
        return RHO_grid


    def loop_function(self, pi):
        print('Point %5d out of %5d' % (pi+1, self.npoints))
        # Initial response
        ini_model = self.model_grid * self.sigma_ini[pi]
        ini_resp_i = self.fop.response(ini_model)
        log_ini_model = np.log(ini_model)

        # Assemble "d" vector (Measured response)
        d_array_i = np.zeros((self.ndata,))
        rel_noise_i = np.zeros((self.ndata,))
        count = 0
        for i_dist, sepi in enumerate(self.sep_content):
            head_Q_HCP_i = 'Q_HCP_' + sepi + 'm'
            head_I_HCP_i = 'I_HCP_' + sepi + 'm'
            head_Q_PRP_i = 'Q_PRP_' + sepi + 'm'
            head_I_PRP_i = 'I_PRP_' + sepi + 'm'

            d_array_i[i_dist*4] = self.df[head_Q_HCP_i][pi]
            d_array_i[i_dist*4+1] = self.df[head_I_HCP_i][pi]
            d_array_i[i_dist*4+2] = self.df[head_Q_PRP_i][pi]
            d_array_i[i_dist*4+3] = self.df[head_I_PRP_i][pi]

            rel_noise_i[i_dist*4] = self.df['relnoise_' + head_Q_HCP_i][pi]
            rel_noise_i[i_dist*4+1] = self.df['relnoise_' + head_I_HCP_i][pi]
            rel_noise_i[i_dist*4+2] = self.df['relnoise_' + head_Q_PRP_i][pi]
            rel_noise_i[i_dist*4+3] = self.df['relnoise_' + head_I_PRP_i][pi]

        self.d_i = d_array_i

        # Assemble "C" matrix
        C_m = self.C_matrix(self.dzN, C_m_type='smooth')  # (smoothness matrix)

        # Inversion
        print('Measured response ')
        print(d_array_i)
        print('Initial response ')
        print(ini_resp_i)

        # invEM = minimize(objective_function, log_ini_model, method="Newton-CG", jac=jacobian_phi_brute_force, args=C_m, tol=1.0e-6)  #
        invEM = minimize(self.objective_function, log_ini_model, method="CG", args=(d_array_i, rel_noise_i, log_ini_model, self.lambda_reg, C_m), tol=1e-6, options={'maxiter': 50})

        inv_model = 1.0 / np.exp(invEM.x)  # resistivity
        inv_resp_i = self.fop.response(inv_model)

        return inv_model



