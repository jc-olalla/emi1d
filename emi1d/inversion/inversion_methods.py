import time
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
from emi1d.modeling.forward_models import calc_rho_app
from emi1d.data_preparation.utils import get_single_metadata, make_regular_grid


class dataframe_inversion():
    def __init__(self):
        pass

class emi1d_invert():
    def __init__(self, fop, dz, nlay, weights, lambda_reg=10, rel_noise=0.05):
        self.fop = fop
        self.lambda_reg = lambda_reg
        self.weights = weights
        self.rel_noise = rel_noise
        self.dz = dz  # for inversion grid
        self.nlay = nlay  # for inversion grid

    def invert(self, observed_data):
        """
        Function to invert the observed emi1d data
        observed_data: dict
            Dictionary whose items contain the responses
            :math:`H_{\mathrm{s}}/H_{\mathrm{p}}`, and
            whose keys represent the metadata of the response. For example:

            * Q_HCP_9000_1.0: Quadrature HCP response at 9000 Hz and 1.0 meter source-receiver separation
            * I_PRP_9000_1.0: Inphase PRP response at 9000 Hz and 1.0 meter source-receiver separation
            * etc
            """
        # Step 1: Initialize the model based on the observed data
        initial_model = self.initialize_model(observed_data)

        # Step 2: Construct the error matrix
        error_vector = 1.0 / self.df_meas['rel_noise'] * np.ones_like(initial_model)

        # Step 3: Construct the regularization matrix
        regularization_matrix = self.construct_regularization_matrix(initial_model)

        # Step 4: Build the objective function
        objective_function = self.objective_function(error_vector, regularization_matrix, initial_model)

        # Step 5: Minimize the objective function (inversion)
        inverted_parameters = self.minimize_objective_function(objective_function)

        return inverted_parameters

    def make_grid(self):
        self.dz = make_regular_grid(self.nlay, self.dz)

    def get_metadata(self, observed_data):
        meas_data = []
        metadata = []

        # Extracting metadata and observed data separately
        for key, observation in observed_data:
            meta_dict_i = get_single_metadata(key)
            metadata.append(meta_dict_i)
            meas_data.append(observation)

        # Transpose list of dictionaries into a dictionary of lists
        dict_meas = {key: [d[key] for d in metadata] for key in metadata[0]}
        dict_meas['measured_data'] = meas_data

        # Add noise
        dict_meas['rel_noise'] = self.rel_noise

        return pd.DataFrame(data=dict_meas)

    def initialize_model(self, observed_data):
        df_meas = self.get_metadata(observed_data)
        df_meas['rho_app'] = map(
            calc_rho_app,
            df_meas['measured_data'],
            df_meas['s'],
            df_meas['freq']
        )
        df_meas['sigma_app'] = 1.0 / df_meas['rho_app']

        self.df_meas = df_meas

        model_ini = df_meas['sigma_app'].mean() * np.ones_like(self.dz)
        return model_ini


    def construct_regularization_matrix(self, regularization_type="smooth"):
        """
        Regularization matrix

        Parameters
        ----------
        regularization_type: str
            Regularization type 'smooth' or 'identity'.
            Smooth regularization creates a second derivative matrix
        """
        if regularization_type == 'smooth':
            C_m = np.zeros((self.nlay-1, self.nlay))
            np.fill_diagonal(C_m, -1)
            np.fill_diagonal(C_m[:, 1:], 1)
            C_m = C_m / self.dz[:, None]

        if regularization_type == 'identity':
            C_m = np.identity(self.nlay)

        return C_m

    def objective_function(self, log_model_values, d, relnoise, log_ini_model, lambda_reg, Cm):
        """
        Objective function to be minimized in the inversion process.
        The objective function has two components the data functional,
        and the regularization term
        :math:`\Phi = \Phi_{\mathrm{d}} + \lambda \Phi_{\mathrm{g}}`
        The data functional 
        :math:`\Phi_\mathrm{d} = (\\underline{d}-f(\\underline{g}))^\mathrm{T}\\underline{\\underline{D}}^\mathrm{T}\\underline{\\underline{D}}(\\underline{d}-f(\\underline{g}))`

        The regularization term
        :math:`\Phi_\mathrm{g} = (\\underline{g} - \\underline{g}_{\mathrm{ref}})^\mathrm{T}\\underline{\\underline{C}}^\mathrm{T}\\underline{\\underline{C}}(\\underline{g} - \\underline{g}_{\mathrm{ref}})`
        """
        model_values = np.exp(log_model_values)
        resp = self.fop.response(model_values)
        log_resp = np.log(np.abs(resp))  # negative values are physically possible
        log_d = np.log(np.abs(d))  # negative values are physically possible

        phi_d = np.linalg.norm(((1 / np.log(1 + relnoise)) ** 2) * self.weights * (log_d - log_resp) ** 2)
        phi_m = np.linalg.norm(lambda_reg * (log_model_values - log_ini_model).dot(Cm.transpose()).dot(Cm).dot((log_model_values - log_ini_model)))

        phi = phi_d + phi_m

        return phi

    def minimize_objective_function(self):
        pass


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



