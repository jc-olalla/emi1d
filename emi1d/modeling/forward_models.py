import numpy as np
from hankel import HankelTransform


def calc_rho_app(hs_h0, s, freq):
    """
    This function computes the apparent resistivity given the
    Quadrature response (Q) of a half space subjected to a
    vertical magnetic dipole. The separation between the source
    and receiver coils is s. This computation is based on  the
    low-frequency approximation.

    Parameters
    ----------
    hs_h0 : float
        Quadrature response measured at the receiver coil and
        expressed as the ratio of the secondary field to the
        primary field in parts per million (ppm).

    s : float
        Distance between the source magnetic dipole and the receiver
        coil, in meters (m).

    freq : float
        Frequency of the electromagnetic signal, in Hertz (Hz).

    Returns
    -------
    rho_app : float
        Apparent resistivity in ohm-m.

    Notes
    -----
    The low-frequency approximation formula to calculate :math:`\\rho_{\mathrm{app}}` is:

    .. math::

        \sigma_{\mathrm{app}} = \\frac{4}{2\pi \cdot f \cdot \mu_0 \cdot s^2} \cdot \\frac{H_{\mathrm{s}}}{H_{0}}

    .. math::

        \\rho_{\mathrm{app}} = 1 / \sigma_{\mathrm{app}}


    where:

    - :math:`f` is the frequency,
    - :math:`s` is the separation distance, and
    - :math:`\mu_0` is the magnetic permeability of free space
    - :math:`H_{\mathrm{s}}` is the magnetic field at the receiver coil
    - :math:`H_0` is the free-space magnetic field at the receiver coil
    - :math:`\sigma_{\mathrm{app}}` is the electrical conductivity.
    - :math:`\\rho_{\mathrm{app}}` is the electrical resisitivity.

    References
    ----------
    McNeill, 1980 Electromagnetic Terrain Conductivity Measurement at
    Low Induction Numbers. Geonics Ltd.(Technical Note TN-6)

    Examples
    --------
    >>> calc_rho_app(10000, 8, 9000)
    113.6978427005494
    """
    u0 = 4 * np.pi * 1e-7
    sigma_app = 4 * hs_h0 * 1e-6 / (2 * np.pi * freq * u0 * s ** 2)
    return 1 / sigma_app


class emi_fop():
    """
    Forward operator class for electromagnetic induction

    Parameters
    ----------
    h : float
        Height of the source and receiver coils with respect to ground level.
        This is a positive number.
    s_hcp : list
        List of source/receiver distances for the hcp coils
    s_prp : list
        List of source/receiver distances for the prp coils
    freqs : list
        List of frequencies for the forward model in Hertz
    nlay: int
        Number of layers (excluding the air layer)
    dz : float
        Cell size of the forward modeling grid.
    step: float
        Step for the Hankel Transform (default 0.03). The smaller
        the step, the slower the forward model.

    """ 
    def __init__(self, h, s_hcp, s_prp, freqs, nlay, dz, step=0.03):
        self.h = h
        self.coilspacing_HCP = np.array(s_hcp)  # coil spacing
        self.coilspacing_PRP = np.array(s_prp)  # coil spacing
        self.freqs = freqs
        self.make_grid(nlay, dz)
        self.u0 = 4 * np.pi * (10 ** -7)
        self.ht_hcp = HankelTransform(nu=0, N=120, h=step)
        self.ht_prp = HankelTransform(nu=1, N=120, h=step)

    def make_grid(self, nlay, dz):
        self.nlay = nlay
        self.dzIN = dz * np.ones((nlay, ))

    def print_grid(self, model):
        """
        Function to visualize the model grid and associated properties

        Parameters
        ----------
        model: numpy.array

        Returns
            Prints an schematic of the grid and associated properties
        """
        z_cum = 0
        print('            Ground level')
        print('-------------------------------------z=0 m')
        for i, (dz_i, model_i) in enumerate(zip(reversed(self.dzIN), reversed(model))):
            z_cum += dz_i
            if i != len(self.dzIN) - 1:
                print(f'        {model_i} ohm-m   dz={dz_i} m')
                print(f'-------------------------------------z={z_cum} m')
            else:
                print(f'        {model_i} ohm-m   dz=infinite')
                print('-------------------------------------z= infinite')

    def f_r_over_r(self, lambda_EM):
        """
        Method to compute the transverse electric reflection coefficient
        :math:`r_{\mathrm{TE}}` times :math:`exp{(-2\lambda h)}` given a
        wavenumber. This quantity appears in the integral to solve
        :math:`H_s/H_p` for both HCP and PRP configurations.

        Parameters
        ----------
        lambda_EM: float
            Wavenumber
        """
        k_N = (-1j * self.u0 * self.model[0] * self.w) ** 0.5
        u_cap_N = (lambda_EM ** 2 - k_N ** 2) ** 0.5  # N: nlay
        u_cap_n = 0  # n: nlay - 1
        for i in range(1, self.nlay):
            k_n = (-1j * self.u0 * self.model[i] * self.w) ** 0.5
            u_n = (lambda_EM ** 2 - k_n ** 2) ** 0.5
            numerator = u_cap_N + u_n * np.tanh(u_n * self.dzIN[i-1])
            denominator = u_n + u_cap_N * np.tanh(u_n * self.dzIN[i-1])
            u_cap_n = u_n * numerator / denominator
            u_cap_N = u_cap_n

        rTE = (lambda_EM - u_cap_N) / (lambda_EM + u_cap_N)
        f_x = rTE * np.exp(-2 * lambda_EM * self.h) * lambda_EM

        return f_x

    def response(self, model):
        """
        Method to compute the response :math:`H_{\mathrm{s}}/H_{\mathrm{p}}`
        This method compute this integral for HCP
        add equation
        and this integral for the PRP
        add equation

        Parameters
        ----------
        model: numpy.array
            Electrical resistivity values associated to the modeling grid
            The first element of the model array represents the
            bottom-most layer.

        Return
        ------
        fop_resp: dict
            Dictionary whose items contain the responses
            :math:`H_{\mathrm{s}}/H_{\mathrm{p}}`, and
            whose keys represent the metadata of the response. For example:

            * Q_HCP_9000_1.0: Quadrature HCP response at 9000 Hz and 1.0 meter source-receiver separation
            * I_PRP_9000_1.0: Inphase PRP response at 9000 Hz and 1.0 meter source-receiver separation
            * etc

        Notes
        -----
        The HCP response is the result of computing the integral
        (Equation 4.46 Ward, 1988) divided by :math:`4 \pi \\rho^3`

        .. math::

            \\frac{H_{\mathrm{s_{HCP}}}}{H_{\mathrm{p}}} = s_\mathrm{HCP}^3 \int_{0}^{\infty} r_\mathrm{TE} e^{-2\lambda h}\lambda^2 \mathrm{J}_0(\lambda s)\mathrm{d}\lambda.

        Meanwhile, the PRP response is the result of computing the integral
        (Equation 4.45 Ward, 1988) divided by :math:`4 \pi \\rho^3`

        .. math::

            \\frac{H_\mathrm{s_\mathrm{PRP}}}{H_\mathrm{p}} = s_{PRP}^3 \int_{0}^{\infty} r_\mathrm{TE} e^{-2\lambda h}\lambda^2 \mathrm{J}_1(\lambda s)\mathrm{d}\lambda.


        References
        ----------
        Ward, Stanley H., and Gerald W. Hohmann. 1988. “4. Electromagnetic
        Theory for Geo-physical Applications.” 1. https://doi.org/10.1190/1.9781560802631.ch4
        """
        self.model = 1.0 / model  # resistivity to conductivity
        fop_resp = {}
        for freq_i in self.freqs:
            self.w = 2 * np.pi * freq_i
            # HCP
            for s_i in self.coilspacing_HCP:
                # Hz equation 4.46 Ward / (4*pi*rho**3)
                ht_i = self.ht_hcp.transform(  # Hankel transform
                    self.f_r_over_r,
                    s_i,
                    ret_err=False
                )
                Q_hsv_h0 = (s_i ** 3) * np.imag(ht_i)  # v: vertical
                I_hsv_h0 = (s_i ** 3) * np.real(ht_i)  # v: vertical

                # Save results into dictionary
                header_qv = f'Q_HCP_{int(freq_i)}_{s_i}'
                header_iv = f'I_HCP_{int(freq_i)}_{s_i}'
                fop_resp[header_qv] = -Q_hsv_h0 * 1e6
                fop_resp[header_iv] = -I_hsv_h0 * 1e6

            # PRP
            for s_i in self.coilspacing_PRP:
                # Hrho equation 4.45 Ward / (4*pi*rho**3)
                ht_i = self.ht_prp.transform(  # Hankel tranform
                    self.f_r_over_r,
                    s_i,
                    ret_err=False
                )
                Q_hsr_h0 = (s_i ** 3) * np.imag(ht_i)  # r: radial
                I_hsr_h0 = (s_i ** 3) * np.real(ht_i)  # r:radial

                # Save results into dictionary
                header_qr = f'Q_PRP_{int(freq_i)}_{s_i}'
                header_ir = f'I_PRP_{int(freq_i)}_{s_i}'
                fop_resp[header_qr] = -Q_hsr_h0 * 1e6
                fop_resp[header_ir] = -I_hsr_h0 * 1e6

        return fop_resp


