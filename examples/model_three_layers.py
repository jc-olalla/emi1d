import numpy as np
from emi1d.modeling import forward_models as fwd

# Input arguments
nlay = 3  # number of layers
h = 0.0  # height of the coils wrt ground
freq = [1000, 9000, 20000, 30000, 40000]  # frequency in Hz
rho = [1, 20, 50]  # electrical resistivity in ohm-m
s_hcp = [1, 1.66, 2, 4, 8]  # source-receiver separation
s_prp = []  # source-receiver separation
dz = 1.0  # dummy value because dz is infinite

# Initialize modeling class
fwd_model = fwd.emi1d_forward(
    h,
    s_hcp,
    s_prp,
    freq,
    nlay,
    dz
)
# fwd_model.dzIN = np.array([1000.0, 10, 1])  # custom grid

# Make model and show
model = rho * np.ones_like(fwd_model.dzIN)

# Calculate response
resp = fwd_model.response(model)

# Show model grid
fwd_model.print_grid(model)


# Calculate apparent resitivity from response
for key in resp:
    if key.startswith('Q_'):
        meta = key.split(sep="_")
        rho_app = fwd.calc_rho_app(
            resp[key],
            float(meta[-1]),  # source-receiver separation
            float(meta[-2]),  # frequency in Hz
        )
        print(f'Apparent resistivity in ohm-m for {key}: {rho_app}')

