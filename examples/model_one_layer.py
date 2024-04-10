import numpy as np

from emi1d.modeling import forward_models as fwd

# Input arguments
nlay = 1  # number of layers
h = 0.0  # height of the coils wrt ground
freq = [9000]  # frequency in Hz
rho = 113.0  # electrical resistivity in ohm-m
s = [8]  # source-receiver separation
dz = 0.0  # dummy value because dz is infinite

# Initialize modeling class
fwd_model = fwd.emi1d_forward(
    h,
    s,
    s,
    freq,
    nlay,
    dz
)

# Print response
model = rho * np.ones_like(fwd_model.dzIN)
resp = fwd_model.response(model)
print(f'Response dictionary: {resp}')

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

