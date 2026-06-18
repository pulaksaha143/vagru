import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==========================================
# VAGRU MODEL DEFINITIONS
# ==========================================

@tf.keras.utils.register_keras_serializable()
class FastVAGRUCell(keras.layers.Layer):
    """
    Implementation of the Volatility-Aware Gated Recurrent Unit (VAGRU).
    """
    def __init__(self, units, rho=0.1, gamma_init=0.01, tau_init=0.1, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rho = float(rho)
        self.gamma_init = gamma_init
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.state_size = [units, None, None]
        self.output_size = units

    def build(self, input_shape):
        self.feature_dim = input_shape[-1] - 1
        self.state_size = [self.units, self.feature_dim, self.feature_dim]
        
        def create_weight(name, shape, init='glorot_uniform'):
            return self.add_weight(shape=shape, initializer=init, name=name)

        # Temporal Rates
        self.gamma = create_weight('gamma', (self.units,), init=keras.initializers.Constant(self.gamma_init))
        self.tau_r = create_weight('tau_r', (1,), init=keras.initializers.Constant(self.tau_init))
        self.tau_z = create_weight('tau_z', (1,), init=keras.initializers.Constant(self.tau_init))
        self.tau_v = create_weight('tau_v', (1,), init=keras.initializers.Constant(self.tau_init))
        self.tau_s = create_weight('tau_s', (1,), init=keras.initializers.Constant(self.tau_init))

        # Weights
        self.W_r = create_weight('W_r', (self.feature_dim, self.units))
        self.U_r = create_weight('U_r', (self.units, self.units), init='orthogonal')
        self.b_r = create_weight('b_r', (self.units,), init='zeros')
        self.W_z = create_weight('W_z', (self.feature_dim, self.units))
        self.b_z = create_weight('b_z', (self.units,), init='zeros')
        self.W_h = create_weight('W_h', (self.feature_dim, self.units))
        self.U_h = create_weight('U_h', (self.units, self.units), init='orthogonal')
        self.b_h = create_weight('b_h', (self.units,), init='zeros')
        
        constrained_init = keras.initializers.TruncatedNormal(stddev=0.01)
        self.W_xv = create_weight('W_xv', (self.feature_dim, self.units), init=constrained_init)
        self.W_vv = create_weight('W_vv', (self.feature_dim, self.units), init=constrained_init)
        self.W_av = create_weight('W_av', (self.feature_dim, self.units), init=constrained_init)
        self.U_v  = create_weight('U_v', (self.units, self.units), init='orthogonal')
        self.b_v  = create_weight('b_v', (self.units,), init='zeros')
        self.built = True

    def call(self, inputs, states, training=None):
        h_prev, x_prev, v_prev = states
        x_t = inputs[..., :-1]
        dt = inputs[..., -1:]
        dt_safe = tf.clip_by_value(dt, self.epsilon, 5.0)
        
        gamma_safe = tf.abs(self.gamma) + self.epsilon
        d_t = self.rho + (1.0 - self.rho) * tf.exp(-gamma_safe * dt_safe)
        h_decayed = d_t * h_prev

        phi_r = 1.0 - tf.exp(-dt_safe / (tf.abs(self.tau_r) + self.epsilon))
        phi_z = 1.0 - tf.exp(-dt_safe / (tf.abs(self.tau_z) + self.epsilon))
        phi_v = 1.0 - tf.exp(-dt_safe / (tf.abs(self.tau_v) + self.epsilon))

        v_t = x_t - x_prev
        a_t = v_t - v_prev
        r_t = tf.sigmoid(tf.matmul(x_t, self.W_r) + tf.matmul(h_decayed, self.U_r) + phi_r * self.b_r)
        z_t = tf.sigmoid(tf.matmul(x_t, self.W_z) + phi_z * self.b_z)
        h_tilde = tf.tanh(tf.matmul(x_t, self.W_h) + r_t * tf.matmul(h_decayed, self.U_h) + self.b_h)

        psi_t = (tf.matmul(x_t, self.W_xv) + tf.matmul(v_t, self.W_vv) + 
                 tf.matmul(a_t, self.W_av) + tf.matmul(h_decayed, self.U_v) + phi_v * self.b_v)
        V_t = tf.nn.softplus(psi_t) + (1.0 - 0.693147)

        tau_s_safe = tf.abs(self.tau_s) + self.epsilon
        s_bar = 1.0 - tf.exp(-(dt_safe / tau_s_safe) * z_t * V_t)

        h_t = (1.0 - s_bar) * h_decayed + s_bar * h_tilde
        return h_t, [h_t, x_t, v_t]

    def get_initial_state(self, batch_size=None):
        return [tf.zeros([batch_size, self.units]), 
                tf.zeros([batch_size, self.feature_dim]), 
                tf.zeros([batch_size, self.feature_dim])]

class FastVAGRU(keras.layers.RNN):
    def __init__(self, units, **kwargs):
        super().__init__(FastVAGRUCell(units), return_sequences=True, return_state=False, **kwargs)

# ==========================================
# PHYSICS DATA GENERATOR
# ==========================================
def generate_fast_physics_data(n_samples=2000, seq_len=60, feature_dim=4, scenario='mixed'):
    np.random.seed(42)
    if scenario == 'noise':
        dt = np.random.uniform(0.005, 0.05, size=(n_samples, seq_len))
    elif scenario == 'void':
        dt = np.random.uniform(0.5, 3.0, size=(n_samples, seq_len))
    else:
        mask = np.random.rand(n_samples, seq_len) < 0.2
        dt = np.where(mask, np.random.uniform(2.0, 4.0), np.random.uniform(0.01, 0.1))
    
    dt = dt.astype(np.float32)
    t_time = np.cumsum(dt, axis=-1)
    latent = 0.5 * np.sin(2.0 * np.pi * 0.2 * t_time) + 0.3 * np.cos(2.0 * np.pi * 0.5 * t_time)
    if scenario == 'void':
        latent += 0.4 * np.sin(2.0 * np.pi * 0.05 * t_time)
    latent += 0.02 * np.random.randn(*t_time.shape)
    
    X = np.zeros((n_samples, seq_len, feature_dim), dtype=np.float32)
    for i in range(feature_dim):
        X[..., i] = latent * np.sin(t_time + (i * 0.5))
        
    y = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
    y[:, :-1, 0] = X[:, 1:, 0]
    y[:, -1, 0] = X[:, -1, 0]
    X_combined = np.concatenate([X, dt[..., np.newaxis]], axis=-1)
    return X, X_combined, y