import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
import os

# Import library functions
from vagru_lib import FastVAGRU, generate_fast_physics_data

# ==========================================
# DETERMINISTIC CONFIGURATION
# ==========================================
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
keras.utils.set_random_seed(42)

# ==========================================
# MODEL BUILDERS
# ==========================================
def build_baseline_gru(units=32, feature_dim=4):
    inp = keras.layers.Input(shape=(None, feature_dim))
    rnn = keras.layers.GRU(units, return_sequences=True)(inp)
    out = keras.layers.TimeDistributed(keras.layers.Dense(1))(rnn)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse', jit_compile=True)
    return model

def build_vagru(units=32, feature_dim=4):
    # Note: feature_dim + 1 because we concat dt
    inp = keras.layers.Input(shape=(None, feature_dim + 1))
    rnn = FastVAGRU(units)(inp)
    out = keras.layers.TimeDistributed(keras.layers.Dense(1))(rnn)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse', jit_compile=True)
    return model

# ==========================================
# BENCHMARK EXECUTION
# ==========================================
if __name__ == "__main__":
    print("-" * 70)
    print("EMPIRICAL BENCHMARK: VAGRU vs. STANDARD GRU")
    print("Framework: Continuous-Time Non-Stationary Trajectories")
    print("-" * 70)

    scenarios = ['noise', 'void', 'mixed']

    for sc in scenarios:
        print(f"\n[INFO] Initializing evaluation for regime: {sc.upper()}")
        
        # Load data using the library function
        X_train, Xc_train, y_train = generate_fast_physics_data(1500, 60, 4, sc)
        X_val, Xc_val, y_val = generate_fast_physics_data(500, 60, 4, sc)
        
        # 1. Baseline Execution
        keras.backend.clear_session()
        gru = build_baseline_gru()
        gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=35, batch_size=256, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        gru_loss = gru.evaluate(X_val, y_val, verbose=0)
        del gru
        gc.collect()
        
        # 2. VAGRU Execution
        keras.backend.clear_session()
        vagru = build_vagru()
        vagru.fit(Xc_train, y_train, validation_data=(Xc_val, y_val), epochs=35, batch_size=256, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        vagru_loss = vagru.evaluate(Xc_val, y_val, verbose=0)
        del vagru
        gc.collect()
        
        # 3. Log Results
        if vagru_loss < gru_loss:
            improvement = ((gru_loss - vagru_loss) / gru_loss) * 100
            print(f"  [RESULT] Standard GRU Loss : {gru_loss:.5f}")
            print(f"  [RESULT] Refined VAGRU Loss: {vagru_loss:.5f}")
            print(f"  [STATUS] VAGRU outperformed baseline by {improvement:.2f}%")
        else:
            print(f"  [RESULT] Standard GRU Loss : {gru_loss:.5f}")
            print(f"  [RESULT] Refined VAGRU Loss: {vagru_loss:.5f}")
            print("  [STATUS] Baseline parity not broken.")