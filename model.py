import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

class PINN(Model):
    def __init__(self, **kwargs):
        super(PINN, self).__init__(**kwargs)
        # Input layer
        self.input_layer = Input(shape=(25,))
        
        # First block
        self.dense1 = Dense(128, activation='relu', input_shape=(25,))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.2)
        
        # Second block
        self.dense2 = Dense(128, activation='relu')
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.2)
        
        # Third block
        self.dense3 = Dense(64, activation='relu')
        self.bn3 = BatchNormalization()
        self.dropout3 = Dropout(0.2)
        
        # Fourth block
        self.dense4 = Dense(64, activation='relu')
        self.bn4 = BatchNormalization()
        self.dropout4 = Dropout(0.2)
        
        # Fifth block
        self.dense5 = Dense(32, activation='relu')
        self.bn5 = BatchNormalization()
        
        # Output layer
        self.output_layer = Dense(1)

    def build(self, input_shape):
        super(PINN, self).build(input_shape)
        # The layers will be built automatically when called

    def call(self, inputs):
        # First block
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = self.dense4(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        
        # Fifth block
        x = self.dense5(x)
        x = self.bn5(x)
        
        # Output
        output = self.output_layer(x)
        return output

    def get_config(self):
        config = super(PINN, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def physics_loss(self, X, y_pred):
        # Implement the physical laws here

        # Euler-Bernoulli Beam Equation
        E = X[:, 9]  # Young's Modulus (Pa)
        I = X[:, 5]  # Cross-Sectional Moment of Inertia (m⁴)
        w = y_pred[:, 0]
        q = -X[:, 16] / X[:, 6]  # Applied Force (N) / Number of Strands
        beam_eq = E * I * w - q

        # Axial Stress
        A = X[:, 4]  # Cross-Sectional Area (m²)
        F = X[:, 16]  # Applied Force (N)
        axial_stress = F / A

        # Axial Deformation
        L = X[:, 2]  # Bridge Height (m)
        axial_deformation = (F * L) / (A * E)

        # Shear Modulus Relationship
        ν = X[:, 10]  # Poisson's Ratio
        G = E / (2 * (1 + ν))

        # Von Mises Yield Criterion
        σ_yield = X[:, 12]  # Tensile Yield Strength (Pa)
        σ_1 = X[:, 18]  # Max Principal Stress (Pa)
        σ_2 = X[:, 18]
        σ_3 = X[:, 18]
        von_mises = np.sqrt((σ_1 - σ_2)**2 + (σ_2 - σ_3)**2 + (σ_3 - σ_1)**2) / 2 - σ_yield

        # Hooke's Law
        ε = axial_deformation / L
        hookes_law = E * ε - axial_stress

        # Shear Stress-Strain Relationship
        τ = X[:, 18]  # Max Principal Stress (Pa)
        γ = τ / G
        shear_stress_strain = τ - G * γ

        # Poisson's Effect
        ε_lateral = -ν * ε

        # Principal Stresses
        σ_x = X[:, 18]
        σ_y = X[:, 18]
        τ_xy = τ
        principal_stresses = (σ_x + σ_y) / 2 + np.sqrt((σ_x - σ_y)**2 / 4 + τ_xy**2)

        # Force Equilibrium
        F_x = X[:, 16]
        F_y = X[:, 16]
        F_z = X[:, 16]
        force_eq = F_x + F_y + F_z

        # Moment Equilibrium
        M_x = X[:, 18]
        M_y = X[:, 18]
        M_z = X[:, 18]
        moment_eq = M_x + M_y + M_z

        # Torsional Shear Stress
        r = X[:, 3] / 2  # Cross-Sectional Diameter (m) / 2
        J = I  # Using Moment of Inertia as an approximation for Polar Moment of Inertia
        T = F * r  # Torque is force times radius
        torsional_shear_stress = np.divide(T * r, J, out=np.zeros_like(T * r), where=J!=0)

        # Angle of Twist
        angle_of_twist = np.divide(T * L, G * J, out=np.zeros_like(T * L), where=(G * J)!=0)

        # Euler Critical Load (Buckling)
        K = 1  # Effective length factor
        euler_critical_load = (np.pi**2 * E * I) / ((K * L)**2)

        # Lateral-Torsional Buckling Critical Moment
        I_y = I  # Moment of Inertia in y
        lateral_torsional_buckling = np.divide((np.pi / L) * np.sqrt(E * I_y * G * J), 1, out=np.zeros_like(E * I_y * G * J), where=L!=0)

        # Strain Energy Density
        strain_energy_density = (axial_stress * ε) / 2

        # Total physics loss
        physics_loss = tf.reduce_mean(tf.square(beam_eq)) + tf.reduce_mean(tf.square(axial_stress)) + \
                       tf.reduce_mean(tf.square(axial_deformation)) + tf.reduce_mean(tf.square(G)) + \
                       tf.reduce_mean(tf.square(von_mises)) + tf.reduce_mean(tf.square(hookes_law)) + \
                       tf.reduce_mean(tf.square(shear_stress_strain)) + tf.reduce_mean(tf.square(ε_lateral)) + \
                       tf.reduce_mean(tf.square(principal_stresses)) + tf.reduce_mean(tf.square(force_eq)) + \
                       tf.reduce_mean(tf.square(moment_eq)) + tf.reduce_mean(tf.square(torsional_shear_stress)) + \
                       tf.reduce_mean(tf.square(angle_of_twist)) + tf.reduce_mean(tf.square(euler_critical_load)) + \
                       tf.reduce_mean(tf.square(lateral_torsional_buckling)) + tf.reduce_mean(tf.square(strain_energy_density))

        return physics_loss

# Create the PINN model
model = PINN()

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# loss_history = []
# physics_loss_history = []
# data_loss_history = []

def train_step(X, y):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        data_loss = tf.keras.losses.MeanSquaredError()(y, predictions)
        physics_loss = model.physics_loss(X, predictions)
        total_loss = 0.7 * data_loss + 0.3 * physics_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Track losses
    loss_history.append(total_loss.numpy())
    physics_loss_history.append(physics_loss.numpy())
    data_loss_history.append(data_loss.numpy())

    return total_loss

# 