# Imports für Flask(API zum Frontend) und NumPy (Netzberechnungen)
from flask import Flask, request, jsonify, render_template
import numpy as np
# Imports für automatisches Öffnen des Browsers
import webbrowser
import threading

app = Flask(__name__)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(a):
    return a * (1 - a)

# Tanh
def tanh(x):
    return np.tanh(x)

def d_tanh(a):
    return 1 - a**2  

# ReLU
def relu(x):
    return np.maximum(0, x)

def d_relu(a):
    return np.where(a > 0, 1, 0)

# Route für die Startseite
@app.route('/')
def index():
    return render_template('index.html')

# Route für die API
@app.route('/train', methods=['POST'])
def train():
    try:
        # Eingabewerte aus dem Request extrahieren mit Fallbackwerten
        data = request.get_json()
        activation_type = data.get('activation', 'sigmoid')
        learning_rate = float(data.get('learning_rate', 0.1))
        epochs = int(data.get('epochs', 10000))
        hidden_layers_count = int(data.get('hidden_layers', 1))
        neurons_per_layer = int(data.get('neurons', 2))
        init_mode = data.get('init_mode', 'symmetric') 
        init_range = data.get('init_range', '0.5') 
    except Exception as e:
        return jsonify({'error': 'Ungültige Eingabewerte'}), 400

    # Funktionsauswahl
    if activation_type == 'relu':
        activation, d_activation = relu, d_relu
    elif activation_type == 'tanh':
        activation, d_activation = tanh, d_tanh
    else:
        activation, d_activation = sigmoid, d_sigmoid

    # XOR Inputs mit erwarteten Outputs
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = np.array([[0], [1], [1], [0]])

    # Architektur definieren: Input(2) -> Hidden... -> Output(1)
    layer_sizes = [2] + [neurons_per_layer] * hidden_layers_count + [1]
    
    # Gewichte und Biases initialisieren
    w = []
    b = []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        
        # Initialisierung des Bereich basierend auf den übergebenen Parametern
        try:
            r = float(init_range)
        except: # Fallback
            r = 0.5
        
        if init_mode == 'positive':
            w_layer = np.random.uniform(0, r, (n_in, n_out))
        elif init_mode == 'negative':
            w_layer = np.random.uniform(-r, 0, (n_in, n_out))
        else: # symmetric
            w_layer = np.random.uniform(-r, r, (n_in, n_out))
        
        w.append(w_layer)
        b.append(np.zeros((1, n_out)))

    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass (speichern der Aktivierungen für Backprop)
        activations = [xor_inputs]
        
        curr_a = xor_inputs
        for i in range(len(w)):
            z = curr_a @ w[i] + b[i]
            curr_a = activation(z)
            activations.append(curr_a)
            
        # Finaler Output
        final_output = activations[-1]
        # Loss berechnen
        loss = np.mean(1/2*(xor_outputs - final_output) ** 2) # Durchschnitt über alle Trainingsbeispiele
        
        # Verlust alle 1% der Epochen speichern
        if epoch % (max(1, epochs // 100)) == 0:
            loss_history.append({'epoch': epoch, 'loss': float(loss)})

        # Backpropagation
        delta = (final_output - xor_outputs) * d_activation(final_output)
        
        for i in reversed(range(len(w))):
            dW = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                # Delta für die vorherige Schicht berechnen
                delta = (delta @ w[i].T) * d_activation(activations[i])
            
            # Gewichte aktualisieren mit Gradientenabstiegsverfahren
            w[i] -= learning_rate * dW
            b[i] -= learning_rate * db

    # Entscheidungsebene berechnen für ein Raster von Punkten im Bereich [-0.2, 1.2] x [-0.2, 1.2]
    steps = 25 # Anzahl der Punkte pro Achse
    x_range = np.linspace(-0.2, 1.2, steps)
    y_range = np.linspace(-0.2, 1.2, steps)
    grid_points = np.array([[x, y] for x in x_range for y in y_range])
    
    curr_grid_a = grid_points
    for i in range(len(w)):
        curr_grid_a = activation(curr_grid_a @ w[i] + b[i])
        
    decision_data = []
    for i in range(len(grid_points)):
        decision_data.append({'x': float(grid_points[i][0]), 'y': float(grid_points[i][1]), 'v': float(curr_grid_a[i][0])})

    # Berechnete Werte zurückgeben an das Forntend
    return jsonify({
        'loss_history': loss_history,
        'final_loss': float(loss),
        'predictions': final_output.flatten().tolist(),
        'decision_boundary': decision_data,
    })

# Server starten
if __name__ == '__main__':
    # Browser automatisch öffnen
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")
    
    threading.Timer(1.0, open_browser).start()
    
    # Flask starten
    app.run(debug=True, port=5000)