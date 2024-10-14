#app/__init__.py
from flask import Flask
from flask_cors import CORS
import os
import pickle
from model.model import Model
from app.config import Config
import pandas as pd
#from .database import init_db




def create_app():
    
    # Load the model
    
    file_name = 'model.pkl'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    file_path = os.path.join(base_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            
        print("Model exists and its loaded.")
        
    else:
        # Get the current working directory
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        data_file_path = os.path.join(current_file_path, '..', 'data', 'processed_data.csv')
        
        # Now read the CSV
        df = pd.read_csv(data_file_path)
        
        y_fit = df['churn']
        X_fit = df.drop('churn', axis = 1)
    
        model = Model()
        model.fit_model(X_fit = X_fit, y_fit = y_fit, save_model = True)
        
        print("Model was fitted and saved")

        #model.evaluate_model(X_fit, y_fit)
    
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    
    #Saving the model
    
    app.config['MODEL'] = model
    
    # Inicializa o banco de dados
    #init_db(app)
    
    # Registra as rotas
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app



app = create_app()

if __name__ == '__main__':
    # Executa o servidor Flask
    app.run(debug=True, host='0.0.0.0')