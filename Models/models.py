import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import torch.nn.functional as F
from chronos import ChronosPipeline
import torch.nn as nn
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.multioutput import MultiOutputRegressor
import os
import json
import numpy as np
import utils as ut



class Moirai(nn.Module):
    def __init__(self, size, input_length, output_length, num_samples):
        super().__init__()
        self.size = size
        self.output_length = output_length
        self.input_length = input_length
        self.num_samples = num_samples
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{self.size}"),
            prediction_length=self.output_length,
            context_length=self.input_length,
            patch_size="auto", # patch size: choose from {"auto", 8, 16, 32, 64, 128}
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0, # ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=0 # ds.num_past_feat_dynamic_real,
        )
    
    def forward(self, x, targets=None):
        B, T = x.shape
        predictor = self.model.create_predictor(batch_size=B)
        time_series_list = [{
            FieldName.TARGET: inputs.cpu().numpy().flatten().tolist(),
            FieldName.START: pd.Timestamp("2020-01-01")  # Same start date for all series; adjust if needed
        } for inputs in x]
        
        dataset_x = ListDataset(
            time_series_list,
            freq="1D"
        )
        
        forecasts = predictor.predict(dataset_x)
        outputs = torch.tensor(np.array([forecast.samples.mean(axis=0) for forecast in forecasts]))
        if targets is None:
            loss = None 
        else:
            logits = outputs.view(-1)
            targets = targets.view(-1)  # Ensure targets have the same shape as outputs
            loss = F.mse_loss(logits, targets)
        
        return outputs, loss



class Chronos(nn.Module):
    def __init__(self, size, output_length, num_samples, dtype=torch.bfloat16, device_map="mps"):
        super().__init__()
        self.size = size
        self.output_length = output_length
        self.num_samples = num_samples
        self.device_map = device_map
        self.dtype = dtype
        self.model = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.size}",
            device_map=self.device_map,
            torch_dtype=self.dtype # you can use torch.bfloat16 for faster inference or torch.float32 for more accuracy
        )
    
    def forward(self, x, targets=None):
        B, T = x.shape
        forecasts = self.model.predict(
                    context=x,
                    prediction_length=self.output_length,
                    num_samples=self.num_samples,
                    )
        outputs = forecasts.mean(axis=1)
        if targets is None:
            loss = None 
        else:
            logits = outputs.view(-1)
            targets = targets.view(-1)  # Ensure targets have the same shape as outputs
            loss = F.mse_loss(logits, targets)
        
        return outputs, loss
    


class XGBoost(nn.Module):
    def __init__(self, n_estimators, lr, max_depth, min_child_weight, subsample, colsample_bytree):
        super().__init__()
        # Create and configure the XGBoost regressor with hyperparameters
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.regressor = XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.lr, max_depth=self.max_depth, min_child_weight=self.min_child_weight, subsample=self.subsample, colsample_bytree=self.colsample_bytree) 
        self.model = MultiOutputRegressor(self.regressor)
    
    def fit(self, x, targets):
        B, T, C = x.shape
        x = x.view(B, -1).numpy()
        self.model.fit(x, targets)

    def save_model(self, config_path, name=""): # function to save parameters, optimizer and scheduler states
        if name != "":
            name = f'{name}_'
        joblib.dump(self.model, config_path + f'{name}XGBoost.lib')
    
    def load_model(self, config_path, name=""): 
        if name != "":
            name = f'{name}_'
        self.model = joblib.load(config_path + f'{name}XGBoost.lib')

    def forward(self, x, targets=None):
        B, T, C = x.shape
        x = x.view(B, -1).numpy()
        outputs = torch.tensor(self.model.predict(x)).contiguous()
        if targets is None:
            loss = None 
        else:
            logits = outputs.view(-1)
            targets = targets.view(-1)  # Ensure targets have the same shape as outputs
            loss = F.mse_loss(logits, targets)
        return outputs, loss
    


class LSTM(nn.Module):
    def __init__(self, input_length, hidden_length, output_length, num_layers, dropout=0):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        self.init_hidden = nn.Parameter(torch.zeros(num_layers, 1, hidden_length))
        self.init_cell = nn.Parameter(torch.zeros(num_layers, 1, hidden_length))
        self.num_layers = num_layers
        self.lstm_cell = nn.LSTM(self.input_length, self.hidden_length, self.num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_length, self.output_length)
        self.dropout_value = dropout
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x, targets=None):
        B, T, C = x.shape
        init_hidden = self.init_hidden.repeat(1, B, 1)
        init_cell = self.init_cell.repeat(1, B, 1)
        cell_output, _ = self.lstm_cell(x, (init_hidden, init_cell))
        cell_dropout = self.dropout(cell_output)
        outputs = self.output_layer(cell_dropout[:, -1, :])
        if targets is None:
            loss = None
        else:
            B, T = outputs.shape
            logits = outputs.view(-1)
            targets = targets.view(-1)
            loss = F.mse_loss(logits, targets)
        return outputs, loss
    
    def get_hyperparameters(self):
        return {'input_length': self.input_length, 'hidden_length': self.hidden_length, 'output_length': self.output_length, 'num_layers': self.num_layers, 'dropout': self.dropout_value}
    
    def save_params(self, config_path, model_path, optimizer=None, scheduler=None, name=""): # function to save parameters, optimizer and scheduler states
        config = self.get_hyperparameters()
        if name != "":
            name = f'{name}_'

        with open(os.path.join(config_path, f'{name}config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        torch.save(self.state_dict(), config_path + f'{name}state_params.pth') # to save parameters
        torch.save(self, model_path + f'{name}complete_params.pth')

        if optimizer is not None:
            optimizer_config = optimizer.defaults
            with open(os.path.join(config_path, f'{name}optimizer_config.json'), 'w') as f:
                json.dump(optimizer_config, f, indent=4)
            torch.save(optimizer.state_dict(), config_path + f'{name}state_optimizer.pth') # to save optimizer state
            torch.save(optimizer, model_path + f'{name}complete_optimizer.pth') 

        if scheduler is not None:
            torch.save(scheduler.state_dict(), config_path + f'{name}state_scheduler.pth') # to save scheduler states
            torch.save(scheduler, model_path + f'{name}complete_scheduler.pth')
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler_config = {
                    'scheduler_type': 'StepLR',
                    'step_size': scheduler.step_size,
                    'gamma': scheduler.gamma
                }
            elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
                scheduler_config = {
                    'scheduler_type': 'ExponentialLR',
                    'gamma': scheduler.gamma
                }
            else:
                raise ValueError("Unsupported scheduler type")
            with open(os.path.join(config_path, f'{name}scheduler_config.json'), 'w') as f:
                json.dump(scheduler_config, f, indent=4)
    
    @staticmethod
    def load_params(model_class, config_path, device, optimizer_class=None, scheduler_class=None, name=""):
        if name != "":
            name = f'{name}_'

        # Load model hyperparameters
        with open(os.path.join(config_path, f'{name}config.json'), 'r') as f:
            config = json.load(f)
        
        model = LSTM(**config)
        model.load_state_dict(torch.load(os.path.join(config_path, f'{name}state_params.pth')))
        
        optimizer = None
        if optimizer_class is not None:
            optimizer = optimizer_class(model.parameters())
            optimizer.load_state_dict(torch.load(os.path.join(config_path, f'{name}state_optimizer.pth')))

        scheduler = None
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer)
            scheduler.load_state_dict(torch.load(os.path.join(config_path, f'{name}state_scheduler.pth')))
            with open(os.path.join(config_path, f'{name}scheduler_config.json'), 'r') as f:
                scheduler_config = json.load(f)
                for param, value in scheduler_config['params'].items():
                    setattr(scheduler, param, value)
        
        return model, optimizer, scheduler
    
    @staticmethod
    def load_params(config_path, include_optimizer=None, include_scheduler=None, name=""):
        if name != "":
            name = f'{name}_'
        

        model = models.LSTM(input_length=input_length, hidden_length=vector_length, output_length=output_length, num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) # setting optimizer scheduler
        model.load_state_dict(torch.load(config_path + f'{name}state_params.pth'))

        optimizer = None
        if include_optimizer is not None:
            optimizer.load_state_dict(torch.load(config_path + f'{name}state_optimizer.pth'))
        
        scheduler = None
        if include_scheduler is not None:
            scheduler.load_state_dict(torch.load(os.path.join(config_path, f'{name}state_scheduler.pth')))
            with open(os.path.join(config_path, f'{name}scheduler_config.json'), 'r') as f:
                scheduler_config = json.load(f)
                if scheduler_config['scheduler_type'] == 'StepLR':
                    scheduler.step_size = scheduler_config['step_size']
                    scheduler.gamma = scheduler_config['gamma']
                elif scheduler_config['scheduler_type'] == 'ExponentialLR':
                    scheduler.gamma = scheduler_config['gamma']
                else:
                    raise ValueError("Unsupported scheduler type")
    
    @staticmethod
    def load_model(model_path, include_optimizer=None, include_scheduler=None, name=""):
        if name != "":
            name = f'{name}_'
        models = []
        print(os.path.join(model_path, f'{name}complete_params.pth'))
        model = torch.load(os.path.join(model_path, f'{name}complete_params.pth'))
        optimizer = None
        if include_optimizer is not None:
            optimizer = torch.load(os.path.join(model_path, f'{name}complete_optimizer.pth'))
            models.append(optimizer)
        scheduler = None
        if include_scheduler is not None:
            scheduler = torch.load(os.path.join(model_path, f'{name}complete_scheduler.pth'))
        return model, optimizer, scheduler

    @staticmethod
    def label_model_files(config_path, model_path, old_name="", new_name=""):
        if old_name != "":
            old_name = f'{old_name}_'
        if new_name != "":
            new_name = f'{new_name}_'
        if os.path.exists(config_path + f'{new_name}state_params.pth'):
            try:
                user_choice = int(input("File with new name already exists. Enter 1 if you want to overwrite the files. Otherwise press 2: "))
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
                return
            if user_choice == 1:
                try:
                    ut.rename(config_path, model_path, old_name, new_name)
                    print("Files have been renamed successfully.")
                except FileNotFoundError:
                    print("One or more files not found.")
            elif user_choice == 2:
                print("Operation cancelled by the user.")
            else:
                print("Invalid choice. Operation cancelled.")
        else:
            try:
                ut.rename(config_path, model_path, old_name, new_name)
                print("Files have been renamed successfully.")
            except FileNotFoundError:
                print("One or more files not found.")

