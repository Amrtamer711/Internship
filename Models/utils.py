import random
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import re

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences_df):
        self.sequences_df = sequences_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        inputs = self.sequences_df.iloc[idx]['inputs']
        targets = self.sequences_df.iloc[idx]['targets']
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()

class VisualizationTimeSeriesDataset(Dataset):
    def __init__(self, sequences_df):
        self.sequences_df = sequences_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        inputs = self.sequences_df.iloc[idx]['inputs']
        targets = self.sequences_df.iloc[idx]['targets']
        well_name = self.sequences_df.iloc[idx]['well_name']
        inputs_dates = self.sequences_df.iloc[idx]['inputs_dates']
        targets_dates = self.sequences_df.iloc[idx]['targets_dates']

        return well_name, inputs_dates, targets_dates, torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()

class UivariateTimeSeriesDataset(Dataset):
    def __init__(self, sequences_df):
        self.sequences_df = sequences_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        inputs = self.sequences_df.iloc[idx]['inputs']
        targets = self.sequences_df.iloc[idx]['targets']
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()

class UnivariateVisualizationTimeSeriesDataset(Dataset):
    def __init__(self, sequences_df):
        self.sequences_df = sequences_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        inputs = self.sequences_df.iloc[idx]['inputs']
        targets = self.sequences_df.iloc[idx]['targets']
        well_name = self.sequences_df.iloc[idx]['well_name']
        inputs_dates = self.sequences_df.iloc[idx]['inputs_dates']
        targets_dates = self.sequences_df.iloc[idx]['targets_dates']

        return well_name, inputs_dates, targets_dates, torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device="cpu"): # Function to estimate train and val loss while training
    model.eval()  # Set model to evaluation mode
    out = {}
    loaders = {'train': train_loader, 'val': val_loader}
    for mode in ['train', 'val']:
        losses = []
        # Iterate over the DataLoader for a fixed number of batches
        for X_batch, Y_batch in loaders[mode]:
            if device != "cpu":
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits, loss = model(X_batch, Y_batch)  # Run inputs and expected outputs through model to get loss
            losses.append(loss.item())  # Add value to losses list
        out[mode] = torch.tensor(losses).mean().item()  # Get mean of losses tensor and save it
    model.train()  # Put model back in train mode
    return out

def get_next_log_file(log_dir, prefix="training_run_"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        return os.path.join(log_dir, f"{prefix}1.txt")
    existing_files = os.listdir(log_dir)
    log_numbers = []
    for filename in existing_files:
        match = re.match(rf"{prefix}(\d+).txt", filename)
        if match:
            log_numbers.append(int(match.group(1)))
    if log_numbers:
        next_number = max(log_numbers) + 1
    else:
        next_number = 1
    return os.path.join(log_dir, f"{prefix}{next_number}.txt")

def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def calculate_metrics(y_true, y_pred):
    # Calculate metrics for the predictions
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    loss = mean_squared_error(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Loss': loss}

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_targets = []
    test_losses = []

    for X_batch, Y_batch in loader:
        if device != "cpu":
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        prediction, loss = model(X_batch, Y_batch)
        prediction = prediction.view(-1)
        Y_batch = Y_batch.view(-1)
        test_losses.append(loss.item())  # Collect test loss
        # Convert predictions and targets to NumPy arrays
        all_preds.append(prediction.cpu().numpy())  # .cpu() moves tensor to CPU, .numpy() converts to NumPy array
        all_targets.append(Y_batch.cpu().numpy())
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['Loss'] = np.mean(test_losses)
    print(f"Metrics on dataset:\n")
    print(f"Loss is {metrics['Loss']:.4f}")
    print(f"MAE is {metrics['MAE']:.4f}")
    print(f"RMSE is {metrics['RMSE']:.4f}")
    print(f"R² is {metrics['R2']:.4f}")
    print("\n")
    return metrics

def get_oil_history(data, date, well_name):
    well = data[data["well_name"] == int(well_name)].reset_index(drop=True)
    well = well[pd.to_datetime(well['date']) < date]
    return well['oil_rate']

def predictions_with_history_graphs(model, data, visualization_loader, predictions_days, path, device="cpu"):
    pdf = PdfPages(path + "/predictions_with_history.pdf")
    with torch.no_grad():
        for well_name, inputs_dates, targets_dates, X_batch, Y_batch in visualization_loader:
            choice = random.randint(0, len(X_batch) - 1)
            X_batch = X_batch[choice].unsqueeze(0)
            Y_batch = Y_batch[choice]
            well_name = well_name[choice]
            inputs_dates = [i[choice] for i in inputs_dates]
            targets_dates = [i[choice] for i in targets_dates]

            history = get_oil_history(data, pd.to_datetime(targets_dates[0]), well_name)
            if device != "cpu":
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            prediction, loss = model(X_batch, Y_batch)
            
            prediction = prediction.view(-1)
            Y_batch = Y_batch.view(-1)
            label = Y_batch.cpu().numpy()
            prediction = prediction.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(label, prediction))
            # Plotting
            forecast_index = range(len(history) + 1, len(history) + predictions_days + 1)
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(history) + 1), history, label='Input Values')
            plt.plot(forecast_index, prediction, label='Forecasted Values')
            plt.plot(forecast_index, label, label='Actual Values')
            # Determine the interval for ticks to have approximately 10 ticks
            tick_interval = max(10, round((len(history) + predictions_days + 1) / 10 / 10) * 10)
            # Generate tick positions
            tick_positions = np.arange(0, len(history) + predictions_days + 1, tick_interval)
            tick_positions[0] = 1  # Start tick at 1 instead of 0

            # Generate tick labels
            tick_labels = tick_positions
            plt.xticks(ticks=tick_positions, labels=tick_labels)
            plt.axvline(x=len(history), color='r', linestyle='--', label='Forecast Start')
            plt.xlabel('Day')
            plt.ylabel('Value')
            plt.title(f'Well: {well_name}, Loss: {loss:.4f}, RMSE = {rmse:.4f}')
            plt.legend()
            plt.grid(True)
            # Save the current figure to the PDF
            pdf.savefig()
            plt.close()
    pdf.close()

def predictions_graphs(model, visualization_loader, predictions_days, path, column_index=None, device="cpu"):
    pdf = PdfPages(path + "/predictions.pdf")
    with torch.no_grad():
        for well_name, inputs_dates, targets_dates, X_batch, Y_batch in visualization_loader:
            choice = random.randint(0, len(X_batch) - 1)
            X_batch = X_batch[choice].unsqueeze(0)
            Y_batch = Y_batch[choice]
            well_name = well_name[choice]
            if device != "cpu":
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            prediction, loss = model(X_batch, Y_batch)
            inputs = X_batch[0].cpu().numpy()
            if column_index is not None:
                inputs = inputs[:, column_index]
            prediction = prediction.view(-1)
            Y_batch = Y_batch.view(-1)
            label = Y_batch.cpu().numpy()
            prediction = prediction.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(label, prediction))
            # Plotting
            forecast_index = range(len(inputs) + 1, len(inputs) + predictions_days + 1)
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(inputs) + 1), inputs, marker='o', label='Input Values')
            plt.plot(forecast_index, prediction, marker='o', label='Forecasted Values')
            plt.plot(forecast_index, label, marker='o', label='Actual Values')
            plt.xticks(ticks=np.arange(1, len(inputs) + predictions_days + 1), labels=np.arange(1, len(inputs) + predictions_days + 1))
            plt.axvline(x=len(inputs), color='r', linestyle='--', label='Forecast Start')
            plt.xlabel('Day')
            plt.ylabel('Value')
            plt.title(f'Well: {well_name}, Loss: {loss:.4f}, RMSE: {rmse:.4f}')
            plt.legend()
            plt.grid(True)
            # Save the current figure to the PDF
            pdf.savefig()
            plt.close()
    pdf.close()

def all_predictions_graphs(model, data, data_test, days_window, predictions_days, path, well_info, test_size, device="cpu"):
    well_dataloaders = {}
    pdf = PdfPages(path + "/all_predictions.pdf")
    with torch.no_grad():
        for well_name in well_info.keys():
            well = data[data['well_name'] == well_name].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(well.index, well['oil_rate'], label='Well Production')
            ax.axvline(x=int((1 - test_size) * len(well) - days_window - 2), color='r', linestyle='--',)
            ax.legend()
            well_dataset = UnivariateVisualizationTimeSeriesDataset(data_test[data_test['well_name'] == well_name].reset_index(drop=True))
            well_dataloaders[well_name] = DataLoader(well_dataset, batch_size=1, shuffle=False)
            total_loss = 0
            total_rmse = 0
            for i, (_, inputs_dates, targets_dates, X_batch, Y_batch) in enumerate(well_dataloaders[well_name]):
                inputs_dates = np.array(inputs_dates).T[0]
                targets_dates = np.array(targets_dates).T[0]
                history = get_oil_history(data, pd.to_datetime(targets_dates[0]), well_name)
                if device != "cpu":
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                prediction, loss = model(X_batch, Y_batch)
                prediction = prediction.view(-1)
                Y_batch = Y_batch.view(-1)
                label = Y_batch.cpu().numpy()
                prediction = prediction.cpu().numpy()
                rmse = np.sqrt(mean_squared_error(label, prediction))
                total_loss += loss
                total_rmse += rmse
                forecast_index = range(len(history) + 1, len(history) + predictions_days + 1)
                ax.plot(forecast_index, prediction, label='Forecasted Values', color='y')
                if i == 0:
                    ax.legend()
            avg_loss = total_loss / len(well_dataloaders[well_name]) 
            avg_rmse = total_rmse / len(well_dataloaders[well_name]) 
            ax.set_xlabel('Day')
            ax.set_ylabel('Value')
            ax.set_title(f'Well: {well_name}, Loss: {avg_loss:.4f}, RMSE = {avg_rmse:.4f}')
            pdf.savefig(fig)
            plt.close()
    pdf.close()

def all_predictions_with_gaps_graphs(model, data, data_test, days_window, predictions_days, path, well_info, test_size, device="cpu"):
    well_dataloaders = {}   
    pdf = PdfPages(path + "/all_predictions_with_gaps.pdf")
    with torch.no_grad():
        for well_name in well_info.keys():
            well = data[data['well_name'] == well_name].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(well.index, well['oil_rate'], label='Well Production')
            ax.axvline(x=int((1 - test_size) * len(well) - days_window - 2), color='r', linestyle='--',)
            ax.legend()
            well_dataset = UnivariateVisualizationTimeSeriesDataset(data_test[data_test['well_name'] == well_name].reset_index(drop=True))
            well_dataloaders[well_name] = DataLoader(well_dataset, batch_size=1, shuffle=False)
            total_loss = 0
            total_rmse = 0
            for i, (_, inputs_dates, targets_dates, X_batch, Y_batch) in enumerate(well_dataloaders[well_name]):
                if i % predictions_days == 0 or i == len(well_dataloaders[well_name]) - 1:
                    inputs_dates = np.array(inputs_dates).T[0]
                    targets_dates = np.array(targets_dates).T[0]
                    history = get_oil_history(data, pd.to_datetime(targets_dates[0]), well_name)
                    if device != "cpu":
                        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    prediction, loss = model(X_batch, Y_batch)
                    prediction = prediction.view(-1)
                    Y_batch = Y_batch.view(-1)
                    label = Y_batch.cpu().numpy()
                    prediction = prediction.cpu().numpy()
                    rmse = np.sqrt(mean_squared_error(label, prediction))
                    total_loss += loss
                    total_rmse += rmse
                    forecast_index = range(len(history) + 1, len(history) + predictions_days + 1)
                    ax.plot(forecast_index, prediction, label='Forecasted Values', color='y')
                    if i == 0:
                        ax.legend()
            avg_loss = total_loss / len(well_dataloaders[well_name]) 
            avg_rmse = total_rmse / len(well_dataloaders[well_name]) 
            ax.set_xlabel('Day')
            ax.set_ylabel('Value')
            ax.set_title(f'Well: {well_name}, Loss: {avg_loss:.4f}, RMSE = {avg_rmse:.4f}')
            pdf.savefig(fig)
            plt.close()
    pdf.close()

def produce_graphs(model, visualization_test_loader, data, data_test, days_window, predictions_days, path, well_info, test_size, column_index=None, device="cpu"):
    predictions_with_history_graphs(model, data, visualization_test_loader, predictions_days, path, device=device)
    predictions_graphs(model, visualization_test_loader, predictions_days, path, column_index=column_index, device=device)
    all_predictions_graphs(model, data, data_test, days_window, predictions_days, path, well_info, test_size, device=device)
    all_predictions_with_gaps_graphs(model, data, data_test, days_window, predictions_days, path, well_info, test_size, device=device)

def rename(config_path, model_path, old_name, new_name):
    os.rename(config_path + f'{old_name}state_params.pth', config_path + f'{new_name}state_params.pth')
    os.rename(config_path + f'{old_name}state_optimizer.pth', config_path + f'{new_name}state_optimizer.pth')
    os.rename(config_path + f'{old_name}state_scheduler.pth', config_path + f'{new_name}state_scheduler.pth')
    os.rename(model_path + f'{old_name}complete_params.pth', model_path + f'{new_name}complete_params.pth')
    os.rename(model_path + f'{old_name}complete_optimizer.pth', model_path + f'{new_name}complete_optimizer.pth')
    os.rename(model_path + f'{old_name}complete_scheduler.pth', model_path + f'{new_name}complete_scheduler.pth')
    
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
                rename(config_path, model_path, old_name, new_name)
                print("Files have been renamed successfully.")
            except FileNotFoundError:
                print("One or more files not found.")
        elif user_choice == 2:
            print("Operation cancelled by the user.")
        else:
            print("Invalid choice. Operation cancelled.")
    else:
        try:
            rename(config_path, model_path, old_name, new_name)
            print("Files have been renamed successfully.")
        except FileNotFoundError:
            print("One or more files not found.")
