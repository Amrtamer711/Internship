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
from sklearn.model_selection import train_test_split

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

class UnivariateTimeSeriesDataset(Dataset):
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

class UnivariateTimeSeriesHistoryDataset(Dataset):
    def __init__(self, sequences_df, original_df):
        self.sequences_df = sequences_df
        self.original_df = original_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        well_name = self.sequences_df.iloc[idx]['well_name']
        inputs = get_oil_history(self.original_df, pd.to_datetime(self.sequences_df.iloc[idx]['targets_dates'][0]), well_name)
        targets = self.sequences_df.iloc[idx]['targets']
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()
    
class UnivariateVisualizationTimeSeriesHistoryDataset(Dataset):
    def __init__(self, sequences_df, original_df):
        self.sequences_df = sequences_df
        self.original_df = original_df

    def __len__(self):
        return len(self.sequences_df)

    def __getitem__(self, idx):
        well_name = self.sequences_df.iloc[idx]['well_name']
        inputs = get_oil_history(self.original_df, pd.to_datetime(self.sequences_df.iloc[idx]['targets_dates'][0]), well_name)
        targets = self.sequences_df.iloc[idx]['targets']
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

def produce_univariate_dataset(data, input_length, output_length, univariate_feature, well_info):
    sequences = []
    for well_name in well_info.keys():
        well_data = data[data['well_name'] == well_name].sort_values(by='date').reset_index()
        for i in range(len(well_data) - (input_length + output_length) + 1):
            inputs = well_data.iloc[i:i + input_length][univariate_feature].values
            targets = well_data.iloc[i + input_length:i + input_length + output_length][univariate_feature].values
            inputs_dates = list(well_data.iloc[i:i + input_length]['date'])
            targets_dates = list(well_data.iloc[i + input_length:i + input_length + output_length]['date'])
            inputs_index = list(well_data.iloc[i:i + input_length]['index'])
            targets_index = list(well_data.iloc[i + input_length:i + input_length + output_length]['index'])
            sequences.append((well_name, inputs_dates, targets_dates, inputs_index, targets_index, inputs, targets))
            if len(well_data.iloc[i:i + input_length + output_length]['well_name'].unique()) > 1:
                print('violation')
    dataset = pd.DataFrame(sequences, columns=['well_name', 'inputs_dates', 'targets_dates', 'inputs_index', 'targets_index', 'inputs', 'targets', ])
    return dataset

def produce_multivariate_dataset(data, input_length, output_length, single_feature_columns, multiple_feature_columns, target_column, well_info):
    sequences = []
    for well_name in well_info.keys():
        well_data = data[data['well_name'] == well_name].sort_values(by='date').reset_index()
        for i in range(len(well_data) - (input_length + output_length) + 1):
            single_features = np.array(well_data.iloc[i:i + input_length][single_feature_columns])
            multiple_features = np.array(well_data.iloc[i:i + input_length][multiple_feature_columns])
            multiple_flattened = []
            for k in multiple_features:
                flattened_list = [item for sublist in k for item in sublist]
                multiple_flattened.append(flattened_list)
            multiple_flattened = np.array(multiple_flattened)
            inputs = np.concatenate((single_features, multiple_flattened), axis=1)
            targets = well_data.iloc[i + input_length:i + input_length + output_length][target_column].values
            inputs_dates = list(well_data.iloc[i:i + input_length]['date'])
            targets_dates = list(well_data.iloc[i + input_length:i + input_length + output_length]['date'])
            inputs_index = list(well_data.iloc[i:i + input_length]['index'])
            targets_index = list(well_data.iloc[i + input_length:i + input_length + output_length]['index'])
            sequences.append((well_name, inputs_dates, targets_dates, inputs_index, targets_index, inputs, targets))
            if len(well_data.iloc[i:i + input_length + output_length]['well_name'].unique()) > 1:
                print('violation')
    dataset = pd.DataFrame(sequences, columns=['well_name', 'inputs_dates', 'targets_dates', 'inputs_index', 'targets_index', 'inputs', 'targets'])
    return dataset

def split_dataset(dataset, val_size, test_size, well_info, shuffle_train=True):
    test_size_param1 = test_size
    test_size_param2 = val_size / (1 - test_size_param1)
    train_size = 1 - (val_size + test_size)
    train_wells = []
    val_wells = []
    test_wells = []
    if shuffle_train:
        for i in well_info.keys():
            well = dataset[dataset['well_name'] == i]
            train_set_length = int((train_size + val_size) * len(well))
            train_wells.append(well[:train_set_length + 1])
            test_set_length = int(test_size * len(well))
            test_wells.append(well[-test_set_length:])
        # Flatten the lists of DataFrames
        temp_data = pd.concat(train_wells).reset_index(drop=True)
        data_test = pd.concat(test_wells).reset_index(drop=True)
        data_train, data_val = train_test_split(temp_data, test_size=test_size_param2, stratify=temp_data['well_name'], shuffle=True)
    else:
        for i in well_info.keys():
            well = dataset[dataset['well_name'] == i]
            train_set_length = int((train_size) * len(well))
            train_wells.append(well[:train_set_length + 1])
            val_set_length = int(val_size * len(well))
            val_wells.append(well[train_set_length:train_set_length + val_set_length])
            test_set_length = int(test_size * len(well))
            test_wells.append(well[-test_set_length:])

    data_train.reset_index(drop=True, inplace=True)
    data_val.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)
    return data_train, data_val, data_test

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


