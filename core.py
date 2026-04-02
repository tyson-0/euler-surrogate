import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim


# local packages import
from pde import PDE
from network import SurrogateNetwork
from parse_data import read_csv_data, parse_yaml, split_input_output_df


class Euler():
    def __init__(self, path: str, physics_loss = False):
        problem_df = read_csv_data(path)
        struct = parse_yaml()
        input_df, output_df = split_input_output_df(struct, problem_df)
        self.physics_loss_flag = physics_loss
        self.input_df_columns = input_df.columns.tolist()
        self.vars = problem_df.columns.tolist()
        self.input_variables = len(self.input_df_columns)

        X_scaled, Y_scaled = self.scale_values(input_df, output_df)

        self.X_min = torch.tensor(self.X_scaler.data_min_, dtype=torch.float32)
        self.X_max = torch.tensor(self.X_scaler.data_max_, dtype=torch.float32)
        self.Y_min = torch.tensor(self.Y_scaler.data_min_, dtype=torch.float32)
        self.Y_max = torch.tensor(self.Y_scaler.data_max_, dtype=torch.float32)



        self.X_train = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
        self.Y_train = torch.tensor(Y_scaled, dtype=torch.float32)

        self._trained = False
        self.model = SurrogateNetwork(self.input_variables)

        self.PDE = PDE()
        self.pde_fn = None

    def set_pde(self, fn):
        self.pde_fn = fn


    def scale_values(self, input_df, output_df):
        self.X_scaler = MinMaxScaler()
        self.Y_scaler = MinMaxScaler()
        X_scaled = self.X_scaler.fit_transform(input_df)
        Y_scaled = self.Y_scaler.fit_transform(output_df.values.reshape(-1, 1))
        return X_scaled, Y_scaled

    
    def fit(self, epochs=5000, lambda_physics=0.01, callback=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5)

        physics_loss = torch.tensor(0.0)

        for _ in range(epochs):
            
            col_tensors = []
            vars = {}
            real_vars = {}

            for i, column in enumerate(self.input_df_columns):
                col = self.X_train[:, i:i+1].detach().requires_grad_(True)
                col_tensors.append(col)
                vars[column] = col
                real_vars[column] = col * (self.X_max[i] - self.X_min[i]) + self.X_min[i]

            X = torch.cat(col_tensors, dim=1)
            pred = self.model(X)

            data_loss = ((pred - self.Y_train)**2).mean()

            if self.physics_loss_flag:
                vars["u"] = pred
                real_vars["u"] = pred * (self.Y_max - self.Y_min) + self.Y_min

                residual = self.PDE.residual(self.pde_fn, vars, real_vars)
                physics_loss = (residual ** 2).mean()


            loss = data_loss + lambda_physics * physics_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())

            if _ % 500 == 0:
                print(f"epoch {_} | loss {loss.item():.4f}", flush=True)
                print(f"  data loss: {data_loss.item():.6f}", flush=True)
                print(f"  physics loss: {physics_loss.item():.6f}", flush=True)

            if callback and _ % 50 == 0:
                callback(_, loss.item(), data_loss.item(), physics_loss.item())

        self._trained = True


    def predict(self, input_list: list):
      if len(input_list) != len(self.input_df_columns):
          raise ValueError("input values should be equal to inputs given to the model.")

      input_df = pd.DataFrame([input_list], columns=self.input_df_columns)
      input_scaled = self.X_scaler.transform(input_df)

      with torch.no_grad():
          output = self.model(torch.tensor(input_scaled, dtype=torch.float32))

      output = self.Y_scaler.inverse_transform(output.detach().numpy())
      print(f"\npredicted: {output[0][0]}")
      return float(output[0][0])

    
    def save(self, path: str):
        if not self._trained:
            raise RuntimeError("Train the model first before saving.")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_columns": self.input_df_columns,
            "n_inputs": self.input_variables,
            "X_min": self.X_min,
            "X_max": self.X_max,
            "Y_min": self.Y_min,
            "Y_max": self.Y_max,
            "X_scaler": self.X_scaler,
            "Y_scaler": self.Y_scaler,
            "physics_loss_flag": self.physics_loss_flag
        }, path)

    @classmethod
    def from_saved(cls, path: str):
        checkpoint = torch.load(path, weights_only=False)
        instance = cls.__new__(cls)
        instance.input_df_columns = checkpoint["input_columns"]
        instance.input_variables = checkpoint["n_inputs"]
        instance.X_min = checkpoint["X_min"]
        instance.X_max = checkpoint["X_max"]
        instance.Y_min = checkpoint["Y_min"]
        instance.Y_max = checkpoint["Y_max"]
        instance.X_scaler = checkpoint["X_scaler"]
        instance.Y_scaler = checkpoint["Y_scaler"]
        instance.physics_loss_flag = checkpoint["physics_loss_flag"]
        instance.model = SurrogateNetwork(instance.input_variables)
        instance.model.load_state_dict(checkpoint["model_state"])
        instance.model.eval()
        instance.PDE = PDE()
        instance.pde_fn = None
        instance._trained = True
        return instance