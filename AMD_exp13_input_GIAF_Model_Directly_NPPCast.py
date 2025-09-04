"""
Experiment 5 Code for NPP Prediction (Directly Using OBS Data with GIAF-trained Models)

Key Steps:
1. Data Loading & MinMax Scaling
2. Loading Pre-trained Models (UNet, VNet, AttUNet, R2UNet)
3. Applying Models Directly on Observational Data (CbPM, EVGPM, SVGPM, Mean OBS)
4. Predicting Output (Last 3 years or Full time span), Inverse Scaling, Saving Results

Author: Bizhi Wu (2025.06.16)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import mat73
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras_unet_collection import models
import NPPCast

# -----------------------------------------------------------------------------
# 1. Environment Variables & Hyperparameters
# -----------------------------------------------------------------------------

# AMD GPU: set visible devices
# os.environ['HIP_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# TF environment settings
# os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '12288'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Training and data shape parameters
BATCH_SIZE = 12
SMOOTH = 1e-9
EPOCHS = 100
SAVE_MODEL = True
TRAINING = False

INPUT_FRAMES =  36       # 3 years monthly
OUTPUT_FRAMES = 36       # 3 years monthly
WIDTH = 384
HEIGHT = 320

# Date settings
START_DATE = "1/1/2002"

# Directory for loading/saving
SAVE_DIR_MODELS = "./AMD_Archives/Model_UNet_family_GIAF/"
SAVE_DIR_OUTPUTS = "./AMD_Archives/npp_outputs/OBS_Input_GIAF_Directly/"
SAVE_DIR_SHOWCASE = os.path.join(SAVE_DIR_OUTPUTS, "ShowCases")

# Make sure directories exist
os.makedirs(SAVE_DIR_OUTPUTS, exist_ok=True)
os.makedirs(SAVE_DIR_SHOWCASE, exist_ok=True)

# Global MinMaxScaler (to keep consistent with experiment1 usage)
MinMax_scaler = MinMaxScaler()

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def CESM_MaxMin(CESM_Data: np.ndarray):
    """
    Fit a MinMaxScaler on the input 3D data (time, lat, lon),
    then transform it and reshape back to (time, lat, lon).

    Args:
        CESM_Data (np.ndarray): shape [time, 384, 320].

    Returns:
        MinMax_scaler (MinMaxScaler): Fitted scaler.
        MinMax_CESM (np.ndarray): Scaled data, same shape as input.
    """
    time_dim = len(CESM_Data)
    reshaped = CESM_Data.reshape([time_dim, -1])  # (time, lat*lon)
    MinMax_scaler.fit(reshaped)
    scaled = MinMax_scaler.transform(reshaped)
    scaled_3d = scaled.reshape([time_dim, WIDTH, HEIGHT])
    return MinMax_scaler, scaled_3d

def get_Input_Output(input_frames: int, output_frames: int, data_3d: np.ndarray):
    """
    Create input (X) and output (Y) sequences from the scaled data.

    Args:
        input_frames (int): Number of historical frames (months) as input.
        output_frames (int): Number of future frames (months) to predict.
        data_3d (np.ndarray): shape [time, WIDTH, HEIGHT].

    Returns:
        INPUT_SEQUENCE (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].
        OUTPUT_SEQUENCE (np.ndarray): shape [N, WIDTH, HEIGHT, output_frames].
    """
    total_time = len(data_3d)
    n_samples = total_time - input_frames - output_frames

    X = np.zeros((n_samples, input_frames, WIDTH, HEIGHT))
    Y = np.zeros((n_samples, output_frames, WIDTH, HEIGHT))

    for i in range(n_samples):
        X[i] = data_3d[i : i + input_frames, :, :]
        Y[i] = data_3d[i + input_frames : i + input_frames + output_frames, :, :]

    # Transpose to [N, WIDTH, HEIGHT, frames]
    X = X.transpose(0, 2, 3, 1)
    Y = Y.transpose(0, 2, 3, 1)
    return X, Y

def get_full_Input(input_frames: int, data_3d: np.ndarray):
    """
    Generate a sliding-window input from time=0 to time_end, 
    ignoring the last frames that can't form a full input sequence.

    Args:
        input_frames (int): Number of historical frames as input.
        data_3d (np.ndarray): shape [time, WIDTH, HEIGHT].

    Returns:
        FULL_INPUT_SEQUENCE (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].
    """
    total_time = len(data_3d)
    n_samples = total_time - input_frames

    X_full = np.zeros((n_samples, input_frames, WIDTH, HEIGHT))
    for i in range(n_samples):
        X_full[i] = data_3d[i : i + input_frames, :, :]

    # Transpose to [N, WIDTH, HEIGHT, frames]
    X_full = X_full.transpose(0, 2, 3, 1)
    return X_full

def making_Prediction(model: tf.keras.Model, input_data: np.ndarray):
    """
    Use the model to predict, splitting in batches of size 12 if needed.

    Args:
        model (tf.keras.Model): Keras model for inference.
        input_data (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].

    Returns:
        np.ndarray: Concatenated predictions from the model.
                    shape [N, WIDTH, HEIGHT, output_frames].
    """
    Pred_list = []
    gap = len(input_data)
    for idx in range(gap // 12):
        batch_data = input_data[idx * 12 : (idx + 1) * 12]
        batch_pred = model.predict(batch_data)
        Pred_list.append(batch_pred)
    # No remainder handling here because original code doesn't handle partial
    # if there's leftover. But you can add it if needed.

    ML_Forecasting = np.concatenate(Pred_list, axis=0)
    return ML_Forecasting

def ssim(x, y, max_val=1.0):
    """TF built-in SSIM."""
    return tf.image.ssim(x, y, max_val)

def psnr(x, y, max_val=1.0):
    """TF built-in PSNR."""
    return tf.image.psnr(x, y, max_val)

def POD(x, y):
    """
    Probability of Detection (Recall).
    x: Ground truth
    y: Prediction
    """
    y_pos = K.clip(x, 0, 1)
    y_pred_pos = K.clip(y, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + SMOOTH) / (tp + fn + SMOOTH)

def FAR(x, y):
    """
    False Alarm Rate.
    x: Ground truth
    y: Prediction
    """
    y_pred_pos = K.clip(y, 0, 1)
    y_pos = K.clip(x, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    return fp / (tp + fp + SMOOTH)

def Inversing_the_PredData(
    start_date: str = "1/1/2002",
    pred_npy: np.ndarray = None,
    month_mat: str = "./OBS_NPP_CbPM_month.mat",
    var_names: str = 'OBS_NPP_CbPM_month'
):
    """
    Inverse the MinMax scaling for predicted data and create an xarray Dataset.

    Args:
        start_date (str): The start date for xarray time coordinate.
        pred_npy (np.ndarray): Predicted data, shape [N, lat, lon, time].
        month_mat (str): Path to .mat file containing original monthly data.
        var_names (str): Variable name inside .mat file.

    Returns:
        (CbPM_Pred_Data, pred_last_10y, real_last_10y, diff_last_10y, ds):
            CbPM_Pred_Data (np.ndarray): shape [N, lat, lon, time].
            pred_last_10y (np.ndarray): shape [time, lat, lon].
            real_last_10y (np.ndarray): shape [time, lat, lon].
            diff_last_10y (np.ndarray): shape [time, lat, lon].
            ds (xr.Dataset): xarray dataset storing the predicted data over time.
    """
    # 1. Load ocean mask
    # spNPP = xr.open_dataset("../NPP_Data/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.photoC_diaz.208101-210012.nc")
    # mask = (spNPP['REGION_MASK'].values > 0).astype(np.int32)
    region_mask = xr.open_dataset('./Common_data/CESM_Parameter.nc')['REGION_MASK'].values
    mask = (region_mask > 0).astype(np.int32)

    # 2. Load original OBS data
    obs_data = mat73.loadmat(month_mat)
    obs_npp = obs_data[var_names]  # shape [lat, lon, time], presumably
    obs_npp = obs_npp.transpose(2, 1, 0)  # match [time, lat, lon]
    obs_npp = np.nan_to_num(obs_npp)

    # 3. Scale (MinMax) to get data_min_ & data_range_
    local_scaler, scaled_npp = CESM_MaxMin(obs_npp)
    data_min = local_scaler.data_min_
    data_range = local_scaler.data_range_

    # 4. Expand data_min / data_range / mask to match predicted shape
    # pred_npy shape: [N, lat, lon, time]
    # data_min shape: [lat*lon]
    data_min_reshaped = data_min.reshape(WIDTH, HEIGHT)
    data_range_reshaped = data_range.reshape(WIDTH, HEIGHT)

    n_samples = len(pred_npy)
    n_time = pred_npy.shape[-1]

    # Expand data_min to [N, lat, lon, time]
    min_4d = np.array([[data_min_reshaped] * n_time] * n_samples)
    min_4d = np.transpose(min_4d, (0, 2, 3, 1))

    # Expand data_range
    range_4d = np.array([[data_range_reshaped] * n_time] * n_samples)
    range_4d = np.transpose(range_4d, (0, 2, 3, 1))

    # Expand mask
    mask_4d = np.array([[mask] * n_time] * n_samples)
    mask_4d = np.transpose(mask_4d, (0, 2, 3, 1))

    # 5. Inverse scaling
    masked_pred = pred_npy * mask_4d
    pred_data = masked_pred * range_4d + min_4d

    # 6. Build xarray dataset
    lat_ds = xr.open_dataset("./NPP_Data/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.photoC_diaz.208101-210012.nc")
    lat_vals = lat_ds["nlat"].values
    lon_vals = lat_ds["nlon"].values

    # Time coords
    time_coords = np.arange(1, 1 + n_time)   # e.g. 1..120
    start_dates = pd.date_range(start=start_date, periods=n_samples, freq='M')

    ds = xr.Dataset(
        data_vars=dict(
            ML_Pred_OBS_NPP=(["start_date", "nlat", "nlon", "time"], pred_data),
        ),
        coords=dict(
            start_date=(["start_date"], start_dates),
            nlat=(["nlat"], lat_vals),
            nlon=(["nlon"], lon_vals),
            time=(["time"], time_coords)
        ),
        attrs=dict(description="NPP Prediction from OBS input using GIAF-trained model.")
    )

    # 7. Extract last 10 years from predictions (the last sample's timeslice)
    pred_last_10y = pred_data[-1]  # shape [lat, lon, time]
    # rearrange to [time, lat, lon]
    pred_last_10y = pred_last_10y.transpose(2, 0, 1)

    # Real data's last 10-year portion (120 months from the end)
    real_last_10y = obs_npp[-OUTPUT_FRAMES:]  # shape [120, lat, lon]

    diff_last_10y = pred_last_10y - real_last_10y  # [time, lat, lon]

    return pred_data, pred_last_10y, real_last_10y, diff_last_10y, ds

# -----------------------------------------------------------------------------
# 3. Main Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------
    # 3.1 Load & Preprocess OBS Data
    # ---------------------------
    print("[INFO] CbPM Preprocessing...")
    obs_cbpm = mat73.loadmat("./NPP_MAT/CbPM_month_intp.mat")['CbPM_month_intp']
    obs_cbpm = np.nan_to_num(obs_cbpm.transpose(2, 1, 0))
    cbpm_scaler, cbpm_scaled = CESM_MaxMin(obs_cbpm)
    cbpm_input, cbpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, cbpm_scaled)
    print("CbPM input shape:", cbpm_input.shape, "output shape:", cbpm_output.shape)

    print("[INFO] EVGPM Preprocessing...")
    obs_evgpm = mat73.loadmat("./NPP_MAT/EVGPM_month_intp.mat")['EVGPM_month_intp']
    obs_evgpm = np.nan_to_num(obs_evgpm.transpose(2, 1, 0))
    evgpm_scaler, evgpm_scaled = CESM_MaxMin(obs_evgpm)
    evgpm_input, evgpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, evgpm_scaled)
    print("EVGPM input shape:", evgpm_input.shape, "output shape:", evgpm_output.shape)

    print("[INFO] SVGPM Preprocessing...")
    obs_svgpm = mat73.loadmat("./NPP_MAT/SVGPM_month_intp.mat")['SVGPM_month_intp']
    obs_svgpm = np.nan_to_num(obs_svgpm.transpose(2, 1, 0))
    svgpm_scaler, svgpm_scaled = CESM_MaxMin(obs_svgpm)
    svgpm_input, svgpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, svgpm_scaled)
    print("SVGPM input shape:", svgpm_input.shape, "output shape:", svgpm_output.shape)

    print("[INFO] MEAN OBS Preprocessing...")
    obs_mean = mat73.loadmat("./NPP_MAT/NPP_month_mean_intp.mat")['NPP_month_mean_intp']
    obs_mean = np.nan_to_num(obs_mean.transpose(2, 1, 0))
    mean_scaler, mean_scaled = CESM_MaxMin(obs_mean)
    mean_input, mean_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, mean_scaled)
    print("Mean input shape:", mean_input.shape, "output shape:", mean_output.shape)

    # ---------------------------
    # 3.2 Define Model Variants & Load Weights
    # ---------------------------
    print("[INFO] Loading Pre-trained Models...")

    unet_2d = models.unet_2d(
        (WIDTH, HEIGHT, INPUT_FRAMES),
        [32, 64, 128, 256, 512, 512, 1024],
        n_labels=OUTPUT_FRAMES,
        stack_num_down=2,
        stack_num_up=1,
        activation='GELU',
        output_activation=None,
        batch_norm=True,
        pool='max',
        unpool='nearest',
        name='unet'
    )

    vnet_2d = models.vnet_2d(
        (WIDTH, HEIGHT, INPUT_FRAMES),
        filter_num=[32, 64, 128, 256, 512, 512, 1024],
        n_labels=OUTPUT_FRAMES,
        res_num_ini=1,
        res_num_max=3,
        activation='PReLU',
        output_activation=None,
        batch_norm=True,
        pool=False,
        unpool=False,
        name='vnet'
    )

    attunet_2d = models.att_unet_2d(
        (WIDTH, HEIGHT, INPUT_FRAMES),
        [32, 64, 128, 256, 512, 512, 1024],
        n_labels=OUTPUT_FRAMES,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        atten_activation='ReLU',
        attention='add',
        output_activation=None,
        batch_norm=True,
        pool=False,
        unpool='bilinear',
        name='attunet'
    )

    r2unet_2d = models.r2_unet_2d(
        (WIDTH, HEIGHT, INPUT_FRAMES),
        [32, 64, 128, 256, 512, 1024],
        n_labels=OUTPUT_FRAMES,
        stack_num_down=1,
        stack_num_up=1,
        recur_num=1,
        activation='ReLU',
        output_activation=None,
        batch_norm=True,
        pool='max',
        unpool='nearest',
        name='r2unet'
    )

    # NPP_unet = NPPCast.get_model(384, 320, INPUT_FRAMES, OUTPUT_FRAMES, is_classification=False)
    # model_name_list = ['UNet', 'VNet', 'AttUNet', 'R2UNet','NPPCast']
    # model_dict = {
    #     'UNet': unet_2d,
    #     'VNet': vnet_2d,
    #     'AttUNet': attunet_2d,
    #     'R2UNet': r2unet_2d,
    #     'NPPCast':NPP_unet
    # }
    # model_name_list = ['NPPCast']
    # model_dict = {'NPPCast':NPP_unet}
    model_name_list = ['UNet', 'VNet', 'AttUNet', 'R2UNet']
    model_dict = {
        'UNet': unet_2d,
        'VNet': vnet_2d,
        'AttUNet': attunet_2d,
        'R2UNet': r2unet_2d
    }

    # ---------------------------
    # 3.3 Predict for each model & each dataset
    # ---------------------------
    for model_key in model_name_list:
        tf.keras.backend.clear_session()
        current_model = model_dict[model_key]
        model_name_str = f"GIAF_{model_key}"

        # Load weights
        weight_file = os.path.join(SAVE_DIR_MODELS, model_name_str + ".h5")
        print(f"[INFO] Loading weights from {weight_file}")
        current_model.load_weights(weight_file)

        # ---------------------------
        # 3.3.1 Predict: CbPM
        # ---------------------------
        print(f"[INFO] Predicting on CbPM data with {model_name_str}...")
        # cbpm_pred = making_Prediction(current_model, cbpm_input)
        cbpm_pred = current_model.predict(cbpm_input)
        np.save(os.path.join(SAVE_DIR_OUTPUTS, f"CbPM_Input_GIAF_MinMax{model_name_str}.npy"), cbpm_pred)

        (cbpm_total_pred,
         cbpm_pred_data,
         cbpm_real_data,
         cbpm_diff_data,
         cbpm_ds) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=cbpm_pred,
            month_mat="./NPP_MAT/CbPM_month_intp.mat",
            var_names='CbPM_month_intp'
        )
        # Save results
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"CbPM_Input_GIAF_Pred{model_name_str}.npy"), cbpm_total_pred)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"CbPM_Pred_10year{model_name_str}.npy"), cbpm_pred_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"CbPM_Real_10year{model_name_str}.npy"), cbpm_real_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"CbPM_Diff_10year{model_name_str}.npy"), cbpm_diff_data)
        cbpm_ds.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"CbPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of CbPM_Total_Pred:", cbpm_total_pred.shape)
        del cbpm_total_pred, cbpm_pred_data, cbpm_real_data, cbpm_diff_data, cbpm_ds

        # ---------------------------
        # 3.3.2 Predict: EVGPM
        # ---------------------------
        print(f"[INFO] Predicting on EVGPM data with {model_name_str}...")
        # evgpm_pred = making_Prediction(current_model, evgpm_input)
        evgpm_pred = current_model.predict(evgpm_input)
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"EVGPM_Input_GIAF_MinMax{model_name_str}.npy"), evgpm_pred)

        (evgpm_total_pred,
         evgpm_pred_data,
         evgpm_real_data,
         evgpm_diff_data,
         evgpm_ds) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=evgpm_pred,
            month_mat="./NPP_MAT/EVGPM_month_intp.mat",
            var_names='EVGPM_month_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"EVGPM_Input_GIAF_Pred{model_name_str}.npy"), evgpm_total_pred)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"EVGPM_Pred_10year{model_name_str}.npy"), evgpm_pred_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"EVGPM_Real_10year{model_name_str}.npy"), evgpm_real_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"EVGPM_Diff_10year{model_name_str}.npy"), evgpm_diff_data)
        evgpm_ds.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"EVGPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of EVGPM_Total_Pred:", evgpm_total_pred.shape)
        del evgpm_total_pred, evgpm_pred_data, evgpm_real_data, evgpm_diff_data, evgpm_ds

        # ---------------------------
        # 3.3.3 Predict: SVGPM
        # ---------------------------
        print(f"[INFO] Predicting on SVGPM data with {model_name_str}...")
        # svgpm_pred = making_Prediction(current_model, svgpm_input)
        svgpm_pred = current_model.predict(svgpm_input)
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"SVGPM_Input_GIAF_MinMax{model_name_str}.npy"), svgpm_pred)

        (svgpm_total_pred,
         svgpm_pred_data,
         svgpm_real_data,
         svgpm_diff_data,
         svgpm_ds) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=svgpm_pred,
            month_mat="./NPP_MAT/SVGPM_month_intp.mat",
            var_names='SVGPM_month_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"SVGPM_Input_GIAF_Pred{model_name_str}.npy"), svgpm_total_pred)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"SVGPM_Pred_10year{model_name_str}.npy"), svgpm_pred_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"SVGPM_Real_10year{model_name_str}.npy"), svgpm_real_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"SVGPM_Diff_10year{model_name_str}.npy"), svgpm_diff_data)
        svgpm_ds.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"SVGPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of SVGPM_Total_Pred:", svgpm_total_pred.shape)
        del svgpm_total_pred, svgpm_pred_data, svgpm_real_data, svgpm_diff_data, svgpm_ds

        # ---------------------------
        # 3.3.4 Predict: MEAN
        # ---------------------------
        print(f"[INFO] Predicting on MEAN OBS data with {model_name_str}...")
        # mean_pred = making_Prediction(current_model, mean_input)
        mean_pred = current_model.predict(mean_input)
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"MEAN_Input_GIAF_MinMax{model_name_str}.npy"), mean_pred)

        (mean_total_pred,
         mean_pred_data,
         mean_real_data,
         mean_diff_data,
         mean_ds) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=mean_pred,
            month_mat="./NPP_MAT/NPP_month_mean_intp.mat",
            var_names='NPP_month_mean_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"MEAN_Input_GIAF_Pred{model_name_str}.npy"), mean_total_pred)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"MEAN_Pred_10year{model_name_str}.npy"), mean_pred_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"MEAN_Real_10year{model_name_str}.npy"), mean_real_data)
        # np.save(os.path.join(SAVE_DIR_SHOWCASE, f"MEAN_Diff_10year{model_name_str}.npy"), mean_diff_data)
        mean_ds.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"MEAN_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of MEAN_Total_Pred:", mean_total_pred.shape)
        del mean_total_pred, mean_pred_data, mean_real_data, mean_diff_data,mean_ds

        # Log shapes
        print("[INFO] Direct Prediction shapes:")
        print("CbPM_Pred:", cbpm_pred.shape)
        print("EVGPM_Pred:", evgpm_pred.shape)
        print("SVGPM_Pred:", svgpm_pred.shape)
        print("MEAN_Pred:", mean_pred.shape)

        # ---------------------------
        # 3.4 Predict Full Output
        # ---------------------------
        print("[INFO] Full Output Prediction for each dataset...")

        cbpm_input_full = get_full_Input(INPUT_FRAMES, cbpm_scaled)
        evgpm_input_full = get_full_Input(INPUT_FRAMES, evgpm_scaled)
        svgpm_input_full = get_full_Input(INPUT_FRAMES, svgpm_scaled)
        mean_input_full = get_full_Input(INPUT_FRAMES, mean_scaled)

        # 3.4.1 CbPM Full
        # cbpm_pred_full = making_Prediction(current_model, cbpm_input_full)
        cbpm_pred_full = current_model.predict(cbpm_input_full)
        (cbpm_total_pred_full,
         cbpm_pred_data_full,
         cbpm_real_data_full,
         cbpm_diff_data_full,
         cbpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=cbpm_pred_full,
            month_mat="./NPP_MAT/CbPM_month_intp.mat",
            var_names='CbPM_month_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"full_CbPM_Input_GIAF_Pred{model_name_str}.npy"), cbpm_total_pred_full)
        cbpm_ds_full.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"full_CbPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of full CbPM_Total_Pred:", cbpm_total_pred_full.shape)
        del cbpm_total_pred_full, cbpm_pred_data_full, cbpm_real_data_full, cbpm_diff_data_full, cbpm_ds_full

        # 3.4.2 EVGPM Full
        # evgpm_pred_full = making_Prediction(current_model, evgpm_input_full)
        evgpm_pred_full = current_model.predict(evgpm_input_full)
        (evgpm_total_pred_full,
         evgpm_pred_data_full,
         evgpm_real_data_full,
         evgpm_diff_data_full,
         evgpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=evgpm_pred_full,
            month_mat="./NPP_MAT/EVGPM_month_intp.mat",
            var_names='EVGPM_month_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"full_EVGPM_Input_GIAF_Pred{model_name_str}.npy"), evgpm_total_pred_full)
        evgpm_ds_full.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"full_EVGPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of full EVGPM_Total_Pred:", evgpm_total_pred_full.shape)
        del evgpm_total_pred_full,evgpm_pred_data_full,evgpm_real_data_full,evgpm_diff_data_full, evgpm_ds_full

        # 3.4.3 SVGPM Full
        # svgpm_pred_full = making_Prediction(current_model, svgpm_input_full)
        svgpm_pred_full = current_model.predict(svgpm_input_full)
        (svgpm_total_pred_full,
         svgpm_pred_data_full,
         svgpm_real_data_full,
         svgpm_diff_data_full,
         svgpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=svgpm_pred_full,
            month_mat="./NPP_MAT/SVGPM_month_intp.mat",
            var_names='SVGPM_month_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"full_SVGPM_Input_GIAF_Pred{model_name_str}.npy"), svgpm_total_pred_full)
        svgpm_ds_full.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"full_SVGPM_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of full SVGPM_Total_Pred:", svgpm_total_pred_full.shape)
        del svgpm_total_pred_full, svgpm_pred_data_full, svgpm_real_data_full, svgpm_diff_data_full, svgpm_ds_full

        # 3.4.4 MEAN Full
        # mean_pred_full = making_Prediction(current_model, mean_input_full)
        mean_pred_full = current_model.predict(mean_input_full)
        (mean_total_pred_full,
         mean_pred_data_full,
         mean_real_data_full,
         mean_diff_data_full,
         mean_ds_full) = Inversing_the_PredData(
            start_date=START_DATE,
            pred_npy=mean_pred_full,
            month_mat="./NPP_MAT/NPP_month_mean_intp.mat",
            var_names='NPP_month_mean_intp'
        )
        # np.save(os.path.join(SAVE_DIR_OUTPUTS, f"full_MEAN_Input_GIAF_Pred{model_name_str}.npy"), mean_total_pred_full)
        mean_ds_full.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, f"full_MEAN_Input_GIAF_Pred{model_name_str}.nc"))
        print("shape of full MEAN_Total_Pred:", mean_total_pred_full.shape)
        del mean_total_pred_full,mean_pred_data_full, mean_real_data_full, mean_diff_data_full, mean_ds_full

        print(f"[INFO] Done processing for model: {model_name_str}")
