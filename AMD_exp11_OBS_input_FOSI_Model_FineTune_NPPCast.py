"""
Experiment 3 Code for NPP Prediction (OBS Fine-Tuning of FOSI-trained Models)

Key Steps:
1. Data Loading & MinMax Scaling (OBS Data)
2. Loading Pre-trained FOSI Models (UNet, VNet, AttUNet, R2UNet)
3. Fine-tuning on part of OBS data, validating on another part
4. Predicting outputs, inverse-scaling, and saving results (both last 10-year slice and full timeline)

Author: Bizhi Wu (2025.06.16)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import mat73
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from keras_unet_collection import models
import NPPCast

# -----------------------------------------------------------------------------
# 1. Environment Configuration & Hyperparameters
# -----------------------------------------------------------------------------

# AMD GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '12288'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Hyperparameters
BATCH_SIZE = 12
SMOOTH = 1e-9
EPOCHS = 100
SAVE_MODEL = True
TRAINING = False

INPUT_FRAMES =  36       # 3 years monthly
OUTPUT_FRAMES = 36       # 3 years monthly
WIDTH = 384
HEIGHT = 320

# Directory for loading pre-trained FOSI model weights
SAVE_DIR_MODELS = "./AMD_Archives/Model_UNet_family_FOSI/"

# Fine-tune partition: e.g., we use some months for training, some for validation
FTyear = 6 * 12 + 1 * 12 + 8 

# Start date for predictions
START_DATE_FINE_TUNE = '1/1/2009'  # Fine-tune scenario start date
START_DATE_FULL = '1/1/2002'       # Full scenario start date

# Global MinMaxScaler (consistent with other experiments)
MinMax_scaler = MinMaxScaler()

# -----------------------------------------------------------------------------
# 2. Utility / Helper Functions
# -----------------------------------------------------------------------------

def CESM_MaxMin(data_3d: np.ndarray):
    """
    Fit a MinMaxScaler on the input data (time, lat, lon),
    then transform it back to the original 3D shape.

    Args:
        data_3d (np.ndarray): shape [time, 384, 320].

    Returns:
        scaler (MinMaxScaler): Fitted scaler.
        scaled_3d (np.ndarray): Scaled data, same shape as input.
    """
    time_len = len(data_3d)
    reshaped = data_3d.reshape([time_len, -1])  # [time, lat*lon]
    MinMax_scaler.fit(reshaped)
    scaled = MinMax_scaler.transform(reshaped)
    scaled_3d = scaled.reshape([time_len, WIDTH, HEIGHT])
    return MinMax_scaler, scaled_3d

def get_Input_Output(input_frames: int, output_frames: int, data_3d: np.ndarray):
    """
    Create input (X) and output (Y) from scaled data with a sliding window.

    Args:
        input_frames (int): # months for historical input.
        output_frames (int): # months for prediction output.
        data_3d (np.ndarray): shape [time, WIDTH, HEIGHT].

    Returns:
        X (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].
        Y (np.ndarray): shape [N, WIDTH, HEIGHT, output_frames].
    """
    total_time = len(data_3d)
    n_samples = total_time - input_frames - output_frames

    X = np.zeros([n_samples, input_frames, WIDTH, HEIGHT])
    Y = np.zeros([n_samples, output_frames, WIDTH, HEIGHT])

    for i in range(n_samples):
        X[i] = data_3d[i : i + input_frames]
        Y[i] = data_3d[i + input_frames : i + input_frames + output_frames]

    X = X.transpose(0, 2, 3, 1)  # [N, WIDTH, HEIGHT, input_frames]
    Y = Y.transpose(0, 2, 3, 1)  # [N, WIDTH, HEIGHT, output_frames]
    return X, Y

def get_full_Input(input_frames: int, data_3d: np.ndarray):
    """
    Generate a sliding-window input over the entire dataset,
    ignoring leftover frames that can't form a full input window.

    Args:
        input_frames (int): number of historical frames.
        data_3d (np.ndarray): shape [time, WIDTH, HEIGHT].

    Returns:
        X_full (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].
    """
    total_time = len(data_3d)
    n_samples = total_time - input_frames

    X_full = np.zeros([n_samples, input_frames, WIDTH, HEIGHT])
    for i in range(n_samples):
        X_full[i] = data_3d[i : i + input_frames]

    X_full = X_full.transpose(0, 2, 3, 1)
    return X_full

def making_Prediction(model: tf.keras.Model, input_sequence: np.ndarray):
    """
    Use the model to predict, splitting in batches of size 12 if needed.

    Args:
        model: Keras model for inference.
        input_sequence (np.ndarray): shape [N, WIDTH, HEIGHT, frames].

    Returns:
        np.ndarray: [N, WIDTH, HEIGHT, output_frames].
    """
    predictions_list = []
    gap = len(input_sequence)
    if gap > 11:
        num_loops = gap // 12
        for idx in range(num_loops):
            batch_data = input_sequence[idx*12 : idx*12 + 12]
            batch_pred = model.predict(batch_data)
            predictions_list.append(batch_pred)

        remainder = gap - (num_loops * 12)
        if remainder > 0:
            rest_pred = model.predict(input_sequence[num_loops*12 + 12 :])
            predictions_list.append(rest_pred)

        return np.concatenate(predictions_list, axis=0)
    else:
        return model.predict(input_sequence)

def ssim_metric(x, y, max_val=1.0):
    """Wrapper for TF's built-in SSIM."""
    return tf.image.ssim(x, y, max_val)

def psnr_metric(x, y, max_val=1.0):
    """Wrapper for TF's built-in PSNR."""
    return tf.image.psnr(x, y, max_val)

def POD_metric(x, y):
    """Probability of Detection (Recall)."""
    y_pos = K.clip(x, 0, 1)
    y_pred_pos = K.clip(y, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + SMOOTH) / (tp + fn + SMOOTH)

def FAR_metric(x, y):
    """False Alarm Rate."""
    y_pred_pos = K.clip(y, 0, 1)
    y_pos = K.clip(x, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    return fp / (tp + fp + SMOOTH)

def Inversing_the_PredData(
    start_date='1/1/2002',
    pred_npy=None,
    month_mat="./OBS_NPP_CbPM_month.mat",
    var_names='OBS_NPP_CbPM_month'
):
    """
    Inverse the MinMax scaling for predicted data, apply ocean mask, create xarray dataset.

    Args:
        start_date (str): Starting date for xarray coordinate.
        pred_npy (np.ndarray): shape [N, lat, lon, time].
        month_mat (str): Path to original .mat file for reference data.
        var_names (str): Variable name in .mat file.

    Returns:
        pred_data (np.ndarray): shape [N, lat, lon, time].
        pred_last_10y (np.ndarray): shape [time, lat, lon].
        real_last_10y (np.ndarray): shape [time, lat, lon].
        diff_last_10y (np.ndarray): shape [time, lat, lon].
        ds (xr.Dataset): xarray dataset of the predictions.
    """
    # 1. Load ocean mask
    # spNPP = xr.open_dataset("../NPP_Data/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.photoC_diaz.208101-210012.nc")
    # mask = (spNPP['REGION_MASK'].values > 0).astype(np.int32)
    region_mask = xr.open_dataset('./Common_data/CESM_Parameter.nc')['REGION_MASK'].values
    mask = (region_mask > 0).astype(np.int32)

    # 2. Load original .mat data (OBS)
    obs_data = mat73.loadmat(month_mat)
    obs_npp_3d = obs_data[var_names]
    obs_npp_3d = np.nan_to_num(obs_npp_3d.transpose(2, 1, 0))  # [time, lat, lon]

    # 3. Fit local MinMax to retrieve data_min, data_range
    local_scaler, scaled_npp = CESM_MaxMin(obs_npp_3d)
    data_min = local_scaler.data_min_     # shape [lat*lon]
    data_range = local_scaler.data_range_ # shape [lat*lon]

    # 4. Reshape data_min / data_range
    data_min_reshaped = data_min.reshape(WIDTH, HEIGHT)
    data_range_reshaped = data_range.reshape(WIDTH, HEIGHT)

    n_samples = len(pred_npy)
    n_time = pred_npy.shape[-1]

    # 5. Expand them to match pred shape: [N, lat, lon, time]
    min_4d = np.transpose(
        np.array([[data_min_reshaped] * n_time] * n_samples),
        (0, 2, 3, 1)
    )
    range_4d = np.transpose(
        np.array([[data_range_reshaped] * n_time] * n_samples),
        (0, 2, 3, 1)
    )
    mask_4d = np.transpose(
        np.array([[mask] * n_time] * n_samples),
        (0, 2, 3, 1)
    )

    # 6. Inverse scaling
    masked_pred = pred_npy * mask_4d
    pred_data = masked_pred * range_4d + min_4d

    # 7. Build xarray dataset
    lat_ds = xr.open_dataset("./NPP_Data/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.photoC_diaz.208101-210012.nc")
    lat_vals = lat_ds["nlat"].values
    lon_vals = lat_ds["nlon"].values

    time_coords = np.arange(1, 1 + n_time)
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
        attrs=dict(description="Fine-tuned model prediction on OBS data.")
    )

    # 8. Get last 10 years
    pred_last_10y = pred_data[-1]  # shape [lat, lon, time]
    pred_last_10y = pred_last_10y.transpose(2, 0, 1)  # [time, lat, lon]

    real_last_10y = obs_npp_3d[-OUTPUT_FRAMES:]  # [120, lat, lon]
    diff_last_10y = pred_last_10y - real_last_10y  # [time, lat, lon]

    return pred_data, pred_last_10y, real_last_10y, diff_last_10y, ds

def model_training(
    model: tf.keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    val_data: tuple,
    model_name: str,
    save: bool = True
):
    """
    Fine-tune a given model with custom metrics, returning the trained model.

    Args:
        model (tf.keras.Model): A loaded Keras model to be fine-tuned.
        X_train (np.ndarray): shape [N, WIDTH, HEIGHT, input_frames].
        Y_train (np.ndarray): shape [N, WIDTH, HEIGHT, output_frames].
        val_data (tuple): (X_val, Y_val) for validation.
        model_name (str): Name for saving model weights.
        save (bool): If True, will save best model checkpoint.

    Returns:
        tf.keras.Model: The fine-tuned model.
    """
    metrics = [
        'accuracy',
        ssim_metric,
        psnr_metric,
        POD_metric,
        FAR_metric
    ]
    print("[INFO] Compiling Model for Fine-tuning...")

    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=metrics)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, mode='min',
        restore_best_weights=True, verbose=2
    )
    mcp_save = ModelCheckpoint(
        os.path.join(SAVE_DIR_MODELS, model_name + ".h5"),
        save_best_only=True, monitor='val_loss', mode='min', verbose=2
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=7,
        verbose=2, mode='min', min_delta=1e-4
    )

    print("[INFO] Fine-tuning started...")
    model.fit(
        X_train, Y_train,
        validation_data=val_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[mcp_save, reduce_lr_loss, early_stopping]
    )
    print("[INFO] Fine-tuning DONE.")

    return model

# -----------------------------------------------------------------------------
# 3. Main Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------
    # 3.1. Load OBS Data & Preprocess
    # ---------------------------
    print("[INFO] CbPM Preprocessing...")
    obs_cbpm = mat73.loadmat("./NPP_MAT/CbPM_month_intp.mat")['CbPM_month_intp']
    obs_cbpm = np.nan_to_num(obs_cbpm.transpose(2, 1, 0))
    cbpm_scaler, cbpm_scaled = CESM_MaxMin(obs_cbpm)
    cbpm_input, cbpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, cbpm_scaled)
    print("CbPM Input shape:", cbpm_input.shape, "Output shape:", cbpm_output.shape)

    print("[INFO] EVGPM Preprocessing...")
    obs_evgpm = mat73.loadmat("./NPP_MAT/EVGPM_month_intp.mat")['EVGPM_month_intp']
    obs_evgpm = np.nan_to_num(obs_evgpm.transpose(2, 1, 0))
    evgpm_scaler, evgpm_scaled = CESM_MaxMin(obs_evgpm)
    evgpm_input, evgpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, evgpm_scaled)
    print("EVGPM Input shape:", evgpm_input.shape, "Output shape:", evgpm_output.shape)

    print("[INFO] SVGPM Preprocessing...")
    obs_svgpm = mat73.loadmat("./NPP_MAT/SVGPM_month_intp.mat")['SVGPM_month_intp']
    obs_svgpm = np.nan_to_num(obs_svgpm.transpose(2, 1, 0))
    svgpm_scaler, svgpm_scaled = CESM_MaxMin(obs_svgpm)
    svgpm_input, svgpm_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, svgpm_scaled)
    print("SVGPM Input shape:", svgpm_input.shape, "Output shape:", svgpm_output.shape)

    print("[INFO] MEAN OBS Preprocessing...")
    # obs_mean = mat73.loadmat("../NPP_MAT/NPP_annual_mean_intp.mat")['NPP_annual_mean_intp']
    obs_mean = mat73.loadmat("./NPP_MAT/NPP_month_mean_intp.mat")['NPP_month_mean_intp']
    obs_mean = np.nan_to_num(obs_mean.transpose(2, 1, 0))
    mean_scaler, mean_scaled = CESM_MaxMin(obs_mean)
    mean_input, mean_output = get_Input_Output(INPUT_FRAMES, OUTPUT_FRAMES, mean_scaled)
    print("MEAN Input shape:", mean_input.shape, "Output shape:", mean_output.shape)

    # ---------------------------
    # 3.2. Define Pre-trained Models & Names
    # ---------------------------
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
    model_name_list = ['UNet', 'VNet', 'AttUNet', 'R2UNet']
    model_dict = {
        'UNet': unet_2d,
        'VNet': vnet_2d,
        'AttUNet': attunet_2d,
        'R2UNet': r2unet_2d
    }
    # # model_name_list = ['NPPCast']
    # # model_dict = {'NPPCast':NPP_unet}

    # model_name_list = ['R2UNet']
    # model_dict = {'R2UNet':r2unet_2d}

    # Add prefixes for your fine-tuned models
    cbpm_model_prefix = "FineTune_CbPM_"
    evgpm_model_prefix = "FineTune_EVGPM_"
    svgpm_model_prefix = "FineTune_SVGPM_"
    mean_model_prefix = "FineTune_MEAN_"

    # ---------------------------
    # 3.3. Fine-tune & Predict for Each Model
    # ---------------------------
    for model_key in model_name_list:
        tf.keras.backend.clear_session()
        current_model = model_dict[model_key]
        model_name_str = f"FOSI_{model_key}"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.3.1  Fine-tune + Predict => CbPM
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"\n[INFO] Fine-tuning {model_name_str} on CbPM data...")
        # Load original pre-trained weights
        weight_path = os.path.join(SAVE_DIR_MODELS, model_name_str + ".h5")
        current_model.load_weights(weight_path)

        # Prepare data partition (train, val)
        cbpm_val_data = (cbpm_input[FTyear:], cbpm_output[FTyear:])
        cbpm_model_name = cbpm_model_prefix + model_name_str

        # Fine-tune
        cbpm_finetuned_model = model_training(
            model=current_model,
            X_train=cbpm_input[:FTyear],
            Y_train=cbpm_output[:FTyear],
            val_data=cbpm_val_data,
            model_name=cbpm_model_name,
            save=True
        )
        # Predict on validation slice
        # cbpm_pred = making_Prediction(cbpm_finetuned_model, cbpm_input[FTyear:])
        cbpm_pred = cbpm_finetuned_model.predict(cbpm_input[FTyear:])
        # np.save(os.path.join("./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune", f"CbPM_Input_FOSI_MinMax{model_name_str}.npy"), cbpm_pred)

        # Inverse scaling
        tf.keras.backend.clear_session()
        (cbpm_total_pred,
         cbpm_pred_data,
         cbpm_real_data,
         cbpm_diff_data,
         cbpm_ds) = Inversing_the_PredData(
            start_date=START_DATE_FINE_TUNE,
            pred_npy=cbpm_pred,
            month_mat="./NPP_MAT/CbPM_month_intp.mat",
            var_names='CbPM_month_intp'
        )
        # Save results
        np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/CbPM_Input_FOSI_Pred{model_name_str}.npy", cbpm_total_pred)
        showcase_dir = "./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/ShowCases"
        os.makedirs(showcase_dir, exist_ok=True)

        # np.save(os.path.join(showcase_dir, f"CbPM_Pred_10year{model_name_str}.npy"), cbpm_pred_data)
        # np.save(os.path.join(showcase_dir, f"CbPM_Real_10year{model_name_str}.npy"), cbpm_real_data)
        # np.save(os.path.join(showcase_dir, f"CbPM_Diff_10year{model_name_str}.npy"), cbpm_diff_data)

        cbpm_nc_file = f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/CbPM_Input_FOSI_Pred{model_name_str}.nc"
        cbpm_ds.to_netcdf(cbpm_nc_file)
        print("CbPM fine-tune done. shape of total:", cbpm_total_pred.shape)
        del cbpm_total_pred, cbpm_pred_data, cbpm_real_data, cbpm_diff_data, cbpm_ds

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.3.2 Fine-tune + Predict => EVGPM
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"\n[INFO] Fine-tuning {model_name_str} on EVGPM data...")
        current_model.load_weights(weight_path)

        evgpm_val_data = (evgpm_input[FTyear:], evgpm_output[FTyear:])
        evgpm_model_name = evgpm_model_prefix + model_name_str

        evgpm_finetuned_model = model_training(
            model=current_model,
            X_train=evgpm_input[:FTyear],
            Y_train=evgpm_output[:FTyear],
            val_data=evgpm_val_data,
            model_name=evgpm_model_name,
            save=True
        )
        # evgpm_pred = making_Prediction(evgpm_finetuned_model, evgpm_input[FTyear:])
        evgpm_pred = evgpm_finetuned_model.predict(cbpm_input[FTyear:])
        # np.save(os.path.join("./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune", f"EVGPM_Input_FOSI_MinMax{model_name_str}.npy"), evgpm_pred)

        tf.keras.backend.clear_session()
        (evgpm_total_pred,
         evgpm_pred_data,
         evgpm_real_data,
         evgpm_diff_data,
         evgpm_ds) = Inversing_the_PredData(
            start_date=START_DATE_FINE_TUNE,
            pred_npy=evgpm_pred,
            month_mat="./NPP_MAT/EVGPM_month_intp.mat",
            var_names='EVGPM_month_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/EVGPM_Input_FOSI_Pred{model_name_str}.npy", evgpm_total_pred)
        # np.save(os.path.join(showcase_dir, f"EVGPM_Pred_10year{model_name_str}.npy"), evgpm_pred_data)
        # np.save(os.path.join(showcase_dir, f"EVGPM_Real_10year{model_name_str}.npy"), evgpm_real_data)
        # np.save(os.path.join(showcase_dir, f"EVGPM_Diff_10year{model_name_str}.npy"), evgpm_diff_data)
        evgpm_ds.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/EVGPM_Input_FOSI_Pred{model_name_str}.nc")
        print("EVGPM fine-tune done. shape of total:", evgpm_total_pred.shape)
        del evgpm_total_pred, evgpm_pred_data, evgpm_real_data, evgpm_diff_data, evgpm_ds

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.3.3 Fine-tune + Predict => SVGPM
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"\n[INFO] Fine-tuning {model_name_str} on SVGPM data...")
        current_model.load_weights(weight_path)

        svgpm_val_data = (svgpm_input[FTyear:], svgpm_output[FTyear:])
        svgpm_model_name = svgpm_model_prefix + model_name_str

        svgpm_finetuned_model = model_training(
            model=current_model,
            X_train=svgpm_input[:FTyear],
            Y_train=svgpm_output[:FTyear],
            val_data=svgpm_val_data,
            model_name=svgpm_model_name,
            save=True
        )
        # svgpm_pred = making_Prediction(svgpm_finetuned_model, svgpm_input[FTyear:])
        svgpm_pred = svgpm_finetuned_model.predict(svgpm_input[FTyear:])
        # np.save(os.path.join("./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune", f"SVGPM_Input_FOSI_MinMax{model_name_str}.npy"), svgpm_pred)

        tf.keras.backend.clear_session()
        (svgpm_total_pred,
         svgpm_pred_data,
         svgpm_real_data,
         svgpm_diff_data,
         svgpm_ds) = Inversing_the_PredData(
            start_date=START_DATE_FINE_TUNE,
            pred_npy=svgpm_pred,
            month_mat="./NPP_MAT/SVGPM_month_intp.mat",
            var_names='SVGPM_month_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/SVGPM_Input_FOSI_Pred{model_name_str}.npy", svgpm_total_pred)
        # np.save(os.path.join(showcase_dir, f"SVGPM_Pred_10year{model_name_str}.npy"), svgpm_pred_data)
        # np.save(os.path.join(showcase_dir, f"SVGPM_Real_10year{model_name_str}.npy"), svgpm_real_data)
        # np.save(os.path.join(showcase_dir, f"SVGPM_Diff_10year{model_name_str}.npy"), svgpm_diff_data)
        svgpm_ds.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/SVGPM_Input_FOSI_Pred{model_name_str}.nc")
        print("SVGPM fine-tune done. shape of total:", svgpm_total_pred.shape)
        del svgpm_total_pred, svgpm_pred_data, svgpm_real_data, svgpm_diff_data, svgpm_ds

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.3.4 Fine-tune + Predict => MEAN OBS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"\n[INFO] Fine-tuning {model_name_str} on MEAN OBS data...")
        current_model.load_weights(weight_path)

        mean_val_data = (mean_input[FTyear:], mean_output[FTyear:])
        mean_model_name = mean_model_prefix + model_name_str

        mean_finetuned_model = model_training(
            model=current_model,
            X_train=mean_input[:FTyear],
            Y_train=mean_output[:FTyear],
            val_data=mean_val_data,
            model_name=mean_model_name,
            save=True
        )
        # mean_pred = making_Prediction(mean_finetuned_model, mean_input[FTyear:])
        mean_pred = mean_finetuned_model.predict(mean_input[FTyear:])
        # np.save(os.path.join("./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune", f"MEAN_Input_FOSI_MinMax{model_name_str}.npy"), mean_pred)

        tf.keras.backend.clear_session()
        (mean_total_pred,
         mean_pred_data,
         mean_real_data,
         mean_diff_data,
         mean_ds) = Inversing_the_PredData(
            start_date=START_DATE_FINE_TUNE,
            pred_npy=mean_pred,
            month_mat="./NPP_MAT/NPP_month_mean_intp.mat",
            var_names='NPP_month_mean_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/MEAN_Input_FOSI_Pred{model_name_str}.npy", mean_total_pred)
        # np.save(os.path.join(showcase_dir, f"MEAN_Pred_10year{model_name_str}.npy"), mean_pred_data)
        # np.save(os.path.join(showcase_dir, f"MEAN_Real_10year{model_name_str}.npy"), mean_real_data)
        # np.save(os.path.join(showcase_dir, f"MEAN_Diff_10year{model_name_str}.npy"), mean_diff_data)
        mean_ds.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/MEAN_Input_FOSI_Pred{model_name_str}.nc")
        print("MEAN fine-tune done. shape of total:", mean_total_pred.shape)
        del mean_total_pred, mean_pred_data, mean_real_data, mean_diff_data,mean_ds

        # ---------------------------
        # 3.4 Predict Full Timeline
        # ---------------------------
        print(f"\n[INFO] Generating full-timeline predictions for {model_name_str}...")

        # Prepare full input
        cbpm_input_full = get_full_Input(INPUT_FRAMES, cbpm_scaled)
        evgpm_input_full = get_full_Input(INPUT_FRAMES, evgpm_scaled)
        svgpm_input_full = get_full_Input(INPUT_FRAMES, svgpm_scaled)
        mean_input_full = get_full_Input(INPUT_FRAMES, mean_scaled)

        # 3.4.1 CbPM Full
        # cbpm_pred_full = making_Prediction(cbpm_finetuned_model, cbpm_input_full)
        cbpm_pred_full = cbpm_finetuned_model.predict(cbpm_input_full)
        (cbpm_total_pred_full,
         cbpm_pred_data_full,
         cbpm_real_data_full,
         cbpm_diff_data_full,
         cbpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE_FULL,
            pred_npy=cbpm_pred_full,
            month_mat="./NPP_MAT/CbPM_month_intp.mat",
            var_names='CbPM_month_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_CbPM_Input_FOSI_Pred{model_name_str}.npy", cbpm_total_pred_full)
        cbpm_ds_full.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_CbPM_Input_FOSI_Pred{model_name_str}.nc")
        print("Full CbPM shape:", cbpm_total_pred_full.shape)
        del cbpm_total_pred_full, cbpm_pred_data_full, cbpm_real_data_full, cbpm_diff_data_full, cbpm_ds_full

        # 3.4.2 EVGPM Full
        # evgpm_pred_full = making_Prediction(evgpm_finetuned_model, evgpm_input_full)
        evgpm_pred_full = evgpm_finetuned_model.predict(evgpm_input_full)
        (evgpm_total_pred_full,
         evgpm_pred_data_full,
         evgpm_real_data_full,
         evgpm_diff_data_full,
         evgpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE_FULL,
            pred_npy=evgpm_pred_full,
            month_mat="./NPP_MAT/EVGPM_month_intp.mat",
            var_names='EVGPM_month_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_EVGPM_Input_FOSI_Pred{model_name_str}.npy", evgpm_total_pred_full)
        evgpm_ds_full.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_EVGPM_Input_FOSI_Pred{model_name_str}.nc")
        print("Full EVGPM shape:", evgpm_total_pred_full.shape)
        del evgpm_total_pred_full,evgpm_pred_data_full,evgpm_real_data_full,evgpm_diff_data_full, evgpm_ds_full

        # 3.4.3 SVGPM Full
        # svgpm_pred_full = making_Prediction(svgpm_finetuned_model, svgpm_input_full)
        svgpm_pred_full = svgpm_finetuned_model.predict(svgpm_input_full)
        (svgpm_total_pred_full,
         svgpm_pred_data_full,
         svgpm_real_data_full,
         svgpm_diff_data_full,
         svgpm_ds_full) = Inversing_the_PredData(
            start_date=START_DATE_FULL,
            pred_npy=svgpm_pred_full,
            month_mat="./NPP_MAT/SVGPM_month_intp.mat",
            var_names='SVGPM_month_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_SVGPM_Input_FOSI_Pred{model_name_str}.npy", svgpm_total_pred_full)
        svgpm_ds_full.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_SVGPM_Input_FOSI_Pred{model_name_str}.nc")
        print("Full SVGPM shape:", svgpm_total_pred_full.shape)
        del svgpm_total_pred_full, svgpm_pred_data_full, svgpm_real_data_full, svgpm_diff_data_full, svgpm_ds_full

        # 3.4.4 MEAN Full
        # mean_pred_full = making_Prediction(mean_finetuned_model, mean_input_full)
        mean_pred_full = mean_finetuned_model.predict(mean_input_full)
        (mean_total_pred_full,
         mean_pred_data_full,
         mean_real_data_full,
         mean_diff_data_full,
         mean_ds_full) = Inversing_the_PredData(
            start_date=START_DATE_FULL,
            pred_npy=mean_pred_full,
            month_mat="./NPP_MAT/NPP_month_mean_intp.mat",
            var_names='NPP_month_mean_intp'
        )
        # np.save(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_MEAN_Input_FOSI_Pred{model_name_str}.npy", mean_total_pred_full)
        mean_ds_full.to_netcdf(f"./AMD_Archives/npp_outputs/OBS_Input_FOSI_FineTune/full_MEAN_Input_FOSI_Pred{model_name_str}.nc")
        print("Full MEAN shape:", mean_total_pred_full.shape)
        del mean_total_pred_full,mean_pred_data_full, mean_real_data_full, mean_diff_data_full, mean_ds_full

        print(f"[INFO] Completed all tasks for {model_name_str}.\n")
