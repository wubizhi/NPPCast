"""
Experiment 4 Code for NPP Prediction (GIAF-based Training)

This script demonstrates:
1. Data loading & MinMax scaling
2. Custom model construction (UNet, VNet, AttUNet, R2UNet)
3. Model training with custom metrics
4. Prediction and result saving (NetCDF, .npy)

Author: Bizhi Wu (2025.06.16)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras_unet_collection import models
import NPPCast

# -----------------------------------------------------------------------------
# 1. Environment & Hyperparameters
# -----------------------------------------------------------------------------

# GPU environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '12288'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Data shape / model I/O
INPUT_FRAMES =  36       # e.g., 3 years * 12 months
OUTPUT_FRAMES = 36       # e.g., 3 years * 12 months
WIDTH = 384
HEIGHT = 320

# Dataset partition years (user-defined)
TRAIN_YEARS = 80
VALID_YEARS = 5
TEST_YEARS = 95 - TRAIN_YEARS - VALID_YEARS

# Training hyperparameters
BATCH_SIZE = 12
EPOCHS = 100
SMOOTH = 1e-9
SAVE_MODEL = True
TRAINING = True

# Start date for generating date range in results
START_DATE = '1/1/1962'

# File paths
# NPP_FILE = "../NPP_Data/g.e11_LENS.GECOIAF.T62_g16.009.pop.h.NPP.024901-031612.nc" # FOSI
NPP_FILE = "./NPP_Data/g.e21.GIAF_JRA.TL319_g17.spinup-cycle5.pop.h.photoC_TOT.195801-201812.nc" # GIAF
# SPNPP_FILE = "./NPP_Data/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.photoC_diaz.208101-210012.nc"
SAVE_DIR_MODELS = "./AMD_Archives/Model_UNet_family_GIAF/"
SAVE_DIR_OUTPUTS = "./AMD_Archives/npp_outputs/GIAF_Input_GIAF_Model/"
SAVE_DIR_SHOWCASE = "./AMD_Archives/npp_outputs/GIAF_Input_GIAF_Model/ShowCases/"

# -----------------------------------------------------------------------------
# 2. Data Preprocessing and Inverse Functions
# -----------------------------------------------------------------------------

min_max_scaler = MinMaxScaler()

def cesm_min_max_scale(data_array: np.ndarray):
    """
    Fit a MinMaxScaler on the input data (reshaped to 2D),
    then transform it back to the original 3D shape.

    Args:
        data_array (np.ndarray): Original data with shape [time, lat, lon].

    Returns:
        scaler (MinMaxScaler): Fitted MinMaxScaler object.
        scaled_data (np.ndarray): Transformed data with same shape as input.
    """
    # Reshape to [time, lat*lon]
    reshaped_data = np.reshape(data_array, [len(data_array), -1])
    min_max_scaler.fit(reshaped_data)
    scaled_data = min_max_scaler.transform(reshaped_data)
    scaled_data = np.reshape(scaled_data, [len(scaled_data), WIDTH, HEIGHT])
    return min_max_scaler, scaled_data


# def inverse_prediction_scaling(
#     pred_data: np.ndarray,
#     data_min: np.ndarray,
#     data_range: np.ndarray,
#     mask: np.ndarray
# ) -> np.ndarray:
#     """
#     Inverse the scaling (MinMax) for the predicted data and apply an ocean mask.

#     Args:
#         pred_data (np.ndarray): Predicted data, shape: [N, lat, lon, time].
#         data_min (np.ndarray): Min values from the MinMaxScaler, shape: [lat*lon].
#         data_range (np.ndarray): Range values from the MinMaxScaler, shape: [lat*lon].
#         mask (np.ndarray): Ocean mask array, shape: [lat, lon]. 1 for ocean, 0 for land.

#     Returns:
#         np.ndarray: Data mapped back to original value scale, with land masked out.
#     """
#     # Expand data_min to match pred_data shape
#     data_min_reshaped = np.reshape(data_min, [WIDTH, HEIGHT])
#     tmp_min = np.array([[data_min_reshaped]*pred_data.shape[-1]]*len(pred_data))
#     min_ = np.transpose(tmp_min, [0, 2, 3, 1])  # shape [N, lat, lon, time]

#     # Expand data_range to match pred_data shape
#     data_range_reshaped = np.reshape(data_range, [WIDTH, HEIGHT])
#     tmp_range = np.array([[data_range_reshaped]*pred_data.shape[-1]]*len(pred_data))
#     range_ = np.transpose(tmp_range, [0, 2, 3, 1])

#     # Expand ocean mask
#     mask_3d = np.array([[mask]*pred_data.shape[-1]]*len(pred_data))
#     mask_3d = np.transpose(mask_3d, [0, 2, 3, 1])  # shape [N, lat, lon, time]

#     # Apply mask and inverse transform
#     masked_pred = pred_data * mask_3d
#     original_scale_pred = masked_pred * range_ + min_
#     return original_scale_pred


def inverse_prediction_scaling(
    pred_data: np.ndarray,
    data_min: np.ndarray,
    data_range: np.ndarray,
    mask: np.ndarray,
    inplace: bool = False
) -> np.ndarray:
    """
    Inverse-transform MinMax-scaled predictions and apply an ocean mask,
    using broadcasting instead of full-array copies.

    Args:
        pred_data (np.ndarray): [N, lat, lon, T] scaled predictions in [0,1].
        data_min (np.ndarray): [lat*lon] flattened per-pixel minima.
        data_range (np.ndarray): [lat*lon] flattened per-pixel ranges.
        mask (np.ndarray): [lat, lon] ocean mask (1=ocean, 0=land).
        inplace (bool): if True, modifies pred_data in place and returns it.

    Returns:
        np.ndarray: [N, lat, lon, T] in original units, with land zeroed.
    """
    # get spatial dims from mask
    lat, lon = mask.shape
    N, lat_p, lon_p, T = pred_data.shape
    assert (lat, lon) == (lat_p, lon_p), "pred_data and mask dims must match"

    # reshape min and range into (1, lat, lon, 1) for broadcasting
    data_min = data_min.reshape(lat, lon)[None, :, :, None]
    data_range = data_range.reshape(lat, lon)[None, :, :, None]
    mask_b = mask[None, :, :, None]

    # choose working array
    out = pred_data if inplace else pred_data.copy()

    # inverse transform: out = out * range + min
    out *= data_range
    out += data_min

    # apply mask
    out *= mask_b

    return out


# -----------------------------------------------------------------------------
# 3. Custom Losses / Metrics
# -----------------------------------------------------------------------------

def custom_ssim(x, y, max_val=1.0):
    """SSIM metric wrapper using tf.image.ssim."""
    return tf.image.ssim(x, y, max_val)

def custom_psnr(x, y, max_val=1.0):
    """PSNR metric wrapper using tf.image.psnr."""
    return tf.image.psnr(x, y, max_val)

def pod_metric(x, y):
    """
    Probability of Detection (POD).
    x: Ground truth
    y: Prediction
    """
    y_pos = K.clip(x, 0, 1)
    y_pred_pos = K.clip(y, 0, 1)
    y_pred_neg = 1 - y_pred_pos

    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + SMOOTH) / (tp + fn + SMOOTH)

def far_metric(x, y):
    """
    False Alarm Rate (FAR).
    x: Ground truth
    y: Prediction
    """
    y_pred_pos = K.clip(y, 0, 1)
    y_pos = K.clip(x, 0, 1)
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    return fp / (tp + fp + SMOOTH)


# -----------------------------------------------------------------------------
# 4. Model Training and Prediction
# -----------------------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_data=None,
    model_name: str = "UNET_model"
):
    """
    Train the given model with custom metrics and optional validation data.

    Args:
        model (tf.keras.Model): A compiled Keras model.
        x_train (np.ndarray): Training input data, shape [N, lat, lon, input_frames].
        y_train (np.ndarray): Training target data, shape [N, lat, lon, output_frames].
        val_data (tuple): (x_val, y_val) if validation is used, else None.
        model_name (str): Name for saving the trained model.

    Returns:
        (tf.keras.Model, History): The trained model and its training history.
    """
    metrics = [
        'accuracy',
        custom_ssim,
        custom_psnr,
        pod_metric,
        far_metric
    ]

    optim = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
        loss='mean_squared_logarithmic_error',
        optimizer=optim,
        metrics=metrics
    )

    if val_data is None:
        early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min', restore_best_weights=True)
        mcp_save = ModelCheckpoint(
            os.path.join(SAVE_DIR_MODELS, model_name + ".h5"),
            save_best_only=True, monitor='loss', mode='min'
        )
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, verbose=2, mode='min', min_delta=1e-4
        )

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            callbacks=[early_stopping, mcp_save, reduce_lr_loss]
        )
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
        mcp_save = ModelCheckpoint(
            os.path.join(SAVE_DIR_MODELS, model_name + ".h5"),
            save_best_only=True, monitor='val_loss', mode='min'
        )
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=10, verbose=2, mode='min', min_delta=1e-4
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=val_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            callbacks=[early_stopping, mcp_save, reduce_lr_loss]
        )

    return model, history


def make_prediction(model: tf.keras.Model, x_input: np.ndarray) -> np.ndarray:
    """
    Make predictions by splitting the input into batches of size 12 if needed.

    Args:
        model (tf.keras.Model): The trained model used for prediction.
        x_input (np.ndarray): Input data, shape [N, lat, lon, input_frames].

    Returns:
        np.ndarray: Concatenated predictions from the model.
    """
    pred_list = []
    gap = len(x_input)

    if gap > 11:
        for idx in range(gap // 12):
            batch_data = x_input[idx*12: idx*12 + 12]
            prediction = model.predict(batch_data)
            pred_list.append(prediction)

        # Handle remainder
        remainder = gap - (idx * 12 + 12)
        if remainder > 0:
            rest_pred = model.predict(x_input[idx*12 + 12:])
            pred_list.append(rest_pred)

        concatenated_pred = np.concatenate(pred_list)
        return concatenated_pred
    else:
        return model.predict(x_input)

# -----------------------------------------------------------------------------
# 5. Main Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 5.1 Create/save directories if needed
    os.makedirs(SAVE_DIR_MODELS, exist_ok=True)
    os.makedirs(SAVE_DIR_OUTPUTS, exist_ok=True)
    os.makedirs(SAVE_DIR_SHOWCASE, exist_ok=True)

    # 5.2 Read and scale NPP data
    xr_data = xr.open_dataset(NPP_FILE)
    npp_data = xr_data["photoC_TOT"][:,:,:,:].fillna(0).values  # [time, lat, lon]
    npp_data = np.sum(npp_data,1)
    time_bound = xr_data["time_bound"].values
    delta_ = time_bound[:,1] - time_bound[:,0]
    delta_month = np.zeros(len(delta_))
    for inx in range(len(delta_)):
        delta_month[inx] = delta_[inx].days
        npp_data[inx,:,:] = npp_data[inx,:,:]*60*60*24*10

    scaler, scaled_npp = cesm_min_max_scale(npp_data)
    data_min = scaler.data_min_
    data_range = scaler.data_range_
    lat = xr_data["nlat"].values
    lon = xr_data["nlon"].values

    # 5.3 Get land-ocean mask
    # sp_npp = xr.open_dataset(SPNPP_FILE)
    # region_mask = sp_npp['REGION_MASK'].values
    # region_mask = (region_mask > 0).astype(np.int32)
    region_mask = xr.open_dataset('./Common_data/CESM_Parameter.nc')['REGION_MASK'].values
    region_mask = (region_mask > 0).astype(np.int32)

    # 5.4 Prepare input and output sequences
    total_samples = len(scaled_npp)  # time dimension
    x_sequence = np.zeros([total_samples - INPUT_FRAMES - OUTPUT_FRAMES, INPUT_FRAMES, WIDTH, HEIGHT])
    y_sequence = np.zeros([total_samples - INPUT_FRAMES - OUTPUT_FRAMES, OUTPUT_FRAMES, WIDTH, HEIGHT])

    for idx in range(len(x_sequence)):
        x_sequence[idx] = scaled_npp[idx : idx + INPUT_FRAMES]
        y_sequence[idx] = scaled_npp[idx + INPUT_FRAMES : idx + INPUT_FRAMES + OUTPUT_FRAMES]

    # Transpose to [N, lat, lon, frames]
    x_sequence = x_sequence.transpose(0, 2, 3, 1)
    y_sequence = y_sequence.transpose(0, 2, 3, 1)

    print("Input shape:", x_sequence.shape)
    print("Output shape:", y_sequence.shape)

    # 5.5 Define 4 Models (UNet, VNet, AttUNet, R2UNet)
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

    # 5.6 Training loop for each model
    for model_key in model_name_list:
        current_model = model_dict[model_key]
        model_name_str = f"GIAF_{model_key}"

        # Clear session to avoid clutter from previous loops
        tf.keras.backend.clear_session()

        # Train
        trained_model, history = train_model(
            model=current_model,
            x_train=x_sequence,
            y_train=y_sequence,
            val_data=None,  # or you can set (x_val, y_val) if needed
            model_name=model_name_str
        )

        # Predict
        predictions = make_prediction(trained_model, x_sequence)
        # predictions = trained_model.predict(x_sequence)
        # np.save(
        #     os.path.join(SAVE_DIR_OUTPUTS, f"GIAF_Input_GIAF_MinMax_{model_name_str}.npy"),
        #     predictions
        # )

        # Inverse scaling
        inv_predictions = inverse_prediction_scaling(
            pred_data=predictions,
            data_min=data_min,
            data_range=data_range,
            mask=region_mask
        )

        # Save results as NetCDF and .npy
        time_coord = np.arange(1, OUTPUT_FRAMES + 1)
        start_dates = pd.date_range(start=START_DATE, periods=len(inv_predictions), freq='ME')

        ds_10year = xr.Dataset(
            data_vars=dict(
                ML_Pred_NPP=(["start_date", "nlat", "nlon", "time"], inv_predictions),
            ),
            coords=dict(
                start_date=(["start_date"], start_dates),
                nlat=(["nlat"], lat),
                nlon=(["nlon"], lon),
                time=(["time"], time_coord)
            ),
            attrs=dict(description="NPP Prediction by Deep Model.")
        )

        ds_10year_file = f"GIAF_Model_Pred_NPP_10y_{model_name_str}"
        ds_10year.to_netcdf(os.path.join(SAVE_DIR_OUTPUTS, ds_10year_file + ".nc"))
        np.save(os.path.join(SAVE_DIR_OUTPUTS, ds_10year_file), inv_predictions)

        # Extract the last 10 years from predictions for comparison
        # pred_last_10year = inv_predictions[-1, :, :, :]
        # pred_last_10year = pred_last_10year.transpose(2, 0, 1)  # shape: [time, lat, lon]
        # real_npp = npp_data[-12*10:, :, :]  # real last 10-year data
        # diff_npp = real_npp - pred_last_10year

        # np.save(
        #     os.path.join(SAVE_DIR_SHOWCASE, f"GIAF_Model_Pred_10year_{model_name_str}.npy"),
        #     pred_last_10year
        # )
        # np.save(
        #     os.path.join(SAVE_DIR_SHOWCASE, f"GIAF_Model_Real_10year_{model_name_str}.npy"),
        #     real_npp
        # )
        # np.save(
        #     os.path.join(SAVE_DIR_SHOWCASE, f"GIAF_Model_Diff_10year_{model_name_str}.npy"),
        #     diff_npp
        # )

        print(f"[INFO] Finished training and saving results for {model_name_str}.")
