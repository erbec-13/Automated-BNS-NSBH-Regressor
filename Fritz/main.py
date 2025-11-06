import os
import numpy as np
import traceback
from gcn_kafka import Consumer
from utils import (
    parse_gcn,
    get_params,
    predict_with_uncertainty,
    plot_all_light_curves_with_uncertainty,
)
from joblib import load

GCN_CLIENT_ID = os.getenv("GCN_CLIENT_ID")
GCN_CLIENT_SECRET = os.getenv("GCN_CLIENT_SECRET")
GCN_GROUP_ID = str(os.getenv("GCN_GROUP_ID", "oraclefritzbot"))
if not GCN_CLIENT_ID or not GCN_CLIENT_SECRET:
    raise ValueError(
        "GCN_CLIENT_ID and GCN_CLIENT_SECRET must be set as environment variables"
    )

config = {
    "group.id": GCN_GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}
consumer = Consumer(
    config=config,
    client_id=GCN_CLIENT_ID,
    client_secret=GCN_CLIENT_SECRET,
    domain="gcn.nasa.gov",
)
consumer.subscribe(
    [
        "gcn.classic.voevent.LVC_PRELIMINARY",
        "gcn.classic.voevent.LVC_INITIAL",
        "gcn.classic.voevent.LVC_UPDATE",
    ]
)


# get the base directory, as the directory where this file is located
base_dir = os.path.dirname(os.path.abspath(__file__))
print("Loading model and scalers...")
model = load(f"{base_dir}/LSTMpredLC__PAstro.joblib")
target_scaler = load(f"{base_dir}/target_scaler_PAstro.joblib")
feature_scaler = load(f"{base_dir}/feature_scaler_PAstro.joblib")

while True:
    try:
        for message in consumer.consume():
            value = message.value()
            parsed = parse_gcn(value)
            params = get_params(parsed)
            (
                superevent_id,
                event_page,
                alert_type,
                group,
                prob_bbh,
                prob_bns,
                prob_nsbh,
                far_format,
                distmean,
                area_90,
                longitude,
                latitude,
                has_ns,
                has_remnant,
                has_mass_gap,
                significant,
                prob_ter,
                skymap,
                PAstro,
                time,
                skymap_type,
            ) = params

            if alert_type != "RETRACTION" and distmean != "error":
                print(f"Processing {superevent_id} ({alert_type})")
                model = load(f"{base_dir}/LSTMpredLC__PAstro.joblib")
                X = np.vstack(
                    (
                        area_90,
                        distmean,
                        longitude,
                        latitude,
                        has_ns,
                        has_remnant,
                        has_mass_gap,
                        PAstro,
                    )
                ).T

                # Time array
                t_min = 0.1
                t_max = 14.0
                num_time_points = 71
                dt = 0.2
                time_single = np.linspace(t_min, t_max, num_time_points)
                filter_order = 3

                # Standardize the target data
                X_new = feature_scaler.transform(X)

                # Reshape X data for LSTM input based on the model's input shape
                X_new_reshaped = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))

                mean_preds_new, uncertainty_new = predict_with_uncertainty(
                    model, X_new_reshaped, n_iter=1000
                )

                # Reshape the mean predictions to match the shape used during scaling (num_samples, num_time_points * num_filters)
                mean_preds_flat = mean_preds_new.reshape(
                    mean_preds_new.shape[0], num_time_points * 3
                )
                # Invert the standardization for the mean predictions
                mean_preds_inverted = target_scaler.inverse_transform(
                    mean_preds_flat
                ).reshape(mean_preds_new.shape[0], num_time_points, 3)

                # Reshape uncertainty to match mean_preds_inverted shape
                uncertainty_reshaped = uncertainty_new.reshape(
                    uncertainty_new.shape[0], num_time_points, 3
                )

                plot_all_light_curves_with_uncertainty(
                    time_single,
                    mean_preds_inverted,
                    uncertainty_reshaped,
                    superevent_id,
                    time,
                    alert_type,
                )

    except Exception as e:
        print(e)
        traceback.print_exc()
        continue

consumer.close()
