from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm

def build_lstm_model(input_shape):
    model = Sequential(name="LSTM_Model")

    # Keep all LSTM layers returning sequences
    model.add(Bidirectional(LSTM(400, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0),
                                 input_shape=input_shape),
                            name="Bidirectional_LSTM_1"))
    model.add(Dropout(0.1, name="Dropout1"))

    model.add(Bidirectional(LSTM(300, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_2"))
    model.add(Dropout(0.1, name="Dropout2"))

    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_3"))
    model.add(Dropout(0.1, name="Dropout3"))

    model.add(Bidirectional(LSTM(150, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_4"))
    model.add(Dropout(0.1, name="Dropout4"))

    # Final TimeDistributed layer to produce 3 outputs per time step
    model.add(TimeDistributed(Dense(3), name="TimeDistributed_Output"))

    return model
