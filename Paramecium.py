import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import torch
import torch.nn as nn


class MlBandwidthEstimator(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 2)

    def forward(self, x, h, c):

        cong_thres = 35 # ms

        queuing_delay_most_rec_1 = x[0,0,30]
        queuing_delay_most_rec_2 = x[0,0,31]
        queuing_delay_most_rec_3 = x[0,0,32]

        queuing_delay_avg = (queuing_delay_most_rec_1 + queuing_delay_most_rec_2 + queuing_delay_most_rec_3) / 3.

        receiving_rate_most_rec_1 = x[0,0,0]
        receiving_rate_most_rec_2 = x[0,0,1]
        receiving_rate_most_rec_3 = x[0,0,2]

        receiving_rate_avg = (receiving_rate_most_rec_1 + receiving_rate_most_rec_2 + receiving_rate_most_rec_3) / 3.

        res1 = receiving_rate_avg * 1.3 
        
        bw_multi =  cong_thres / (queuing_delay_avg + 1)
        res2 = receiving_rate_avg * torch.sqrt(bw_multi)

        res = (res1 + res2) / 2.

        x = self.fc(x)

        x[0, 0, 0] = res
        x[0, 0, 1] = 0.

        return x, h, c  

    
if __name__ == "__main__":
    # batch size
    BS = 1
    # time steps
    T = 2000
    # observation vector dimension
    obs_dim = 150
    # number of hidden units in the LSTM
    hidden_size = 1
    
    # instantiate the ML BW estimator
    torchBwModel = MlBandwidthEstimator(obs_dim)
    
    # create dummy inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = np.asarray(np.random.uniform(0, 1, size=(BS, T, obs_dim)), dtype=np.float32)
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    torch_initial_hidden_state = torch.zeros((BS, hidden_size))
    torch_initial_cell_state = torch.zeros((BS, hidden_size))

    # predict dummy outputs: 1 episode x T timesteps x 2 (mean and std)
    dummy_outputs, final_hidden_state, final_cell_state = torchBwModel(torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state)

    # save onnx model
    model_path = "./tmp/onnxBwModel.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torchBwModel.to("cpu")
    torchBwModel.eval()
    torch.onnx.export(
        torchBwModel,
        (torch_dummy_inputs[0:1, 0:1, :], torch_initial_hidden_state, torch_initial_cell_state),
        model_path,
        opset_version=11,
        input_names=['obs', 'hidden_states', 'cell_states'], # the model's input names
        output_names=['output', 'state_out', 'cell_out'], # the model's output names
    )
    
    # verify torch and onnx models outputs
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    onnx_hidden_state, onnx_cell_state = (np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
    torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
    # online interaction: step through the environment 1 time step at a time
    with torch.no_grad():
        for i in tqdm(range(dummy_inputs.shape[1])):
            torch_estimate, torch_hidden_state, torch_cell_state = torchBwModel(torch_dummy_inputs[0:1, i:i+1, :], torch_hidden_state, torch_cell_state)
            feed_dict= {'obs': dummy_inputs[0:1, i:i+1, :], 'hidden_states': onnx_hidden_state, 'cell_states': onnx_cell_state}
            onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=1e-6), 'Failed to match model outputs!'
            assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
            assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'
        
        assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
        assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
        print("Torch and Onnx models outputs have been verified successfully!")