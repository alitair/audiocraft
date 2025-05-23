import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
from functools import lru_cache
from transformers import AutoProcessor, EncodecModel
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft.models import CompressionModel

def save_multi_pmi_data(path, pmi_matrices):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'pmi_matrices': pmi_matrices}, path)
    print(f"Saved multi-distance PMI data to {path}")

def load_multi_pmi_data(path):
    return torch.load(path, map_location='cpu')['pmi_matrices']

def load_all_wav_files(args, model, processor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_tensors = [torch.tensor(encode_wav_tokens(f, model, processor)[2], dtype=torch.long, device=device) for f in args.wav_files]
    # Concatenate along the sample dimension: [channel, codebook, total_samples]
    concat = torch.cat(token_tensors, dim=2)
    print("Shape of concat before permute:", concat.shape)
    if len(concat.shape) == 4:
        concat = concat.squeeze(0)
    # Permute to [codebook, channel, total_samples], then flatten channel and sample into one dim
    reshaped = concat.permute(1, 0, 2).reshape(concat.size(1), -1)
    return reshaped

def compute_pmi_matrices(token_tensors,codebook=None):
    return  [compute_pmi_matrix(token_tensors[idx, :],codebook=codebook)[0] for idx in range(token_tensors.size(0))]

def encode_wav_tokens(file_path, model, processor, return_wav_file=False):
    def load_wav_file(file_path):
        print("loading - ", file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        print("waveform.shape:", waveform.shape)
        if sample_rate != 24000:
            waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
            sample_rate = 24000
        return waveform, sample_rate

    os.makedirs('tokenize_test_data', exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join('tokenize_test_data', base + '.pt')
    
    if os.path.exists(cache_path):
        print("loading - ", cache_path)
        codes = torch.load(cache_path)
        print("codes.shape:", codes.shape)
        if return_wav_file:
            waveform, sample_rate = load_wav_file(file_path)
            print("waveform.shape:", waveform.shape)
            return waveform, sample_rate, codes
        return None, 24000, codes

    waveform, sample_rate = load_wav_file(file_path)
    all_codes = []
    for ch in range(waveform.shape[0]):
        if isinstance(model, CompressionModel):
            # Handle CompressionModel
            x = waveform[ch].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            codes, _ = model.encode(x)
            all_codes.append(codes)
            print("encode.shape:", codes.shape)
        else:
            # Handle Hugging Face EncodecModel
            inputs = processor(raw_audio=waveform[ch].numpy(),
                             sampling_rate=sample_rate,
                             return_tensors="pt")
            codes = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=1.5)
            print("encode.shape:", codes.shape)
            all_codes.append(codes.audio_codes)
    if isinstance(model, CompressionModel):
        combined = torch.cat(all_codes, dim=0)
    else :
        combined = torch.cat(all_codes, dim=1)

    print("Shape of combined:", combined.shape)
    if len(combined.shape) == 4:
        combined = combined.squeeze(0)
    torch.save(combined, cache_path)
    return waveform, sample_rate, combined

def compute_pmi_matrix(token_list, max_distance=2, codebook=None, rho=1.0, sigma=None): #.87
    """
    Computes PMIρ(A,B) using Eq. (1) from crisp_boundaries.pdf:
      P(A,B) = (1/Z) Σ_d w(d) p(A,B; d)
    where w(d) is a Gaussian weighting over distances.
    Returns:
      pmi_matrix (vocab_size x vocab_size), min_val, max_val
    """
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 1024

    # Convert tokens to Tensor on device
    tokens = token_list.to(device) if isinstance(token_list, torch.Tensor) else torch.tensor(token_list, dtype=torch.long, device=device)

    # Prepare distance weights w(d)
    distances = list(range(1, max_distance + 1))
    if sigma is None:
        sigma = float(max_distance) / 2.0
    w = torch.tensor([math.exp(-((d - 1) ** 2) / (2 * sigma ** 2)) for d in distances], device=device)
    w = w / w.sum()  # Normalize weights so sum_d w(d) = 1

    # Accumulate weighted joint probabilities
    joint_probs = torch.zeros((vocab_size, vocab_size), dtype=torch.float64, device=device)
    for i, d in enumerate(distances):
        if tokens.size(0) <= d:
            break
        a, b = tokens[:-d], tokens[d:]
        flat_idx = a * vocab_size + b
        counts_d = torch.bincount(flat_idx, minlength=vocab_size * vocab_size).to(torch.float64).view(vocab_size, vocab_size)
        p_d = counts_d / counts_d.sum()  # p(A,B; d)
        joint_probs += w[i] * p_d

    # Marginals: P(A) and P(B)
    p_A = joint_probs.sum(dim=1)
    p_B = joint_probs.sum(dim=0)

    # Compute PMIρ = log(P(A,B)^ρ / (P(A) P(B)))
    mask = joint_probs > 0
    denom = (p_A.unsqueeze(1) * p_B.unsqueeze(0))
    pmi = torch.zeros_like(joint_probs, dtype=torch.float64)
    pmi[mask] = torch.log(joint_probs[mask].pow(rho) / denom[mask])


    # Compute NPMI = PMI / -log(P(A,B))
    npmi = torch.zeros_like(pmi)
    log_joint = torch.zeros_like(joint_probs)
    log_joint[mask] = -torch.log(joint_probs[mask])
    npmi[mask] = pmi[mask] / log_joint[mask]

    # Modulate PMI by codebook L2 distance, if codebook is available
    # if codebook is not None:
    #     with torch.no_grad():
    #         distances = torch.cdist(codebook, codebook, p=2).to(pmi.device)
    #         normalized = distances / (distances.mean() + 1e-6)
    #         npmi *= normalized

    return npmi, float(npmi.min().item()), float(npmi.max().item())

def plot_pmi_matrix(pmi_matrix, cmap, output_path):
    mat = pmi_matrix.detach().cpu().numpy()

    nonzero_rows = np.any(mat != 0, axis=1)
    nonzero_cols = np.any(mat != 0, axis=0)
    valid = nonzero_rows & nonzero_cols
    mat = mat[np.ix_(valid, valid)]
    # Optional: subset and reorder as before
    # mask_nonzero = np.any(mat != 1.0, axis=1)
    # idx = np.where(mask_nonzero)[0]
    # mat = mat[np.ix_(idx, idx)]

    dists = pdist(mat, metric='euclidean')
    order = leaves_list(linkage(dists, method='average'))
    mat = mat[np.ix_(order, order)]
    # Set up centered normalization

    vmin, vmax = mat.min(), mat.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('PMI')

    ax.set_title('Pointwise Mutual Information Matrix')
    ax.set_xlabel('Next Token')
    ax.set_ylabel('Current Token')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_codebook_matrix(codebook, cmap, output_path):
    import torch.nn.functional as F


    # Compute L2 distance matrix between codebook vectors
    sim_matrix = torch.cdist(codebook, codebook, p=2).detach().cpu().numpy()
    sim_matrix = sim_matrix / (sim_matrix.mean() + 1e-6)

    # Mask near-zero rows/columns
    mask_nonzero = np.any(np.abs(sim_matrix - 1.0) > 1e-3, axis=1)
    idx = np.where(mask_nonzero)[0]
    sim_matrix = sim_matrix[np.ix_(idx, idx)]

    # Reorder using hierarchical clustering
    dists = pdist(sim_matrix, metric='euclidean')
    order = leaves_list(linkage(dists, method='average'))
    sim_matrix = sim_matrix[np.ix_(order, order)]

    # Use a standard colormap without normalization
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('L2 distance')

    ax.set_title('Codebook L2 distance Matrix')
    ax.set_xlabel('Codebook Entry')
    ax.set_ylabel('Codebook Entry')
    plt.savefig(output_path.replace('.png', '.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)




# @lru_cache(maxsize=1)
# def get_model_and_processor(model_name="facebook/encodec_24khz"):
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = EncodecModel.from_pretrained(model_name)
#     codebook = model.quantizer.layers[0].codebook.embed  # [codebook_size, embedding_dim]
#     return model, processor, codebook

@lru_cache(maxsize=1)
def get_model_and_processor(model_path="facebook/encodec_24khz"):
    # if model_path.endswith(".th"):
    #     # Load checkpoint
    #     checkpoint = torch.load(model_path, map_location="cpu")
    #     if "model" in checkpoint:
    #         state_dict = checkpoint["model"]
    #     elif "best_state" in checkpoint:
    #         state_dict = checkpoint["best_state"]
    #     else:
    #         raise ValueError("Checkpoint format not recognized. Expected 'model' or 'best_state' key.")
        
    #     # Load base model from transformers
    #     from transformers import EncodecModel
    #     model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    #     # Load state dict with strict=False to handle missing keys
    #     model.load_state_dict(state_dict, strict=False)
    #     processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    if model_path.endswith(".bin") or os.path.isfile(model_path):
        model = CompressionModel.get_pretrained(model_path)
        processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        codebook = model.quantizer.vq.layers[0]._codebook.embed
    else:
        from transformers import EncodecModel
        model = EncodecModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        codebook = model.quantizer.layers[0].codebook.embed
    return model, processor, codebook

def plot_waveform_and_pmi_windowed(channel_waveform, sample_rate, token_ch, pmi_matrix, cmap,
                                 output_prefix, channel_idx, window_idx, window_duration=30.0, max_distance=2,codebook=None,rho=1.0):
    
    mat = pmi_matrix.detach().cpu().numpy()

    # Compute consistent amplitude y-axis limits from full channel waveform
    amp_min = channel_waveform.min()
    amp_max = channel_waveform.max()
    total_samples = channel_waveform.shape[-1]
    total_duration = total_samples / sample_rate
    
    start_time   = window_idx * window_duration
    end_time     = min(start_time + window_duration, total_duration)
    start_sample = int(start_time * sample_rate)
    end_sample   = int(end_time   * sample_rate)
    
    wf_win = channel_waveform[start_sample:end_sample]
    dur_win = (end_sample - start_sample) / sample_rate
    t_audio = np.linspace(0, dur_win, wf_win.shape[-1])

    T = len(token_ch)
    frame_duration = total_duration / T
    start_idx = max(0, int(np.floor(start_time / frame_duration - 0.5)) + 1)
    end_idx   = min(T, int(np.ceil( end_time   / frame_duration - 0.5)) + 1)

    mat = compute_pmi_matrix(token_ch[start_idx:end_idx],codebook=codebook,rho=rho)[0]
    if torch.is_tensor(mat):
        mat = mat.cpu().numpy()

    #compute an array of pmi_win values for 1-5 distances, make sure you don't go out of range on the left side
    
    pmi_win_by_distance = [[ mat[token_ch[i - d], token_ch[i]] if i - d >= 0 else 0 for d in range(1, max_distance)] for i in range(start_idx, end_idx) ]
    pmi_win             = np.mean(pmi_win_by_distance, axis=1)

    # pmi_win   = np.array([ mat[token_ch[i - 1], token_ch[i]] for i in range(start_idx, end_idx) ])
    t_pmi_win = np.array([ (i + 0.5) * frame_duration - start_time for i in range(start_idx, end_idx) ])
    
    # Compute consistent normalization
    vmin, vmax = mat.min(), mat.max()
    # print("vmin, vmax:", vmin, vmax)
    try:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    except ValueError:
        # print("Warning: Invalid TwoSlopeNorm range, falling back to default colormap without normalization.")
        norm = None


    # Prepare background image of PMI values for full window
    bg2 = np.vstack([pmi_win, pmi_win])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 6))
    if norm is not None:
        ax1.imshow(bg2, cmap=cmap, norm=norm, aspect='auto', extent=[0, dur_win, -.5, .5])
    else:
        ax1.imshow(bg2, cmap='viridis', aspect='auto', extent=[0, dur_win, -.5, .5])


    ax1.imshow(bg2, cmap=cmap, norm=norm, aspect='auto',extent=[0, dur_win, -.5, .5])

    ax1.plot(t_audio, wf_win, linewidth=0.5)
    # Set consistent y-axis limits for amplitude
    ax1.set_ylim(amp_min, amp_max)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Ch{channel_idx} Waveform (window {window_idx*window_duration:.0f}-{end_time:.0f}s)")

    # Compute L2 distance between adjacent codebook entries in window
    with torch.no_grad():
        l2_sim_win = []
        for i in range(start_idx + 1, end_idx):
            tok_prev = token_ch[i - 1]
            tok_curr = token_ch[i]
            sim = torch.dist(codebook[tok_prev], codebook[tok_curr], p=2).item()
            l2_sim_win.append(sim)
        t_sim_win = np.array([ (i + 0.5) * frame_duration - start_time for i in range(start_idx + 1, end_idx) ])


    if norm is not None:    
        ax2.imshow(bg2, cmap=cmap, norm=norm, aspect='auto', extent=[0, dur_win, 0.0, max(l2_sim_win) * 1.1 if l2_sim_win else 1.0], zorder=0)
    else:
        ax2.imshow(bg2, cmap='viridis', aspect='auto', extent=[0, dur_win, 0.0, max(l2_sim_win) * 1.1 if l2_sim_win else 1.0], zorder=0)
    # ax2.imshow(bg2, cmap=cmap, norm=norm, aspect='auto', extent=[0, dur_win, -.5, .5])
    ax2.plot(t_sim_win, l2_sim_win, linewidth=0.5, color='black', zorder=1)
    ax2.set_ylim(0.0, max(l2_sim_win) * 1.1 if l2_sim_win else 1.0)
    ax2.set_ylabel("L2 distance")
    ax2.set_title(f"Ch{channel_idx} Codebook L2 distance")

    ax3.plot(t_pmi_win, pmi_win, linewidth=0.5)
    # Set consistent y-axis limits for PMI
    ax3.set_ylim(vmin, vmax)
    ax3.set_ylabel("PMI")
    ax3.set_xlabel("Time in window (s)")
    ax3.set_title(f"Ch{channel_idx} PMI (window {window_idx*window_duration:.0f}-{end_time:.0f}s)")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ch{channel_idx}_win{window_idx}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_pngs(file_path, waveform, sample_rate, tokens , pmi_matrices, codebook,rho=1.0):
    vq_layer = 0
    # Single red-white-blue diverging colormap: red=min, white=0, blue=max
    cmap = LinearSegmentedColormap.from_list('red_white_blue', ['red', 'white', 'blue'], N=256 )
    
    outdir = os.path.join('tokenize_test_data')
    os.makedirs(outdir, exist_ok=True)
    plot_pmi_matrix(pmi_matrices[vq_layer], cmap,os.path.join(outdir, os.path.basename(file_path) + '_pmi.png'))
    plot_codebook_matrix(codebook, cmap,os.path.join(outdir, os.path.basename(file_path) + '_codebook.png'))
    
    num_windows = int((waveform.shape[1] / sample_rate) // 30)

    print("waveform.shape:", waveform.shape)
    print("tokens.shape:", tokens.shape)
    print("codebook.shape:", codebook.shape)

    for ch in range(waveform.shape[0]):
        
        token_ch    = tokens[ch,vq_layer,:].detach().cpu().numpy().tolist()
        waveform_ch = waveform[ch].numpy()
        for w in range(num_windows):
            plot_waveform_and_pmi_windowed(
                waveform_ch, sample_rate, token_ch,
                pmi_matrices[vq_layer], cmap,
                os.path.join(outdir, os.path.basename(file_path)),
                ch, w, window_duration=30.0, codebook=codebook,rho=rho
            )

def main():
    parser = argparse.ArgumentParser(description="Process 2-channel WAV files through EnCodec model")
    parser.add_argument('wav_files', nargs='+', help='Input WAV files')
    parser.add_argument('--pmi', type=str, default=None,
                       help='Path to existing PMI pickle to load. If omitted, PMI will be computed and saved to tokenize_test_data/composite_pmi.pkl.')
    parser.add_argument('--pngs', action='store_true', default=False, help='Generate PNG outputs using PMI data')
    parser.add_argument('--model', type=str, default="facebook/encodec_24khz",
                       help='Hugging Face model name to use (default: facebook/encodec_24khz)')

    args = parser.parse_args()
    default_pmi_path = os.path.join('tokenize_test_data', 'composite_pmi.pkl')
    model, processor, codebook = get_model_and_processor(args.model)

    pmi_matrices = (load_multi_pmi_data(args.pmi) if args.pmi else compute_pmi_matrices(load_all_wav_files(args, model, processor),codebook=codebook))
    if not args.pmi:
        save_multi_pmi_data(default_pmi_path, pmi_matrices)

    if args.pngs:
        f = args.wav_files[0]
        waveform, sample_rate, tokens = encode_wav_tokens(f, model, processor, return_wav_file=True)

        while True:
            val = input("Enter rho value (or 'q' to quit): ")
            if val.lower() == 'q':
                break
            try:
                rho_val = float(val)
            except ValueError:
                print("Invalid rho value, please enter a number or 'q' to quit.")
                continue
            generate_pngs(f, waveform, sample_rate, tokens, pmi_matrices, codebook,rho=rho_val)

if __name__ == "__main__":
    main()