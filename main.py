# ============================== main.py (CPU/CUDA/MPS) ==============================
import os
import os.path
import argparse
import time
import numpy as np
import torch

from utils import *          # load_dataset, graph_nsgt, normalize_adj_tensor, calc_distance, normalize_score
from model import Model      # model class


def get_best_device():
    # Prefer Apple MPS on Apple Silicon, else CUDA, else CPU
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()  # changed (from eha)
    if mps_ok:
        return torch.device("mps"), "mps"  # changed (from eha)
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"  # changed (from eha)
    return torch.device("cpu"), "cpu"  # changed (from eha)


device, device_name = get_best_device()  # changed (from eha)

# Where to cache distances / outputs
save_path = os.path.join(os.path.dirname(__file__), "data")  # changed (from eha)


def main(args):
    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load and preprocess data
    raw_features, features, adj1, adj2, ano_label, raw_adj1, raw_adj2, config = load_dataset(args)

    # Move tensors to the selected device (works for cpu/cuda/mps)
    features = features.to(device)         # changed (from eha)
    raw_features = raw_features.to(device) # changed (from eha)
    raw_adj1 = raw_adj1.to(device)         # changed (from eha)
    if raw_adj2 is not None:
        raw_adj2 = raw_adj2.to(device)     # changed (from eha)

    # Build models & optimizers (use config['cutting'])
    optimiser_list, model_list = [], []
    for _ in range(config['cutting']):
        model = Model(config['ft_size'], args.embedding_dim, 'prelu', args.readout, config).to(device)  # changed (from eha)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimiser_list.append(optimiser)
        model_list.append(model)

    start = time.time()
    total_epoch = args.num_epoch * config['cutting']

    # Initial cut-adj (view 1)
    all_cut_adj1 = torch.cat([raw_adj1])

    # Distance cache (view 1)
    os.makedirs(save_path, exist_ok=True)
    distance1_file = os.path.join(save_path, f"{args.dataset}_distance1.npy")
    if os.path.exists(distance1_file):
        dis_array1 = torch.tensor(np.load(distance1_file), dtype=torch.float32, device=device)          # changed (from eha)
    else:
        dis_array1 = calc_distance(raw_adj1[0, :, :], raw_features[0, :, :])                             # changed (from eha)
        np.save(distance1_file, dis_array1.detach().cpu().numpy())

    # Optional second view
    if raw_adj2 is not None:
        all_cut_adj2 = torch.cat([raw_adj2])
        dist2_path = os.path.join(save_path, f"{args.dataset}_distance2.npy")
        if os.path.exists(dist2_path):
            dis_array2 = torch.tensor(np.load(dist2_path), dtype=torch.float32, device=device)          # changed (from eha)
        else:
            dis_array2 = calc_distance(raw_adj2[0, :, :], raw_features[0, :, :])                         # changed (from eha)
            np.save(dist2_path, dis_array2.detach().cpu().numpy())
    else:
        all_cut_adj2 = None
        dis_array2 = None

    index = 0
    message_mean_list = []

    # Per-cut training
    for n_cut in range(config['cutting']):
        message_list = []
        optimiser_list[index].zero_grad()
        model_list[index].train()

        # Ensure distance tensors match adjacency device
        dis_array1 = dis_array1.to(all_cut_adj1.device)                                                  # changed (from eha)

        # Cut adj for view 1
        cut_adj1 = graph_nsgt(dis_array1, all_cut_adj1[0, :, :])                                         # changed (from eha)
        cut_adj1 = cut_adj1.unsqueeze(0)
        adj_norm1 = normalize_adj_tensor(cut_adj1, args.dataset)

        # Cut adj for view 2
        if all_cut_adj2 is not None:
            dis_array2 = dis_array2.to(all_cut_adj2.device)                                              # changed (from eha)
            cut_adj2 = graph_nsgt(dis_array2, all_cut_adj2[0, :, :])                                     # changed (from eha)
            cut_adj2 = cut_adj2.unsqueeze(0)
            adj_norm2 = normalize_adj_tensor(cut_adj2, args.dataset)
        else:
            adj_norm2 = None

        # Epoch loop
        for epoch in range(args.num_epoch):
            optimiser_list[index].zero_grad()
            node_emb, cluster_sim, loss = model_list[index].forward(
                features[0], adj_norm1, raw_adj1, adj_norm2, raw_adj2
            )
            loss.backward()
            optimiser_list[index].step()

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{n_cut * args.num_epoch + epoch + 1}/{total_epoch}], "
                    f"device={device_name}, time: {time.time() - start:.4f}, Loss: {loss.item():.4f}"
                )

        # Scoring
        message_sum = (
            model_list[index].inference(node_emb, cluster_sim)
            + model_list[index].view_consistency(features[0], adj_norm1, adj_norm2)
        )
        message_list.append(torch.unsqueeze(message_sum.detach().cpu(), 0))

        # Update bases for next cut
        all_cut_adj1[0, :, :] = torch.squeeze(cut_adj1)
        if all_cut_adj2 is not None:
            all_cut_adj2[0, :, :] = torch.squeeze(cut_adj2)

        index += 1

        # Aggregate & evaluate
        message_list = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(torch.unsqueeze(message_list, 0))
        message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
        message_mean = np.array(message_mean_cut.detach())
        score = 1 - normalize_score(message_mean)
        model_list[index - 1].evaluation(score, ano_label)

    print("Total time (s):", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-Aware Graph Anomaly Detection (CARE-demo)")
    parser.add_argument('--dataset', type=str, default='Amazon', help='Amazon | BlogCatalog | imdb | dblp')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--readout', type=str, default='avg')      # max | min | avg | weighted_sum
    parser.add_argument('--cutting', type=int, default=25)         # config['cutting'] is used internally
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--clusters', type=int, default=10)

    args = parser.parse_args()
    print(f"Using device: {device_name}")  # changed (from eha)
    print('Dataset:', args.dataset)
    main(args)
# ==============================================================================
