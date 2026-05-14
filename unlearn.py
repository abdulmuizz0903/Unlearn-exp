import torch
def profile_activations(model, forget_loader, device):
    model.eval()
    all_activations = []
    with torch.no_grad():
        for data, _ in forget_loader:
            data = data.to(device)
            _ = model(data)
            all_activations.append(model.hidden_activations)
    all_activations = torch.cat(all_activations, dim=0)
    return torch.mean(all_activations, dim=0)

def unlearn(model, forget_loader, device, penalty_scale=0.1, threshold_percentile=90, iter_times=5):
    for i in range(iter_times):
        print(f'Unlearning iteration {i+1}/{iter_times}')
        mean_activations = profile_activations(model, forget_loader, device)
        threshold = torch.quantile(mean_activations, threshold_percentile / 100.0)
        highly_active_mask = mean_activations >= threshold
        
        with torch.no_grad():
            model.targeted_layer.weight[highly_active_mask] *= penalty_scale
            model.targeted_layer.bias[highly_active_mask] *= penalty_scale
            model.output_layer.weight[:, highly_active_mask] *= penalty_scale
            
    return model
