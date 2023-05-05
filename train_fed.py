import torch
import numpy as np  
import time
    
def train_fed(device, num_epochs, top_percent, train_loaders, test_loader, optimizers, criterions, local_models, global_model, local_buffer=True, global_buffer=True):
    num_models = len(local_models)
    t0 = time.time()
    
    local_cumulative_gradients = [{} for i in range(num_models)]
    global_cumulative_gradient = {}
    test_acc = []
    for epoch in range(num_epochs):
        for worker_id in range(num_models):
            local_models[worker_id], local_cumulative_gradients[worker_id] = train_local_model(device, top_percent, train_loaders[worker_id], \
                        optimizers[worker_id], criterions[worker_id], local_models[worker_id], global_model, local_cumulative_gradients[worker_id], local_buffer)

        # Federated Averaging
        with torch.no_grad():
            for name, global_param in global_model.named_parameters():
                local_weights = [local_models[worker_id].state_dict()[name] for worker_id in range(num_models)]
                avg_weights = sum(local_weights) / num_models
                weight_diff = avg_weights - global_param.data
                if global_buffer:
                    if name not in global_cumulative_gradient:
                        global_cumulative_gradient[name] = torch.zeros_like(weight_diff)
                    global_cumulative_gradient[name] += weight_diff
                    top_gradients = filter_top_gradients(global_cumulative_gradient[name], top_percent)
                    updated_mask = top_gradients != 0
                    weight_diff = top_gradients.clone()
                    global_cumulative_gradient[name] = global_cumulative_gradient[name] * (~updated_mask)
                else:
                    weight_diff = filter_top_gradients(weight_diff, top_percent)
                global_param.copy_(weight_diff + global_param)
        
        print('Epoch %d - Time %d s' % (epoch + 1, time.time()-t0))
        acc = test(device, test_loader, global_model)
        test_acc.append(acc)

    print('Finished Federated Training')
    del local_models, global_model, local_cumulative_gradients, global_cumulative_gradient
    torch.cuda.empty_cache()
    return test_acc
    
    
def train_local_model(device, top_percent, train_loader, optimizer, criterion, local_model, global_model, cumulative_gradient, buffer=True):
    local_model.train()
    local_model.load_state_dict(global_model.state_dict())
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = local_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() 
        
    with torch.no_grad():
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                orginal_param = global_model.state_dict()[name]
                local_gradient = param.data - orginal_param.data
                if buffer:
                    if name not in cumulative_gradient:
                        cumulative_gradient[name] = torch.zeros_like(orginal_param.data)
                    cumulative_gradient[name] += local_gradient
                    top_gradients = filter_top_gradients(cumulative_gradient[name], top_percent)
                    updated_mask = top_gradients != 0
                    param.copy_(top_gradients + orginal_param.data)
                    cumulative_gradient[name] = cumulative_gradient[name] * (~updated_mask)
                else:
                    top_gradients = filter_top_gradients(local_gradient, top_percent)
                    param.copy_(top_gradients + orginal_param.data)
        
    return local_model, cumulative_gradient

# Function to filter top x% gradients
def filter_top_gradients(grads, top_percent):
    flat_grads = torch.abs(grads.flatten())
    num_to_keep = int(np.ceil(top_percent * flat_grads.numel()))
    threshold = flat_grads.topk(num_to_keep).values[-1]
    mask = torch.abs(grads) >= threshold
    return grads * mask

# Testing the model
def test(device, test_loader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = (100 * correct / total)
    print('Accuracy of the network on the {num:d} test images: {acc:.2f} %'.format(num = len(test_loader.dataset), acc = acc))
    return acc
    


