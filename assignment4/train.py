from tqdm.notebook import tqdm
import torch
from losses import PSNRLoss
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


def train_epoch(model, train_loader, criterion, optimizer, device='cpu:0'):
    
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, leave=False, ncols=700)
    for step, batch in (enumerate(pbar)):

        data = [el.to(device) for el in batch]
        X, y = data[0], data[1:]

        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_description(f'Training: Loss: {loss.item():.3f}', f'\tSamples:{(step+1) * len(X):>10}/{len(train_loader.dataset)}')
        
    return train_loss / len(train_loader) 


def eval_model(model, val_loader, criterion, device='cpu:0'):

    model.eval()
    
    val_loss = 0
    psnr = []

    pbar = tqdm(val_loader, leave=False, ncols=700)

    for step, batch in enumerate(pbar):
        
        data = [el.to(device) for el in batch]
        X, y = data[0], data[1:]

        outputs = model(X)
        loss = criterion(outputs, y)
        val_loss += loss.item()
#         psnr.append(PSNRLoss(outputs, y, validation=True).item())
        psnr.append([el.item() for el in criterion(outputs, y, validation=True)])
        
        pbar.set_description(f'Validation: Samples:{(step+1) * len(X):>10} / {len(val_loader.dataset)}')
        
    return val_loss / len(val_loader), psnr


def train(model, tr_loader, val_loader, criterion, optimizer, device, num_epochs=10, return_log=False, plot_logs=True):
    print('Start training')
    
    if return_log or plot_logs:
        training_log = []

    if torch.has_cuda:
        model.to(device)

    pbar = tqdm(range(num_epochs), ncols=700)
    
    for epoch in pbar:
        pbar.set_description(f'Epoch: {epoch + 1:>2}/{num_epochs}')
        tr_loss = train_epoch(model, tr_loader, criterion, optimizer, device=device)
        val_loss, psnr = eval_model(model, val_loader, criterion, device=device)
        
#         print(f'Epoch: {epoch + 1:>2} \t validation loss: {val_loss:.4f}')

#         if return_log:
#             training_log.append({
#                 'epoch': epoch,
#                 'train_loss': tr_loss,
#                 'validation_loss': val_loss,
#                 'psnr': [sum(x) / len(val_loader.dataset) for x in zip(*psnr)]
#                 })
        if return_log or plot_logs:
            training_log.append((epoch+1,
                                 tr_loss,
                                 val_loss,
                                 [sum(x) / len(val_loader) for x in zip(*psnr)], 
                                 
                                ))
        
    print('Finished training')
    
    if plot_logs:
        plot_results(training_log)
        
    if return_log:
        return training_log
    
    
def plot_results(log):
    epoch, train_loss, val_loss, psnr = zip(*log)
    psnr = list(zip(*psnr))

    psnr_labels = ['medium output', 'large output']
    psnr_colors = ['b', 'g']
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_loss, label='train')
    plt.plot(epoch, val_loss, label='validation')
    plt.ylabel('PSNRLoss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, p in enumerate(psnr):
        plt.plot(epoch, p, label=psnr_labels[i], c=psnr_colors[i])
    plt.ylim(bottom=15, top=35)
    plt.ylabel('PSNR (dB)')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    
def show_examples(examples, model, device='cpu:0', title=''):
    
    X, y = examples[0], examples[1:]
    model.eval()
    output = model(X.to(device))
    
    if isinstance(output, torch.Tensor):
            output = [output]

    cols = ['X', 'y_med', 'output_med', 'y_large', 'output_large']
    fig, axes = plt.subplots(nrows=X.size(0), ncols=len(output) * 2 + 1, 
                             figsize=(6*len(output), 12))
    
    for i, row_axes in enumerate(axes):
        
        row_axes[0].imshow(ToPILImage()(X[i]))
        
        for j, out in enumerate(output):
            row_axes[2*j + 1].imshow(ToPILImage()(y[j][i]))
            row_axes[2*j + 2].imshow(ToPILImage()(out[i].cpu()))

        [ax.set_axis_off() for ax in row_axes]

        if i == 0:
            for ax, col in zip(row_axes, cols):
                ax.set_title(col)
    
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
