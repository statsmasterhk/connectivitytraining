import os
import copy, time
import torch

from util.logger import get_logger
from torch.utils.tensorboard import SummaryWriter

### Modified from util/train_helper.py train_model function
def get_default_param():
    time_str = '{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    new_dict = {}
    new_dict["old_model_path"] = "" # existing model path
    new_dict["train_batch_size"] = 128
    new_dict["val_batch_size"] = 128

    ## For Tensorboard SummaryWriter
    new_dict["log_dir"] = "train_log"
    new_dict["project_name"] = f"train_{time_str}"
    new_dict["model_base_dir"] = "./res/" # Output directory of the model

    new_dict["log_interval"] = 1
    new_dict["save_interval"] = 5
    new_dict["val_interval"] = 3
    new_dict["max_epoch"] = 100
    new_dict['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_dict['time_str'] = time_str

    return new_dict

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                param_dict = {}):

    ## For Tensorboard SummaryWriter
    log_dir = param_dict["log_dir"]
    project_name = param_dict["project_name"]
    time_str = param_dict["time_str"]
    proj_log_dir = os.path.join(log_dir,project_name)
    logger = get_logger(proj_log_dir,time_str)
    writer = SummaryWriter(proj_log_dir)

    # Setup
    model_base_dir = param_dict["model_base_dir"]
    model_output_dir = os.path.join(model_base_dir,project_name)
    if not os.path.exists(model_output_dir): os.makedirs(model_output_dir)

    ## train
    logger.info('Start training ...')
    device = param_dict['device']

    # Load old model path if exists
    old_model_path = param_dict["old_model_path"]
    if os.path.exists(old_model_path):
        checkpoint = torch.load(old_model_path)
        model.load_state_dict(checkpoint)
        logger.info("=> Loaded checkpoint '{}'".format(old_model_path))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    max_epochs = param_dict["max_epoch"]

    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        to_run = []
        if epoch%param_dict["val_interval"] == 0:
            to_run.append("val")
        to_run.append("train")

        # phase for train dataset/validation dataset
        for phase in to_run:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            t_start = time.time()
            # loop over all data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.flatten(labels)
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Back Propagate 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistic
                no_of_correct = torch.sum(preds == labels.detach())
                no_of_samples = len(labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.detach())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)
            time_interval = time.time() - t_start

            # Log the loss/acc/time out for tensorboard to visualise the training progress
            if epoch%param_dict["log_interval"] == 0:
                if phase == "train":
                    writer.add_scalar("train/loss", epoch_loss, epoch)
                    writer.add_scalar("train/acc", epoch_acc, epoch)
                    writer.add_scalar("train/time", time_interval, epoch)
                elif phase == "val":
                    writer.add_scalar("val/loss", epoch_loss, epoch)
                    writer.add_scalar("val/acc", epoch_acc, epoch)
                    writer.add_scalar("val/time", time_interval, epoch)
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                           epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                logger.info(f'saving new best model, val_acc: {epoch_acc}')
                torch.save(model.state_dict(), f'{model_output_dir}/model_best_{epoch}.pkl')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch%param_dict["log_interval"] == 0:
            writer.flush()

        if epoch%param_dict["save_interval"] == 0:
            logger.info('saving trained model')
            torch.save(model.state_dict(), f'{model_output_dir}/model_{epoch}.pkl')

    return model
