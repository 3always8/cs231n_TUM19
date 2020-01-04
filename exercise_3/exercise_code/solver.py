import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        """
        n = 0

        for i in np.arange(iter_per_epoch) :
        
            loss = 0.0
        
            for input_data, target in train_loader:

                n += 1
                optim.zero_grad()
                input_data, target = input_data.to(device), target.to(device)
                
                func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
                output = model.forward(input_data)
                loss = func(output, target)
                self.train_loss_history.append(loss.data)

                _, pred = torch.max(output, dim=1)
                
                targets_mask = target >= 0
    
                train_acc = np.mean(
                    (pred == target)[targets_mask].data.cpu().numpy()
                )
                loss.backward()
                optim.step()

                if n % log_nth == 0 :
                    print("%dth    loss:%f    train acc:%f" %(n, loss.item(), train_acc))
            self.train_acc_history.append(train_acc)
            mean_loss = np.mean(self.train_loss_history[-6:-1])
            print("mean_loss:",mean_loss)
        """

        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = inputs.to(device), targets.to(device)

                optim.zero_grad()

#                outputs = model(inputs.squeeze(1).permute(1, 0, 2).float())
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.detach().cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           train_loss))

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0 e.g. for segmentation
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[
                                targets_mask].detach().cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                        num_epochs, train_acc, train_loss))

        
        
        
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
