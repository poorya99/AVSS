import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import argparse
import time
from tqdm import tqdm

# If your environment supports apex:
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

# If you use TensorBoard:
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------
# 1. YOUR EXISTING SOLVER CLASS (Unmodified)
#    Make sure the following code is exactly
#    as you provided:
# -------------------------------------------------
class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, model2, optimizer2, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args
        self.amp = amp
        self.saved = 0
        self.MSE = nn.MSELoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        # apex amp initialization
        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                         opt_level=args.opt_level,
                                                         patch_torch_functions=args.patch_torch_functions)
        self.model2, self.optimizer2 = model2, optimizer2
        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict.pt' % self.args.continue_from, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.prev_val_loss = checkpoint['prev_val_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            self.model2.load_state_dict(checkpoint['model2'])
            self.prev_val_loss2 = checkpoint['prev_val_loss2']
            self.best_val_loss2 = checkpoint['best_val_loss2']
            self.val_no_impv2 = checkpoint['val_no_impv2']

            if self.print: 
                print("Resume training from epoch: {}".format(self.start_epoch))
        else:
            self.prev_val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_no_impv = 0

            self.prev_val_loss2 = float("inf")
            self.best_val_loss2 = float("inf")
            self.val_no_impv2 = 0

            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        
        for epoch in range(self.args.epochs):
            if self.args.distributed: 
                self.args.train_sampler.set_epoch(epoch)

            # Train
            self.model.train()
            self.model2.train()
            start = time.time()
            tr_loss, tr_loss2 = self._run_one_epoch(data_loader = self.train_data)
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print: 
                print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.6f} | valid loss2 {3:.6f}'.format(
                        epoch, time.time() - start, reduced_tr_loss, tr_loss2))

            # Validation
            self.model.eval()
            self.model2.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_loss2 = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)


            if self.print: 
                print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.6f} | valid loss2 {3:.6f}'.format(
                          epoch, time.time() - start, reduced_val_loss, val_loss2))

            # Early stopping / no improvement logic
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 20:
                    if self.print: print("No imporvement for 20 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            if self.val_no_impv == 6:
                self.halving = True

            # Half the learning rate
            self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * 0.99491024759420038251406254302534
                self.optimizer.load_state_dict(optim_state)
                if self.print: 
                    print('Learning rate adjusted to: {lr:.6f}'.format(
                        lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = reduced_val_loss

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)


                # Save model if improvement
                if reduced_val_loss < self.best_val_loss:
                    self.best_val_loss = reduced_val_loss
                #     checkpoint = {'model': self.model.state_dict(),
                #                   'optimizer': self.optimizer.state_dict(),
                #                   'amp': self.amp.state_dict(),
                #                   'epoch': epoch+1,
                #                   'prev_val_loss': self.prev_val_loss,
                #                   'best_val_loss': self.best_val_loss,
                #                   'val_no_impv': self.val_no_impv,
                                  
                #                   'model2': self.model2.state_dict(),
                #                   'optimizer2': self.optimizer2.state_dict(),
                #                   'prev_val_loss2': self.prev_val_loss2,
                #                   'best_val_loss2': self.best_val_loss2,
                #                   'val_no_impv2': self.val_no_impv2 
                #                   }
                #     self.saved = 1
                #     torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict.pt")
                    print("Found new best model, dict saved")

                # if (not self.saved and epoch==self.args.epochs-1):  
                self.best_val_loss = reduced_val_loss
                checkpoint = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'amp': self.amp.state_dict(),
                                'epoch': epoch+1,
                                'prev_val_loss': self.prev_val_loss,
                                'best_val_loss': self.best_val_loss,
                                'val_no_impv': self.val_no_impv,
                                
                                'model2': self.model2.state_dict(),
                                'optimizer2': self.optimizer2.state_dict(),
                                'prev_val_loss2': self.prev_val_loss2,
                                'best_val_loss2': self.best_val_loss2,
                                'val_no_impv2': self.val_no_impv2 
                                }
                self.saved = 1
                torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict.pt")
                print("no improvement but saved, dict saved")


    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        total_loss2 = 0
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        sum_t = 0
        for i, (audio, visual, face_frame_batch, speaker_embedding_batch) in enumerate(tqdm(data_loader)):
            audio = audio.cuda().squeeze(0).float()
            visual = visual.cuda().squeeze(0).unsqueeze(1).float()
            face_frame_batch = face_frame_batch.cuda().squeeze(0)
            speaker_embedding_batch = speaker_embedding_batch.cuda().squeeze(0)
            if ( torch.isnan(audio).any().item()):
              continue;
            batch, _, fr_time, _ = visual.shape
            sum_t = sum_t + fr_time
            est_audio = self.model(visual)
            est_spk_emb = self.model2(face_frame_batch)
            # print(est_audio.shape)
            loss = self.MSE(est_audio, audio)
            
            loss2 = self.MSE(est_spk_emb, speaker_embedding_batch)
  
            if state == 'train':
                with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                               self.args.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()


                loss2.backward()
                self.optimizer2.step()
                self.optimizer2.zero_grad()

            total_loss += loss.data
            total_loss2 += loss2.data
            
        return total_loss / sum_t, 1000*total_loss2 / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

