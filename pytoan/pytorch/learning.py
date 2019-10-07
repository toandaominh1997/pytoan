
import torch
import os
import pandas as pd 
import time 
from tqdm import tqdm_notebook as tqdm
class Learning(object):
    def __init__(self,
            model,
            criterion,
            metric_ftns,
            optimizer,
            device,
            num_epoch,
            scheduler,
            grad_clipping,
            grad_accumulation_steps,
            early_stopping,
            validation_frequency,
            save_period,
            checkpoint_dir,
            resume_path):
        self.device, device_ids = self._prepare_device(device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.num_epoch = num_epoch 
        self.scheduler = scheduler
        self.grad_clipping = grad_clipping
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping = early_stopping
        self.validation_frequency =validation_frequency
        self.save_period = save_period
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_score = 0
        if resume_path is not None:
            self._resume_checkpoint(resume_path)
        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        
    def train(self, train_dataloader, valid_dataloader):
        for epoch in range(self.start_epoch, self.num_epoch+1):
            print("{} epoch: \t start training....".format(epoch))
            start = time.time()
            train_result  = self._train_epoch(train_dataloader)
            train_result.update({'time': time.time()-start})
            
            for key, value in train_result.items():
                print('    {:15s}: {}'.format(str(key), value))

            if (epoch+1) % self.validation_frequency!=0:
                print("skip validation....")
                continue
            print('{} epoch: \t start validation....'.format(epoch))
            start = time.time()
            valid_result = self._valid_epoch(valid_dataloader)
            valid_result.update({'time': time.time() - start})
            score = -1
            for key, value in valid_result.items():
                if 'score' in key and score ==-1:
                    score = value 
                print('   {:15s}: {}'.format(str(key), value))
            
            self.post_processing(score, epoch)
            if epoch - self.best_epoch > self.early_stopping:
                print('EARLY STOPPING')
                break
    def _train_epoch(self, data_loader):
        self.model.train()
        self.optimizer.zero_grad()
        self.train_metrics.reset()
        for idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.train_metrics.update('loss', loss.item())
            if (idx+1) % self.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                self.optimizer.step()
                self.optimizer.zero_grad()
        return self.train_metrics.result()
    def _valid_epoch(self, data_loader):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
            return self.valid_metrics.result()
    def post_processing(self, score, epoch):
        best = False
        if score > self.best_score:
            self.best_score = score 
            self.best_epoch = epoch 
            best = True
            print("best model: {} epoch - {:.5}".format(epoch, score))
        if best==True or (self.save_period>=0 and epoch % self.save_period == 0):
            self._save_checkpoint(epoch = epoch, save_best = best)
        
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()
    
    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.get_state_dict(self.model),
            'best_score': self.best_score
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
    
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    
    @staticmethod
    def _prepare_device(device):
        if type(device)==int:
            n_gpu_use = device
        else:
            n_gpu_use = len(device)
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if type(device)==int:
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            list_ids = list(range(n_gpu_use))
        elif len(device) == 1:
            list_ids = device
            if device[0] >= 0 and device[0] < n_gpu:    
                device = torch.device('cuda:{}'.format(device[0]))
            else:
                device = torch.device('cuda:0')
        else:
            list_ids = device
            device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')
            
        return device, list_ids


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)